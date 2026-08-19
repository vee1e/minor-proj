"""Bench pipeline: acquire formulas (downloads + generators), write CNF files,
and emit the manifest parquet/json consumed by features/data stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from satsel.bench import download, generators
from satsel.core.dimacs import parse_cnf

log = logging.getLogger(__name__)

# Plan §8.1 targets. crypto is the appendix, hardness-capped downstream.
DEFAULT_COUNTS: dict[str, int] = {
    "random3sat": 400,
    "random3sat_large": 400,
    "colouring": 300,
    "planning": 200,
    "industrial": 400,
    "scheduling": 300,
    "ood_scheduling": 200,
    "crypto": 100,
}


def acquire_formulas(cfg: dict, counts: dict[str, int] | None = None) -> dict[str, list]:
    """Return {family: [Cnf, ...]} reaching `counts` per family.

    Downloads run first (best-effort); generators fill any shortfall so every
    family hits its target.
    """
    counts = counts or DEFAULT_COUNTS
    raw = Path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    https_only = bool(cfg.get("download_https_only", True))

    downloaded: dict[str, list] = {fam: [] for fam in counts}
    try:
        for data, fam, name in download.download_satlib(counts, raw, https_only=https_only):
            downloaded.setdefault(fam, []).append((data, fam, name))
    except Exception as exc:  # noqa: BLE001
        log.warning("SATLIB download failed: %s", exc)
    try:
        for data, fam, name in download.download_gbd(counts, raw, https_only=https_only):
            downloaded.setdefault(fam, []).append((data, fam, name))
    except Exception as exc:  # noqa: BLE001
        log.warning("GBD download failed: %s", exc)

    # Families covered by the generators.
    gen_specs = {
        "random3sat": (generators.gen_random3sat_batch, {}),
        "random3sat_large": (generators.gen_random3sat_large_batch, {}),
        "colouring": (generators.gen_colouring_batch, {}),
        "planning": (generators.gen_planning_batch, {}),
        "industrial": (generators.gen_industrial_batch, {}),
        "scheduling": (generators.gen_scheduling_batch, {}),
        "ood_scheduling": (generators.gen_scheduling_batch, {"family": "ood_scheduling"}),
        "crypto": (generators.gen_crypto_batch, {}),
    }

    out: dict[str, list] = {}
    base_seed = int(cfg.get("seeds", {}).get("generators", 2026))
    for fam, target in counts.items():
        pool = list(downloaded.get(fam, []))
        n_dl = len(pool)
        n_need = target - n_dl
        cnfs: list = []
        for data, _fam, name in pool:
            c = parse_cnf(data)
            c.source = "satlib" if "satlib" in str(name) else "downloaded"
            c.family = fam
            c.name = name
            cnfs.append(c)
        if n_need > 0 and fam in gen_specs:
            fn, kw = gen_specs[fam]
            seed = base_seed + _seed_offset(fam)
            cnfs.extend(fn(n_need, seed, **kw))
        out[fam] = cnfs
        log.info("family %s: %d downloaded + %d generated = %d",
                 fam, n_dl, len(cnfs) - n_dl, len(cnfs))
    return out


def _seed_offset(fam: str) -> int:
    order = list(DEFAULT_COUNTS)
    return order.index(fam) + 1 if fam in order else 0


def write_formulas(cfg: dict, formulas: dict[str, list]) -> list[dict]:
    """Write each Cnf to raw/<formula_id>.cnf; return record dicts."""
    raw = Path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen_ids: set[str] = set()
    for fam, cnfs in formulas.items():
        for i, c in enumerate(cnfs):
            base = c.name or f"{fam}-{i:04d}"
            fid = base
            k = 1
            while fid in seen_ids:
                fid = f"{base}_{k}"
                k += 1
            seen_ids.add(fid)
            path = raw / f"{fid}.cnf"
            data = c.to_dimacs()
            path.write_bytes(data)
            is_ood = fam == "ood_scheduling"
            records.append({
                "formula_id": fid,
                "family": fam,
                "source": c.source,
                "name": c.name,
                "nvars": int(c.nvars),
                "nclauses": int(c.nclauses),
                "canonical_hash": c.canonical_hash,
                "renamesafe_hash": c.renamesafe_hash,
                "is_ood": is_ood,
                "path": str(path),
                "size_bytes": len(data),
                "ground_truth": c.extra.get("ground_truth"),
            })
    return records


def verify_roundtrip(records: list[dict]) -> list[dict]:
    """Drop any record whose file does not parse back to the same clauses."""
    kept: list[dict] = []
    for rec in records:
        try:
            parsed = parse_cnf(Path(rec["path"]).read_bytes())
            if parsed.nvars == rec["nvars"] and parsed.nclauses == rec["nclauses"]:
                kept.append(rec)
                continue
        except Exception:  # noqa: BLE001
            pass
        log.warning("dropping formula that failed round-trip: %s", rec["formula_id"])
        try:
            Path(rec["path"]).unlink()
        except OSError:
            pass
    return kept


def write_manifest(cfg: dict, records: list[dict]) -> None:
    manifests = Path(cfg["paths"]["manifests"])
    manifests.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(manifests / "formulas.parquet", index=False)
    with (manifests / "formulas.json").open("w") as fh:
        json.dump(records, fh, indent=2, default=str)


def bench(cfg: dict, counts: dict[str, int] | None = None) -> dict:
    """Run the full bench step. Returns a summary dict."""
    formulas = acquire_formulas(cfg, counts)
    records = write_formulas(cfg, formulas)
    records = verify_roundtrip(records)
    write_manifest(cfg, records)

    summary: dict = {
        "families": {},
        "n_total": len(records),
        "n_ood": sum(1 for r in records if r["is_ood"]),
        "manifest": str(Path(cfg["paths"]["manifests"]) / "formulas.parquet"),
    }
    for rec in records:
        fam = rec["family"]
        entry = summary["families"].setdefault(fam, {"n": 0, "generated": 0, "downloaded": 0})
        entry["n"] += 1
        if rec["source"] == "generated":
            entry["generated"] += 1
        else:
            entry["downloaded"] += 1
    for fam, entry in summary["families"].items():
        nvars = [r["nvars"] for r in records if r["family"] == fam]
        ncl = [r["nclauses"] for r in records if r["family"] == fam]
        entry["nvars_range"] = [min(nvars), max(nvars)] if nvars else [0, 0]
        entry["nclauses_range"] = [min(ncl), max(ncl)] if ncl else [0, 0]
    return summary
