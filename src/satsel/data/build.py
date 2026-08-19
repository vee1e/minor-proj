"""Dataset build: dedup by canonical hash, cluster-safe 70/15/15 splits, OOD
holdout, and dataset assembly. Writes artifacts consumed by the model stage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from satsel.data.labels import derive_labels

ID_FAMILIES = ("random3sat", "random3sat_large", "colouring", "planning",
               "industrial", "scheduling")


def _cluster_safe_split(cfg: dict, labels: pd.DataFrame) -> dict[str, list[str]]:
    """Split ID formulas 70/15/15 by cluster (canonical hash) so no cluster
    crosses a split, roughly preserving family proportions."""
    seed = int(cfg.get("seeds", {}).get("split", 42))
    rng = __import__("numpy").random.default_rng(seed)

    id_df = labels[~labels["is_ood"]].copy()
    clusters: dict[str, list[str]] = {}
    for fid, row in id_df.iterrows():
        clusters.setdefault(row["canonical_hash"], []).append(fid)

    cluster_list = list(clusters.values())
    rng.shuffle(cluster_list)
    # sort by family for determinism-ish family balance after shuffle
    def fam_key(cl):
        return labels.loc[cl[0], "family"]

    cluster_list.sort(key=fam_key)
    # greedy assignment preserving ratio per family
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    counts = {"train": 0, "val": 0, "test": 0}
    total = len(id_df)
    for cl in cluster_list:
        target = {
            "train": 0.70 * total,
            "val": 0.15 * total,
            "test": 0.15 * total,
        }
        k = len(cl)
        best = min(counts, key=lambda s: (counts[s] + k) / target[s] if target[s] else 1e9)
        if best == "train":
            train.extend(cl)
        elif best == "val":
            val.extend(cl)
        else:
            test.extend(cl)
        counts[best] += k

    ood = [str(f) for f in labels[labels["is_ood"]].index]
    return {"train": train, "val": val, "test": test, "ood": ood}


def build_dataset(cfg: dict) -> None:
    labels = derive_labels(cfg)
    datasets = Path(cfg["paths"]["datasets"])
    datasets.mkdir(parents=True, exist_ok=True)

    # cluster integrity: no OOD canonical hash among ID formulas
    id_hashes = set(labels.loc[~labels["is_ood"], "canonical_hash"])
    ood_hashes = set(labels.loc[labels["is_ood"], "canonical_hash"])
    leak = sorted(ood_hashes & id_hashes)
    n_deduped = int(len(labels[~labels["is_ood"]]) - len(id_hashes))

    splits = _cluster_safe_split(cfg, labels)
    split_map: dict[str, str] = {}
    for part, ids in splits.items():
        for fid in ids:
            split_map[fid] = part

    feat_path = Path(cfg["paths"]["features"]) / "features.parquet"
    feats: pd.DataFrame | None = None
    if feat_path.exists():
        feats = pd.read_parquet(feat_path).set_index("formula_id")

    manifest = pd.read_parquet(Path(cfg["paths"]["manifests"]) / "formulas.parquet")
    path_map = dict(zip(manifest["formula_id"], manifest["path"], strict=False))

    rows = []
    for fid, row in labels.iterrows():
        r = row.to_dict()
        r["formula_id"] = fid
        r["split"] = split_map.get(fid, "ood" if row["is_ood"] else "train")
        r["path"] = path_map.get(fid, "")
        if feats is not None and fid in feats.index:
            for col in feats.columns:
                r[col] = feats.loc[fid, col]
        rows.append(r)
    dataset = pd.DataFrame(rows).set_index("formula_id")

    dataset.to_parquet(datasets / "dataset.parquet")
    labels.to_parquet(datasets / "labels.parquet")
    (datasets / "splits.json").write_text(json.dumps(splits, indent=2))

    # summary
    summary = {
        "n_id": int(len(labels[~labels["is_ood"]])),
        "n_ood": int(len(labels[labels["is_ood"]])),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_deduped": n_deduped,
        "cluster_leak": bool(leak),
        "cluster_leak_hashes": leak[:5],
        "families": {},
        "ood_vbs_solve_rate": _vbs_rate(labels[labels["is_ood"]]),
    }
    for fam in sorted(labels["family"].unique()):
        sub = labels[labels["family"] == fam]
        entry = {
            "n": int(len(sub)),
            "vbs_solve_rate": _vbs_rate(sub),
        }
        for part in ("train", "val", "test"):
            entry[f"n_{part}"] = int(sum(
                1 for fid in splits[part] if labels.loc[fid, "family"] == fam))
        summary["families"][fam] = entry

    (datasets / "dataset_summary.json").write_text(json.dumps(summary, indent=2))


def _vbs_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["vbs_solved"].mean())


def load_labels(cfg: dict) -> pd.DataFrame:
    p = Path(cfg["paths"]["datasets"]) / "labels.parquet"
    if not p.exists():
        raise FileNotFoundError(f"labels not found: {p} (run the build step)")
    return pd.read_parquet(p)


def load_dataset(cfg: dict) -> pd.DataFrame:
    p = Path(cfg["paths"]["datasets"]) / "dataset.parquet"
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p} (run the build step)")
    return pd.read_parquet(p)


def load_splits(cfg: dict) -> dict:
    p = Path(cfg["paths"]["datasets"]) / "splits.json"
    if not p.exists():
        raise FileNotFoundError(f"splits not found: {p} (run the build step)")
    return json.loads(p.read_text())
