"""Command-line runner for the satsel pipeline.

Commands mirror the stages: bench, features, profile, build, train,
train-gnn, evaluate, oracle, diagnose, web, and `all` (full pipeline).

Runnable both as the ``satsel`` console script and as a module
(``python -m satsel.cli <step>``); the web server launches steps the
second way so it does not depend on ``uv`` being on PATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from satsel.core.config import load_config

# Corpus presets for `bench --small` / `bench --tiny`.
#
# PILOT (~80 per in-distribution family + 40 OOD): large enough that the
# label distribution is informative and a classifier has something to learn,
# while most instances still solve in well under a second. This is what
# `--small` now means.
#
# TINY: the old `--small` sizes. Only useful as a CI smoke test of the
# plumbing; the resulting dataset is far too small to train anything
# meaningful (expect the "majority class" warning from `build`).
PILOT_COUNTS: dict[str, int] = {
    "random3sat": 80, "random3sat_large": 80, "colouring": 80,
    "planning": 80, "industrial": 80, "scheduling": 80,
    "ood_scheduling": 40, "crypto": 20,
}
TINY_COUNTS: dict[str, int] = {
    "random3sat": 30, "random3sat_large": 10, "colouring": 20,
    "planning": 10, "industrial": 15, "scheduling": 15,
    "ood_scheduling": 12, "crypto": 5,
}


# ---------------------------------------------------------------------------
# helpers for surfacing data-quality problems
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return None


def _banner(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}", flush=True)


def _print_warnings(warnings: list[str] | None, source: str) -> None:
    if not warnings:
        return
    print(f"\n!! DATA-QUALITY WARNINGS ({source}) !!", flush=True)
    for w in warnings:
        print(f"  - {w}", flush=True)
    print(flush=True)


def _fmt_hist(hist: dict[str, Any] | None, total: float | None = None) -> str:
    if not hist:
        return "  (none)"
    items = sorted(hist.items(), key=lambda kv: -float(kv[1]))
    tot = total if total else sum(float(v) for _, v in items) or 1.0
    return "\n".join(f"  {k:<18}{int(v):>7}  {100.0 * float(v) / tot:5.1f}%"
                     for k, v in items)


def print_dataset_summary(cfg: dict, verbose: bool = True) -> list[str]:
    """Print label distribution + warnings from dataset_summary.json.
    Returns the warnings list (empty if file missing)."""
    path = Path(cfg["paths"]["datasets"]) / "dataset_summary.json"
    summ = _read_json(path)
    if summ is None:
        print(f"  dataset summary not found at {path}")
        return []
    warnings = list(summ.get("warnings") or [])
    if verbose:
        keys = ["n_formulas", "n_train", "n_val", "n_test", "n_train_usable",
                "tie_rate", "informative_rate", "label_majority_share"]
        present = {k: summ[k] for k in keys if k in summ}
        if present:
            print("  " + ", ".join(f"{k}={v}" for k, v in present.items()))
        ld = summ.get("label_distribution")
        if isinstance(ld, dict) and ld:
            print("  label distribution (which solver was fastest):")
            print(_fmt_hist(ld))
    _print_warnings(warnings, "dataset_summary.json")
    return warnings


def print_model_diagnostics(cfg: dict) -> None:
    path = Path(cfg["paths"]["models"]) / "model_diagnostics.json"
    diag = _read_json(path)
    if diag is None:
        print(f"  model diagnostics not found at {path} (run `satsel train`)")
        return
    # Be tolerant about the exact schema: print scalars and any histogram-like
    # dicts we recognise.
    scalars = {k: v for k, v in diag.items()
               if isinstance(v, (str, int, float, bool)) or v is None}
    if scalars:
        print("  " + ", ".join(f"{k}={v}" for k, v in scalars.items()))
    for key in ("predicted_class_distribution", "predicted_distribution",
                "pred_hist", "prediction_histogram", "true_class_distribution"):
        if isinstance(diag.get(key), dict):
            print(f"  {key}:")
            print(_fmt_hist(diag[key]))
    for key in ("cv", "cv_scores", "per_model", "models"):
        if key in diag:
            print(f"  {key}: {json.dumps(diag[key], default=str)[:800]}")
    _print_warnings(diag.get("warnings"), "model_diagnostics.json")

    port = _read_json(Path(cfg["paths"]["models"]) / "portfolio.json")
    if port:
        print(f"  portfolio: best_average={port.get('best_average_solver')} "
              f"kept={port.get('solvers')}")
        if isinstance(port.get("marginal_contribution"), dict):
            print("  marginal contribution (VBS loss if solver removed):")
            for k, v in sorted(port["marginal_contribution"].items(),
                               key=lambda kv: -float(kv[1] or 0)):
                print(f"    {k:<18}{v}")


def print_results_summary(cfg: dict) -> list[str]:
    path = Path(cfg["paths"]["reports"]) / "results.json"
    res = _read_json(path)
    if res is None:
        print(f"  results not found at {path} (run `satsel evaluate`)")
        return []
    for row in res.get("systems", []):
        try:
            print(f"  {row['system']:14s} PAR-2={row['par2']:.4f} "
                  f"solve={row['solve_rate']:.2f}")
        except Exception:  # noqa: BLE001
            print(f"  {row}")
    hist = res.get("per_system_choice_hist")
    if isinstance(hist, dict):
        print("  per-system choice histogram (how often each solver was picked):")
        for system, h in hist.items():
            if isinstance(h, dict):
                print(f"   {system}:")
                print("\n".join("   " + line for line in _fmt_hist(h).splitlines()))
    cls = res.get("classification")
    if isinstance(cls, dict):
        print("  classification: " + ", ".join(
            f"{k}={v}" for k, v in cls.items()
            if isinstance(v, (int, float, str))))
    warnings = list(res.get("data_quality_warnings") or [])
    _print_warnings(warnings, "results.json")
    return warnings


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _cmd_bench(args) -> int:
    from satsel.bench import pipeline as bench

    cfg = load_config()
    counts = None
    if getattr(args, "tiny", False):
        counts = TINY_COUNTS
        print("bench: TINY corpus (CI smoke only; too small to train on)")
    elif getattr(args, "small", False):
        counts = PILOT_COUNTS
        print("bench: PILOT corpus (~80 per family + 40 OOD)")
    summary = bench.bench(cfg, counts)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cmd_features(args) -> int:
    from satsel.features import extractor as fx

    cfg = load_config()
    manifest = Path(cfg["paths"]["manifests"]) / "formulas.parquet"
    if not manifest.exists():
        print("manifest not found; run `satsel bench` first", file=sys.stderr)
        return 1
    fx.batch_extract(cfg, manifest, workers=args.workers,
                     enable_probe=not args.no_probe)
    print(f"features written to {cfg['paths']['features']}")
    return 0


def _cmd_profile(args) -> int:
    import sqlite3

    import pandas as pd

    from satsel.profiling import engine as prof

    cfg = load_config()
    manifest = Path(cfg["paths"]["manifests"]) / "formulas.parquet"
    if not manifest.exists():
        print("manifest not found; run `satsel bench` first", file=sys.stderr)
        return 1
    man = pd.read_parquet(manifest)
    formulas = [(r.formula_id, r.path, r.family) for r in man.itertuples()]
    # Process easy families first so progress is visible immediately; the
    # hard random3sat_large tail is handled last.
    priority = {"colouring": 0, "planning": 1, "scheduling": 2, "industrial": 3,
                "crypto": 4, "random3sat": 5, "random3sat_large": 6,
                "ood_scheduling": 7}
    formulas.sort(key=lambda f: priority.get(f[2], 99))
    n = prof.run_batch(formulas, cfg, max_workers=args.workers, resume=not args.no_resume,
                       heartbeat_every_s=args.heartbeat, limit=args.limit)
    print(f"profiled {n} formulas into {cfg['paths']['db']}")

    con = sqlite3.connect(str(cfg["paths"]["db"]))
    rows = con.execute(
        "SELECT family, COUNT(DISTINCT formula_id) AS n, "
        "SUM(status='solved') AS solved, SUM(status='timeout') AS to_ FROM runs "
        "GROUP BY family ORDER BY family").fetchall()
    con.close()
    print(f"{'family':<18}{'formulas':>10}{'solved_runs':>13}{'timeout_runs':>14}")
    for fam, nf, sv, to in rows:
        print(f"{fam:<18}{nf:>10}{sv:>13}{to:>14}")
    return 0


def _cmd_build(args) -> int:
    from satsel.data import build as dbuild

    cfg = load_config()
    dbuild.build_dataset(cfg)
    print(f"dataset written to {cfg['paths']['datasets']}")
    print_dataset_summary(cfg)
    return 0


def _cmd_train(args) -> int:
    from satsel.data import build as dbuild
    from satsel.models import classical as cl
    from satsel.models import select as sel

    cfg = load_config()
    ds = dbuild.load_dataset(cfg)
    seed = args.seed if args.seed is not None else int(
        cfg.get("seeds", {}).get("model_train", [100])[0])
    portfolio = sel.prune_portfolio(ds, cfg)
    print("portfolio:", portfolio["best_average_solver"],
          "kept:", portfolio["solvers"])
    model = cl.train_classical(ds, cfg, seed)
    print("classical:", model.get("model_type"),
          f"val_par2={model.get('val_par2', float('nan')):.4f}")
    theta = cl.calibrate_theta(ds, cfg, model, portfolio["best_average_solver"])
    print("theta:", theta["theta"])
    cl.fit_satzilla_baseline(ds, cfg, seed)
    print("satzilla baseline saved")
    print_model_diagnostics(cfg)
    return 0


def _cmd_train_gnn(args) -> int:
    from satsel.data import build as dbuild
    from satsel.models import gnn as gnn_mod

    cfg = load_config()
    ds = dbuild.load_dataset(cfg)
    seed = args.seed if args.seed is not None else 100
    res = gnn_mod.train_gnn(ds, cfg, seed=seed, epochs=args.epochs)
    print("gnn:", res)
    return 0


def _cmd_evaluate(args) -> int:
    import os

    from satsel.eval import evaluate, report

    if args.no_gnn:
        os.environ["SATEL_SKIP_GNN"] = "1"
    cfg = load_config()
    results = evaluate.run_evaluation(cfg, selector_kind=args.selector,
                                      include_gnn=not args.no_gnn)
    ood = evaluate.run_ood(cfg)
    rob = evaluate.run_robustness(cfg)
    path = report.write_report(cfg, results, ood, rob)
    print("report:", path)
    print_results_summary(cfg)
    print_dataset_summary(cfg, verbose=False)
    return 0


def _cmd_oracle(args) -> int:
    from satsel.oracle.oracle import SolverOracle

    cfg = load_config()
    oracle = SolverOracle(cfg)
    data = Path(args.input).read_bytes()
    result = oracle.solve(data, timeout_s=args.timeout)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


def _cmd_sources(args) -> int:
    """List the benchmark archives, what they hold, and what is cached."""
    from satsel.bench import download as dl

    cfg = load_config(getattr(args, "config", None))
    raw = Path(cfg["paths"]["raw"])

    if args.fetch:
        target = raw / "_cache"
        print(f"Prefetching {len(dl.SATLIB_ARCHIVES)} archives into {target} ...")
        print("The cache is reusable: copy that folder to an offline machine")
        print("and `satsel bench` will run there without any network access.")
        print()

        def _tick(label: str) -> None:
            print(f"  fetching {label} ...", flush=True)

        results = dl.prefetch(raw, progress=_tick)
        print()
        bad = [k for k, v in results.items() if not v.startswith("ok")]
        for label, status in results.items():
            mark = "ok  " if status.startswith("ok") else "FAIL"
            print(f"  {mark}  {label:24s} {status}")
        print()
        print(f"{len(results) - len(bad)}/{len(results)} archives cached.")
        return 1 if bad else 0

    cached = dl.cache_status(raw)
    probe: dict[str, bool] = {}
    if args.probe:
        print("probing reachability ...")
        print()
        for a in dl.SATLIB_ARCHIVES:
            data = dl._get(f"{dl.SATLIB_BASE}/{a.path}", timeout=15.0)
            probe[a.label] = data is not None

    print(f"SATLIB archives  (source: {dl.SATLIB_BASE})")
    print()
    head = (f"  {'archive':24s} {'family':12s} {'kind':6s} {'n':>5s} "
            f"{'vars':>5s} {'wt':>3s}  cached")
    if args.probe:
        head += "  online"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for a in dl.SATLIB_ARCHIVES:
        mark = "yes" if cached[a.label] else "-"
        row = (f"  {a.label:24s} {a.family:12s} {a.status:6s} "
               f"{a.n_instances:5d} {a.nvars:5d} {a.weight:3d}  {mark:6s}")
        if args.probe:
            row += f"  {'yes' if probe.get(a.label) else 'NO':6s}"
        print(row)

    print()
    for fam in sorted({a.family for a in dl.SATLIB_ARCHIVES}):
        print(f"  {fam:12s} {dl.total_available(fam):5d} real instances available")
    print(f"  {'TOTAL':12s} {dl.total_available():5d}")
    print()
    print("  `satsel bench` downloads these automatically. A family's quota is")
    print("  split by weight, so hard phase-transition sets are preferred over")
    print("  easy ones. Families SATLIB does not cover (industrial, crypto,")
    print("  ood_scheduling) come from the seeded generators.")
    print()
    print("  satsel sources --probe    check every archive is reachable")
    print("  satsel sources --fetch    download them all into the cache now")
    return 0

def _cmd_diagnose(args) -> int:
    """Read-only health check of every artifact. Never trains anything."""
    cfg = load_config()
    _banner("dataset (artifacts/datasets/dataset_summary.json)")
    print_dataset_summary(cfg)
    _banner("models (artifacts/models/model_diagnostics.json, portfolio.json)")
    print_model_diagnostics(cfg)
    theta = _read_json(Path(cfg["paths"]["models"]) / "theta.json")
    if theta:
        print(f"  theta={theta.get('theta')} "
              f"(confidence below this falls back to the pre-solver)")
    _banner("evaluation (artifacts/reports/results.json)")
    print_results_summary(cfg)
    print()
    print("Interpretation: if label_majority_share is close to 1.0, or the choice "
          "histogram shows one solver picked for nearly every instance, the corpus "
          "is too easy/small for solver differentiation. Use `satsel bench` (full) "
          "or at least `--small`, never `--tiny`, for real results.")
    return 0


def _cmd_all(args) -> int:
    steps = [(_cmd_bench, args), (_cmd_features, args), (_cmd_profile, args),
             (_cmd_build, args), (_cmd_train, args)]
    if not args.no_gnn:
        steps.append((_cmd_train_gnn, args))
    steps.append((_cmd_evaluate, args))
    cfg = load_config()
    for fn, a in steps:
        _banner(fn.__name__.replace("_cmd_", ""))
        rc = fn(a)
        if rc:
            print(f"step {fn.__name__} failed with exit code {rc}", file=sys.stderr)
            return rc
    # Final recap so the warnings are the last thing on screen.
    _banner("pipeline finished: data-quality recap")
    w1 = print_dataset_summary(cfg, verbose=False)
    w2 = print_results_summary(cfg)
    if not (w1 or w2):
        print("  no data-quality warnings recorded.")
    return 0


def _cmd_web(args) -> int:
    from satsel.web_server import main as web_main  # lazy: needs flask

    argv = ["--port", str(args.port), "--host", args.host]
    if args.debug:
        argv.append("--debug")
    return web_main(argv)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_corpus_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--small", action="store_true",
                   help="pilot corpus: ~80 formulas per family + 40 OOD. Fast "
                        "(most solve in <1s) but large enough to train on")
    g.add_argument("--tiny", action="store_true",
                   help="CI smoke corpus (30/10/20/...). Plumbing test only; far "
                        "too small to learn a selector")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="satsel",
        description="ML-driven SAT solver selection over a PySAT portfolio.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bench", help="download + generate formulas")
    _add_corpus_flags(p)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("features", help="extract feature vectors")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-probe", action="store_true")
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("profile", help="time solvers on every formula")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="profile at most N formulas (for testing)")
    p.add_argument("--heartbeat", type=float, default=30.0,
                   help="seconds between progress heartbeats")
    p.set_defaults(func=_cmd_profile)

    p = sub.add_parser("build", help="assemble dataset + splits (prints label stats)")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("train", help="train classical selectors + calibrate theta")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("train-gnn", help="train the GNN selector")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.set_defaults(func=_cmd_train_gnn)

    p = sub.add_parser("evaluate", help="run the pre-registered evaluation")
    p.add_argument("--selector", choices=["auto", "classical", "gnn"], default="auto")
    p.add_argument("--no-gnn", action="store_true",
                   help="skip loading the GNN (use the classical selector only)")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("oracle", help="run the SolverOracle on a CNF file")
    p.add_argument("--input", required=True)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=_cmd_oracle)

    p = sub.add_parser("sources",
                       help="list benchmark archives, instance counts and cache state")
    p.add_argument("--probe", action="store_true",
                   help="check that each archive is reachable right now")
    p.add_argument("--fetch", action="store_true",
                   help="download every archive into the local cache (works offline after)")
    p.set_defaults(func=_cmd_sources)

    p = sub.add_parser("diagnose",
                       help="print dataset warnings, label distribution, model "
                            "diagnostics and choice histogram (read-only)")
    p.set_defaults(func=_cmd_diagnose)

    p = sub.add_parser("all", help="bench -> features -> profile -> build -> train "
                                   "-> train-gnn -> evaluate")
    p.add_argument("--no-gnn", action="store_true")
    _add_corpus_flags(p)
    # defaults for the sub-steps that `all` drives
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--heartbeat", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--selector", choices=["auto", "classical", "gnn"], default="auto")
    p.set_defaults(func=_cmd_all)

    p = sub.add_parser("web", help="start the Flask web interface")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--debug", action="store_true", help="Flask debug mode")
    p.set_defaults(func=_cmd_web)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
