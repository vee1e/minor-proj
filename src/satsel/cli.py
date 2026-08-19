"""Command-line runner for the satsel pipeline.

Commands mirror the stages: bench, features, profile, build, train,
train-gnn, evaluate, oracle, and `all` (full pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from satsel.core.config import load_config


def _cmd_bench(args) -> int:
    from satsel.bench import pipeline as bench

    cfg = load_config()
    counts = None
    if args.small:
        counts = {"random3sat": 30, "random3sat_large": 10, "colouring": 20,
                  "planning": 10, "industrial": 15, "scheduling": 15,
                  "ood_scheduling": 12, "crypto": 5}
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
    print("classical:", model["model_type"],
          f"val_par2={model['val_par2']:.4f}")
    theta = cl.calibrate_theta(ds, cfg, model, portfolio["best_average_solver"])
    print("theta:", theta["theta"])
    cl.fit_satzilla_baseline(ds, cfg, seed)
    print("satzilla baseline saved")
    return 0


def _cmd_train_gnn(args) -> int:
    from satsel.data import build as dbuild
    from satsel.models import gnn as gnn_mod

    cfg = load_config()
    ds = dbuild.load_dataset(cfg)
    res = gnn_mod.train_gnn(ds, cfg, seed=args.seed, epochs=args.epochs)
    print("gnn:", res)
    return 0


def _cmd_evaluate(args) -> int:
    from satsel.eval import evaluate, report

    cfg = load_config()
    results = evaluate.run_evaluation(cfg, selector_kind=args.selector)
    ood = evaluate.run_ood(cfg)
    rob = evaluate.run_robustness(cfg)
    path = report.write_report(cfg, results, ood, rob)
    print("report:", path)
    for row in results["systems"]:
        print(f"  {row['system']:14s} PAR-2={row['par2']:.4f} solve={row['solve_rate']:.2f}")
    return 0


def _cmd_oracle(args) -> int:
    from satsel.oracle.oracle import SolverOracle

    cfg = load_config()
    oracle = SolverOracle(cfg)
    data = Path(args.input).read_bytes()
    result = oracle.solve(data, timeout_s=args.timeout)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


def _cmd_all(args) -> int:
    steps = [(_cmd_bench, args), (_cmd_features, args), (_cmd_profile, args),
             (_cmd_build, args), (_cmd_train, args)]
    if not args.no_gnn:
        steps.append((_cmd_train_gnn, args))
    steps.append((_cmd_evaluate, args))
    for fn, a in steps:
        print(f"== {fn.__name__} ==", flush=True)
        rc = fn(a)
        if rc:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="satsel")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bench", help="download + generate formulas")
    p.add_argument("--small", action="store_true", help="small validation corpus")
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

    p = sub.add_parser("build", help="assemble dataset + splits")
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
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("oracle", help="run the SolverOracle on a CNF file")
    p.add_argument("--input", required=True)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=_cmd_oracle)

    p = sub.add_parser("all", help="run the full pipeline")
    p.add_argument("--no-gnn", action="store_true")
    p.add_argument("--small", action="store_true")
    p.set_defaults(func=_cmd_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
