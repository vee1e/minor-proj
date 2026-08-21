"""Profiling worker: solve one (formula, solver) for one or more runs in a
subprocess.

Runs via `python -m satsel.profiling.worker` so the caller can kill the whole
process group on timeout (a Python-level signal cannot stop a blocked C
solver).

Output (stdout, one line):
- ``--runs 1`` (default): a single JSON object (backwards compatible).
- ``--runs N`` with N > 1: a JSON list of N objects, one per run, each with
  ``run_idx`` 0..N-1 and seed ``--seed + run_idx``. The formula is parsed once
  so N repeats cost one Python start-up instead of N.

Each object carries ``runtime_s`` (solve() wall time only), ``solve_cpu_s``
(solver-reported CPU time or null) and ``load_s`` (clause-loading time).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from satsel.core.dimacs import parse_cnf
from satsel.core.solvers import solve_in_process


def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, 6) if math.isfinite(xf) else None


def _run_payload(cnf, solver: str, timeout: float, seed: int, formula_id: str,
                 run_idx: int) -> dict:
    out = solve_in_process(cnf.clauses, cnf.nvars, solver, timeout, seed=seed)
    return {
        "ok": True,
        "formula_id": formula_id,
        "solver": solver,
        "run_idx": run_idx,
        "seed": seed,
        "status": out.status,
        "runtime_s": round(out.runtime_s, 6),
        "solve_cpu_s": _finite_or_none(out.solve_cpu_s),
        "load_s": round(out.load_s, 6),
        "satisfiable": out.satisfiable,
        "message": out.message,
        "nvars": cnf.nvars,
        "nclauses": cnf.nclauses,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--timeout", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formula-id", default="")
    ap.add_argument("--runs", type=int, default=1,
                    help="number of independent runs (seed, seed+1, ...) in this process")
    args = ap.parse_args(argv)

    try:
        text = Path(args.cnf).read_bytes()
        cnf = parse_cnf(text, max_vars=2_000_000, max_clauses=10_000_000)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    n_runs = max(1, int(args.runs))
    payloads = [
        _run_payload(cnf, args.solver, args.timeout, args.seed + r, args.formula_id, r)
        for r in range(n_runs)
    ]
    if n_runs == 1:
        print(json.dumps(payloads[0]))
    else:
        print(json.dumps(payloads))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
