"""Profiling worker: solve one (formula, solver, run) in a subprocess.

Runs via `python -m satsel.profiling.worker` so the caller can kill the whole
process group on timeout (a Python-level signal cannot stop a blocked C
solver). Prints one JSON object to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from satsel.core.dimacs import parse_cnf
from satsel.core.solvers import solve_in_process


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--timeout", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formula-id", default="")
    args = ap.parse_args(argv)

    try:
        text = Path(args.cnf).read_bytes()
        cnf = parse_cnf(text, max_vars=2_000_000, max_clauses=10_000_000)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    out = solve_in_process(cnf.clauses, cnf.nvars, args.solver, args.timeout, seed=args.seed)
    print(json.dumps({
        "ok": True,
        "formula_id": args.formula_id,
        "solver": args.solver,
        "status": out.status,
        "runtime_s": round(out.runtime_s, 6),
        "satisfiable": out.satisfiable,
        "message": out.message,
        "nvars": cnf.nvars,
        "nclauses": cnf.nclauses,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
