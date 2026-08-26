"""Oracle worker: runs parse -> features -> infer -> solve in a subprocess.

Invoked by the parent SolverOracle via `python -m satsel.oracle.worker`.
Prints exactly one JSON line to stdout. The parent enforces the wall clock.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from satsel.core.config import feat_budget_s, load_config
from satsel.core.dimacs import parse_cnf
from satsel.core.schemas import (
    RUN_CRASH,
    RUN_ERROR,
    RUN_SOLVED,
    RUN_TIMEOUT,
    ST_CRASH,
    ST_ERROR,
    ST_SAT,
    ST_TIMEOUT,
    ST_UNSAT,
    SolverOracleResult,
)
from satsel.core.solvers import SOLVER_VERSION, solve_in_process

DEFAULT_FALLBACK = "cadical153"

_MODEL_CACHE: dict[str, object] = {}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _portfolio(model_dir: Path) -> dict:
    return _load_json(model_dir / "portfolio.json") or {}


def _theta(model_dir: Path) -> dict:
    return _load_json(model_dir / "theta.json") or {}


def _classical(model_dir: Path):
    import joblib

    p = model_dir / "classical.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


def _gnn_state(model_dir: Path):
    p = model_dir / "gnn.pt"
    if not p.exists():
        return None
    import torch

    return torch.load(p, map_location="cpu", weights_only=False)


def _get_model(clauses, nvars: int, solver: str) -> list[int] | None:
    from pysat.solvers import Solver

    try:
        s = Solver(name=solver)
        for cl in clauses:
            s.add_clause(list(cl))
        ok = s.solve()
        model = s.get_model() if ok is True else None
        s.delete()
        return model
    except Exception:  # noqa: BLE001
        return None


def _selector_pick(model_dir: Path, cnf, features: dict, gnn_cap: int
                   ) -> tuple[str, float, bool, str]:
    """Return (solver, confidence, fallback_used, reason)."""
    portfolio = _portfolio(model_dir)
    theta_info = _theta(model_dir)
    theta = float(theta_info.get("theta", 1.0))
    fallback = theta_info.get("fallback_solver") or portfolio.get(
        "best_average_solver", DEFAULT_FALLBACK)

    gnn = _gnn_state(model_dir)
    if gnn is not None and cnf.nvars + cnf.nclauses <= gnn_cap:
        try:
            from satsel.models.gnn import predict_gnn

            sol, conf = predict_gnn(gnn, cnf, cnf.nvars,
                                    gnn.get("solver_list", []))
            if sol not in gnn.get("solver_list", []):
                raise ValueError("gnn returned unknown solver")
            return sol, conf, conf < theta, "low_confidence" if conf < theta else ""
        except Exception:  # noqa: BLE001
            pass

    classical = _classical(model_dir)
    if classical is not None:
        from satsel.models.classical import predict_classical

        sol, conf = predict_classical(classical, features)
        return sol, conf, conf < theta, "low_confidence" if conf < theta else ""

    return fallback, 0.0, True, "no_model"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf-b64", required=True)
    ap.add_argument("--timeout", type=float, required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args(argv)

    try:
        cfg = load_config()
        cfg["paths"]["models"] = Path(args.model_dir)
        cnf_bytes = base64.b64decode(args.cnf_b64)

        t0 = time.monotonic()
        cnf = parse_cnf(
            cnf_bytes,
            max_vars=int(cfg["oracle"]["max_vars"]),
            max_clauses=int(cfg["oracle"]["max_clauses"]),
            max_literals=int(cfg["oracle"]["max_literals"]),
        )
        parse_ms = int((time.monotonic() - t0) * 1000)

        portfolio = _portfolio(Path(args.model_dir))
        tau_feat = feat_budget_s(cfg, float(portfolio.get("mean_best_solver_median", 0.5)))

        t1 = time.monotonic()
        from satsel.features.extractor import extract_features

        fv = extract_features(cnf, tau_feat, seed=0)
        feature_ms = int((time.monotonic() - t1) * 1000)
        features = {n: float(v) for n, v in zip(fv.names, fv.values, strict=True)}

        gnn_cap = int(cfg.get("gnn_node_cap", 500000))
        t2 = time.monotonic()
        solver, confidence, fallback_used, reason = _selector_pick(
            Path(args.model_dir), cnf, features, gnn_cap)
        infer_ms = int((time.monotonic() - t2) * 1000)

        t3 = time.monotonic()
        out = solve_in_process(cnf.clauses, cnf.nvars, solver, args.timeout, seed=0)
        solve_ms = int((time.monotonic() - t3) * 1000)

        if out.status in (RUN_CRASH, RUN_ERROR) and not fallback_used:
            # worst case must be no worse than the fallback
            fb = _theta(Path(args.model_dir)).get("fallback_solver") or portfolio.get(
                "best_average_solver", DEFAULT_FALLBACK)
            if fb != solver:
                out = solve_in_process(cnf.clauses, cnf.nvars, fb, args.timeout, seed=0)
                solver = fb
                fallback_used = True
                reason = "solver_failure"

        if out.status == RUN_SOLVED and out.satisfiable is True:
            status = ST_SAT
            assignment = _get_model(cnf.clauses, cnf.nvars, solver)
        elif out.status == RUN_SOLVED:
            status = ST_UNSAT
            assignment = None
        elif out.status == RUN_TIMEOUT:
            status = ST_TIMEOUT
            assignment = None
        elif out.status == RUN_CRASH:
            status = ST_CRASH
            assignment = None
        else:
            status = ST_ERROR
            assignment = None

        result = SolverOracleResult(
            status=status,
            satisfiable=True if status == ST_SAT else (False if status == ST_UNSAT else None),
            assignment=tuple(assignment) if assignment else None,
            solver_used=solver,
            solver_version=SOLVER_VERSION,
            runtime_ms=int((time.monotonic() - t0) * 1000),
            parse_ms=parse_ms,
            feature_ms=feature_ms,
            infer_ms=infer_ms,
            solve_ms=solve_ms,
            features=tuple(fv.values),
            confidence=confidence,
            theta_used=float(_theta(Path(args.model_dir)).get("theta", 1.0)),
            fallback_used=fallback_used,
            fallback_reason=reason or None,
            error=None,
        )
        print(json.dumps({"ok": True, "result": result.to_dict()}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
