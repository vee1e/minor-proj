"""Label derivation: turn profiling runs into per-formula labels.

Label rule (EVAL_PROTOCOL.md §4):
- solved(i,s) = median of the solver's finished runs is under the family tau;
  when solved, T(i,s) = that median. When not solved, T(i,s) = family tau
  (censored marker).
- w(i,s) = clamp(T(i,s)/T(i,s*), 1, cost_clip) where s* is the fastest solver
  that solved the formula. Formulas no solver could finish get weights 1.0 and
  vbs_solved=False (excluded from cost-weighted training, kept in test PAR-2).
- Tie: second-best time <= tie_margin * best_time OR best_time < tie_floor.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from satsel.core.config import tau_for_family
from satsel.core.schemas import RUN_SOLVED

SOLVERS = ("minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
           "maplecm", "kissat404")


def _load_manifest(cfg: dict) -> pd.DataFrame:
    manifest = Path(cfg["paths"]["manifests"]) / "formulas.parquet"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest} (run the bench step)")
    return pd.read_parquet(manifest)


def derive_labels(cfg: dict) -> pd.DataFrame:
    """Return a wide per-formula label DataFrame (index = formula_id)."""
    manifest = _load_manifest(cfg)
    db_path = Path(cfg["paths"]["db"])
    if not db_path.exists():
        raise FileNotFoundError(f"profiling db not found: {db_path} (run the profile step)")

    conn = sqlite3.connect(str(db_path))
    runs = pd.read_sql_query(
        "SELECT formula_id, solver, family, run_idx, status, runtime_s, tau_s "
        "FROM runs", conn)
    conn.close()
    if runs.empty:
        raise ValueError("profiling db is empty; nothing to label")

    cost_clip = float(cfg.get("cost_clip", 10))
    tie_margin = float(cfg.get("tie_margin", 1.2))
    tie_floor_ms = float(cfg.get("tie_floor_ms", 50))

    rows = []
    for formula_id, meta in manifest.set_index("formula_id").iterrows():
        fam = str(meta["family"])
        tau = tau_for_family(cfg, fam)
        sub = runs[runs["formula_id"] == formula_id]
        times: dict[str, float] = {}
        solved: dict[str, bool] = {}
        for s in SOLVERS:
            sr = sub[sub["solver"] == s]
            if sr.empty:
                times[s] = tau
                solved[s] = False
                continue
            ok_runs = sr[sr["status"] == RUN_SOLVED]
            ok_times = [float(x) for x in ok_runs["runtime_s"] if x == x]
            if ok_times and min(ok_times) < tau:
                times[s] = float(pd.Series(ok_times).median())
                solved[s] = True
            else:
                times[s] = tau
                solved[s] = False

        solved_solvers = [s for s in SOLVERS if solved[s]]
        vbs_solved = bool(solved_solvers)
        best_solver = min(solved_solvers, key=lambda s: times[s]) if solved_solvers else ""
        best_time = times[best_solver] if solved_solvers else tau

        if vbs_solved:
            weights = {s: max(1.0, min(times[s] / best_time, cost_clip)) for s in SOLVERS}
        else:
            weights = {s: 1.0 for s in SOLVERS}

        tie_flag = 0
        if solved_solvers:
            others = [times[s] for s in solved_solvers if s != best_solver]
            second_best = min(others) if others else best_time
            if second_best <= tie_margin * best_time or best_time < tie_floor_ms / 1000.0:
                tie_flag = 1

        row = {
            "formula_id": formula_id,
            "family": fam,
            "is_ood": bool(meta.get("is_ood", False)),
            "nvars": int(meta.get("nvars", 0)),
            "nclauses": int(meta.get("nclauses", 0)),
            "canonical_hash": str(meta.get("canonical_hash", "")),
            "vbs_solved": vbs_solved,
            "best_solver": best_solver,
            "best_time_s": float(best_time),
            "tie_flag": int(tie_flag),
        }
        for s in SOLVERS:
            row[f"{s}_time_s"] = float(times[s])
            row[f"{s}_solved"] = bool(solved[s])
            row[f"{s}_weight"] = float(weights[s])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("formula_id")
    return df
