"""Label derivation: turn profiling runs into per-formula labels.

Label rule (EVAL_PROTOCOL.md §4):
- solved(i,s) = the MEDIAN of the solver's finished runs is under the family
  tau; when solved, T(i,s) = that median. When not solved, T(i,s) = family tau
  (censored marker). The solved-check and the label use the same statistic so
  a solver that finished once by luck is not counted as solving.
- w(i,s) = clamp(T(i,s)/T(i,s*), 1, cost_clip) where s* is the fastest solver
  that solved the formula. Formulas no solver could finish get weights 1.0 and
  vbs_solved=False (excluded from cost-weighted training, kept in test PAR-2).
- regret(i,s) = T(i,s) - T(i,s*)  (seconds lost by picking s; 0 for s*).
- tie_flag:   second-best time <= tie_margin * best_time
              OR best_time < tie_floor_ms  (best and runner-up indistinguishable)
- informative: max_{s solved} T(i,s) / best_time >= label_min_spread, i.e. the
              portfolio shows a meaningful runtime gap somewhere on this
              formula. Training should use informative rows; `tie_flag` is
              the stricter (and on easy corpora far too aggressive) criterion.
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


def _median(xs: list[float]) -> float:
    return float(pd.Series(xs, dtype=float).median())


def derive_labels(cfg: dict) -> pd.DataFrame:
    """Return a wide per-formula label DataFrame (index = formula_id).

    Only formulas that actually appear in the profiling DB are labelled, so a
    partially-profiled corpus builds a valid dataset from what exists.
    """
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

    profiled_ids = set(runs["formula_id"])
    manifest = manifest[manifest["formula_id"].isin(profiled_ids)]
    if manifest.empty:
        raise ValueError("no manifest formulas appear in the profiling DB")

    cost_clip = float(cfg.get("cost_clip", 10))
    tie_margin = float(cfg.get("tie_margin", 1.1))
    tie_floor_s = float(cfg.get("tie_floor_ms", 20)) / 1000.0
    min_spread = float(cfg.get("label_min_spread", 1.5))

    runs = runs[runs["solver"].isin(SOLVERS)]
    by_formula = {fid: sub for fid, sub in runs.groupby("formula_id")}

    rows = []
    for formula_id, meta in manifest.set_index("formula_id").iterrows():
        fam = str(meta["family"])
        tau = tau_for_family(cfg, fam)
        sub = by_formula.get(formula_id)
        times: dict[str, float] = {}
        solved: dict[str, bool] = {}
        n_runs: dict[str, int] = {}
        for s in SOLVERS:
            sr = sub[sub["solver"] == s] if sub is not None else None
            if sr is None or sr.empty:
                times[s] = tau
                solved[s] = False
                n_runs[s] = 0
                continue
            n_runs[s] = int(len(sr))
            # presolve rows (run_idx < 0) are not label evidence
            sr = sr[sr["run_idx"] >= 0] if "run_idx" in sr else sr
            ok_runs = sr[sr["status"] == RUN_SOLVED]
            ok_times = [float(x) for x in ok_runs["runtime_s"] if x == x]
            # median over finished runs; a solver that finished fewer than half
            # its runs has median = tau (censored), hence not solved.
            n_timeout = int((sr["status"] == "timeout").sum())
            n_total = len(ok_times) + n_timeout   # crashes/errors are not evidence
            if ok_times and 2 * len(ok_times) >= n_total:
                med = _median(ok_times)
            else:
                med = tau
            if ok_times and med < tau:
                times[s] = med
                solved[s] = True
            else:
                times[s] = tau
                solved[s] = False

        solved_solvers = [s for s in SOLVERS if solved[s]]
        vbs_solved = bool(solved_solvers)
        best_solver = min(solved_solvers, key=lambda s: times[s]) if solved_solvers else ""
        best_time = times[best_solver] if solved_solvers else tau
        best_time_safe = max(best_time, 1e-6)

        if vbs_solved:
            weights = {s: max(1.0, min(times[s] / best_time_safe, cost_clip)) for s in SOLVERS}
            regrets = {s: max(0.0, times[s] - best_time) for s in SOLVERS}
        else:
            weights = {s: 1.0 for s in SOLVERS}
            regrets = {s: 0.0 for s in SOLVERS}

        tie_flag = 0
        informative = False
        spread = 1.0
        second_best = float("nan")
        if solved_solvers:
            others = [times[s] for s in solved_solvers if s != best_solver]
            second_best = min(others) if others else float("nan")
            if (others and second_best <= tie_margin * best_time) or best_time < tie_floor_s:
                tie_flag = 1
            # spread over the whole portfolio, timeouts count at tau (a solver
            # that times out while another finishes is maximally informative)
            worst = max(times[s] for s in SOLVERS)
            spread = worst / best_time_safe
            informative = bool(spread >= min_spread and best_time >= tie_floor_s / 2)

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
            "second_best_time_s": float(second_best),
            "time_spread": float(spread),
            "n_solved_solvers": int(len(solved_solvers)),
            "tie_flag": int(tie_flag),
            "informative": bool(informative),
        }
        for s in SOLVERS:
            row[f"{s}_time_s"] = float(times[s])
            row[f"{s}_solved"] = bool(solved[s])
            row[f"{s}_weight"] = float(weights[s])
            row[f"{s}_regret_s"] = float(regrets[s])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("formula_id")
    return df
