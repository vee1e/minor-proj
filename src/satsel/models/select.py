"""Portfolio pruning, single best-average solver, and model selection.

Pruning, complementarity and best-average computation use train+val rows
only, so the test set stays untouched.

``portfolio.json`` keys (contract for the frontend / report):
  solvers, best_average_solver, best_average_time_s, mean_best_solver_median,
  unique_share, pruned                                   (existing)
  complementarity[a][b]  fraction of solved instances where a is strictly
                         faster than b by more than the tie margin
  marginal_contribution[s]  increase in VBS PAR-2 (s) when s is removed
                            from the portfolio (0 => s is dominated)
  marginal_solve_contribution[s]  instances only s solves (count)
  par2_mean[s]           per-solver mean PAR-2 on train+val
  vbs_par2               VBS PAR-2 on train+val
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from satsel.models.classical import SOLVERS


def _solvers_in(dataset: pd.DataFrame) -> list[str]:
    present = [s for s in SOLVERS if f"{s}_time_s" in dataset.columns]
    return present or list(SOLVERS)


def _par2_cols(df: pd.DataFrame, solvers: list[str]) -> np.ndarray:
    cols = []
    for s in solvers:
        t = df[f"{s}_time_s"].to_numpy(dtype=float)
        t = np.nan_to_num(t, nan=np.nanmax(t) if np.isfinite(t).any() else 1.0)
        ok = df[f"{s}_solved"].fillna(False).astype(bool).to_numpy() \
            if f"{s}_solved" in df.columns else np.ones(len(df), dtype=bool)
        cols.append(np.where(ok, t, 2.0 * t))
    return np.stack(cols, axis=1) if cols else np.zeros((len(df), 0))


def prune_portfolio(dataset: pd.DataFrame, cfg: dict) -> dict:
    solvers = _solvers_in(dataset)
    id_df = dataset[(dataset["split"].isin(["train", "val"]))].copy()
    solved = id_df[id_df["vbs_solved"].fillna(False).astype(bool)]
    tie_margin = float(cfg.get("tie_margin", 1.2))

    unique_share: dict[str, float] = {}
    for s in solvers:
        unique_share[s] = float((solved["best_solver"] == s).mean()) if len(solved) else 0.0

    prune_min = float(cfg.get("prune_min_unique_share", 0.03))
    pruned = [s for s in solvers if unique_share[s] < prune_min]
    kept = [s for s in solvers if s not in pruned]
    if len(kept) < 2:
        # never prune below 2 solvers
        top = sorted(solvers, key=lambda s: -unique_share[s])
        kept = top[:2]
        pruned = [s for s in solvers if s not in kept]

    mean_times = {s: float(solved[f"{s}_time_s"].mean()) if len(solved) else float("nan")
                  for s in solvers}
    best_avg = min(mean_times, key=lambda s: (np.nan_to_num(mean_times[s], nan=np.inf)))
    mean_best_median = float(solved["best_time_s"].mean()) if len(solved) else float("nan")

    # --- pairwise complementarity: P(a strictly faster than b by > margin) ---
    complementarity: dict[str, dict[str, float]] = {a: {} for a in solvers}
    if len(solved):
        T = {s: solved[f"{s}_time_s"].to_numpy(dtype=float) for s in solvers}
        OK = {s: (solved[f"{s}_solved"].fillna(False).astype(bool).to_numpy()
                  if f"{s}_solved" in solved.columns else np.ones(len(solved), dtype=bool))
              for s in solvers}
        for a in solvers:
            for b in solvers:
                if a == b:
                    complementarity[a][b] = 0.0
                    continue
                a_wins = OK[a] & (~OK[b] | (T[a] * tie_margin < T[b]))
                complementarity[a][b] = float(np.mean(a_wins))
    else:
        for a in solvers:
            for b in solvers:
                complementarity[a][b] = 0.0

    # --- marginal contribution to the VBS (PAR-2 on all train+val rows) ---
    P = _par2_cols(id_df, solvers)
    n = max(len(id_df), 1)
    vbs_par2 = float(P.min(axis=1).mean()) if P.size else float("nan")
    par2_mean = {s: float(P[:, i].mean()) if P.size else float("nan")
                 for i, s in enumerate(solvers)}
    marginal_contribution: dict[str, float] = {}
    marginal_solve: dict[str, int] = {}
    for i, s in enumerate(solvers):
        if P.shape[1] < 2:
            marginal_contribution[s] = 0.0
            marginal_solve[s] = 0
            continue
        rest = np.delete(P, i, axis=1)
        without = float(rest.min(axis=1).mean())
        marginal_contribution[s] = max(without - vbs_par2, 0.0)
        if f"{s}_solved" in id_df.columns:
            ok = np.stack([id_df[f"{t}_solved"].fillna(False).astype(bool).to_numpy()
                           if f"{t}_solved" in id_df.columns else np.ones(n, dtype=bool)
                           for t in solvers], axis=1)
            only_s = ok[:, i] & ~np.delete(ok, i, axis=1).any(axis=1)
            marginal_solve[s] = int(only_s.sum())
        else:
            marginal_solve[s] = 0

    result = {
        "solvers": kept,
        "best_average_solver": best_avg,
        "best_average_time_s": float(mean_times[best_avg]),
        "mean_best_solver_median": mean_best_median,
        "unique_share": unique_share,
        "pruned": pruned,
        "complementarity": complementarity,
        "marginal_contribution": marginal_contribution,
        "marginal_solve_contribution": marginal_solve,
        "par2_mean": par2_mean,
        "vbs_par2": vbs_par2,
        "tie_margin": tie_margin,
        "n_instances": int(len(id_df)),
        "n_vbs_solved": int(len(solved)),
    }
    out = Path(cfg["paths"]["models"]) / "portfolio.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    return result


def select_model(cfg: dict) -> str:
    """Return 'classical' or 'gnn' based on validation PAR-2."""
    models_dir = Path(cfg["paths"]["models"])
    classical_path = models_dir / "classical.joblib"
    gnn_path = models_dir / "gnn.pt"
    if not classical_path.exists():
        raise FileNotFoundError(f"no trained model at {classical_path}")
    if not gnn_path.exists():
        return "classical"
    c_state = joblib.load(classical_path)
    try:
        import torch

        g_state = torch.load(gnn_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return "classical"
    c_par2 = float(c_state.get("val_par2", float("inf")))
    g_par2 = float(g_state.get("val_par2", float("inf")))
    return "gnn" if g_par2 < c_par2 else "classical"
