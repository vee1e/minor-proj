"""Portfolio pruning, single best-average solver, and model selection.

Pruning and best-average computation use train+val solved rows only, so the
test set stays untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from satsel.models.classical import SOLVERS


def prune_portfolio(dataset: pd.DataFrame, cfg: dict) -> dict:
    id_df = dataset[(dataset["split"].isin(["train", "val"]))].copy()
    solved = id_df[id_df["vbs_solved"]]

    unique_share: dict[str, float] = {}
    for s in SOLVERS:
        share = float((solved["best_solver"] == s).mean())
        unique_share[s] = share

    prune_min = float(cfg.get("prune_min_unique_share", 0.03))
    pruned = [s for s in SOLVERS if unique_share[s] < prune_min]
    kept = [s for s in SOLVERS if s not in pruned]
    if len(kept) < 2:
        # never prune below 2 solvers
        top = sorted(SOLVERS, key=lambda s: -unique_share[s])
        kept = top[:2]
        pruned = [s for s in SOLVERS if s not in kept]

    mean_times = {s: float(solved[f"{s}_time_s"].mean()) for s in SOLVERS}
    best_avg = min(mean_times, key=mean_times.get)
    mean_best_median = float(solved["best_time_s"].mean())

    result = {
        "solvers": kept,
        "best_average_solver": best_avg,
        "best_average_time_s": float(mean_times[best_avg]),
        "mean_best_solver_median": mean_best_median,
        "unique_share": unique_share,
        "pruned": pruned,
    }
    out = Path(cfg["paths"]["models"]) / "portfolio.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
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
