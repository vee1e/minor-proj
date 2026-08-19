"""Metric definitions (EVAL_PROTOCOL.md §4).

PAR-2 counts a timeout as double the limit; the censored per-solver time
(`<solver>_time_s`) equals the family tau for unsolved rows, so PAR-2(i) =
time if solved else 2*time. A per-formula feature-overhead term is added on
top when provided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def par2_series(predicted_times_s: pd.Series, solved: pd.Series,
                feat_overhead_s: pd.Series | None = None) -> pd.Series:
    times = predicted_times_s.to_numpy(dtype=float)
    solv = solved.to_numpy(dtype=bool)
    par2 = np.where(solv, times, 2.0 * times)
    if feat_overhead_s is not None:
        par2 = par2 + feat_overhead_s.to_numpy(dtype=float)
    return pd.Series(par2, index=predicted_times_s.index)


def par2_mean(predicted_times_s: pd.Series, solved: pd.Series,
              feat_overhead_s: pd.Series | None = None) -> float:
    return float(par2_series(predicted_times_s, solved, feat_overhead_s).mean())


def solve_rate(solved: pd.Series) -> float:
    return float(solved.mean()) if len(solved) else 0.0


def oracle_gap(predicted_times_s: pd.Series, best_time_s: pd.Series,
               cap: float = 10.0) -> float:
    """Oracle gap = sum of predicted times / sum of best times over VBS-solvable
    rows. The per-instance ratio is capped at `cap` before summing the
    numerator (a single pathological instance cannot dominate the gap)."""
    mask = best_time_s > 0
    if mask.sum() == 0:
        return 0.0
    p = predicted_times_s[mask].to_numpy(dtype=float)
    b = best_time_s[mask].to_numpy(dtype=float)
    p_capped = np.minimum(p, b * cap)
    return float(p_capped.sum() / b.sum())


def classification_accuracy(predicted: pd.Series, optimal: pd.Series,
                            tie: pd.Series) -> float:
    """Accuracy over non-tie rows only (ties excluded per protocol)."""
    mask = tie.to_numpy(dtype=bool) == False  # noqa: E712
    if mask.sum() == 0:
        return 0.0
    return float((predicted[mask].to_numpy() == optimal[mask].to_numpy()).mean())


def cost_weighted_accuracy(weights_for_predicted: pd.Series) -> float:
    """Mean inverse cost weight of the chosen solver (1.0 = best pick)."""
    w = weights_for_predicted.to_numpy(dtype=float)
    w = np.maximum(w, 1e-9)
    return float((1.0 / w).mean()) if len(w) else 0.0
