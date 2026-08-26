"""Statistical tests (EVAL_PROTOCOL.md §5).

- Primary: family-stratified paired bootstrap on per-instance PAR-2
  differences; claim a win iff the 95% CI excludes zero. The same resamples
  give a two-sided bootstrap p-value (fraction of resample means on the
  opposite side of zero from the observed mean, doubled).
- Wilcoxon signed-rank only on pairs where both systems solved.
- McNemar for solve rate.
- Holm-Bonferroni across the planned comparisons.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def paired_bootstrap(diffs: np.ndarray, family: np.ndarray, n_resamples: int = 10000,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """Stratified bootstrap: resample within each family with replacement,
    family sizes fixed, then pool and take the mean diff.

    Returns the CI, the resample means (``means``) and a two-sided bootstrap
    p-value (``p_two_sided``) computed from those same resamples.
    """
    diffs = np.asarray(diffs, dtype=float)
    family = np.asarray(family)
    n = int(len(diffs))
    n_resamples = max(int(n_resamples), 1)
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "ci_excludes_zero": False, "n_resamples": n_resamples, "n": 0,
                "means": np.zeros(n_resamples), "p_two_sided": 1.0}
    rng = np.random.default_rng(seed)
    fams = np.unique(family)

    # vectorised per family: sum of resampled diffs, then divide by pooled n
    sums = np.zeros(n_resamples)
    for f in fams:
        idx = np.where(family == f)[0]
        picks = rng.integers(0, len(idx), size=(n_resamples, len(idx)))
        sums += diffs[idx[picks]].sum(axis=1)
    means = sums / n

    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    mean_diff = float(diffs.mean())
    if mean_diff > 0:
        opposite = float(np.mean(means <= 0.0))
    elif mean_diff < 0:
        opposite = float(np.mean(means >= 0.0))
    else:
        opposite = 0.5
    p_two = float(min(1.0, 2.0 * opposite))
    return {
        "mean_diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "ci_excludes_zero": bool((lo > 0) or (hi < 0)),
        "n_resamples": n_resamples,
        "n": n,
        "means": means,
        "p_two_sided": p_two,
    }


def wilcoxon_both_solved(a: np.ndarray, b: np.ndarray) -> dict:
    """Signed-rank on pairs where both systems solved (no censoring ties)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    x, y = a[mask], b[mask]
    if len(x) < 2 or np.all(x == y):
        return {"n": int(len(x)), "statistic": None, "p_value": None}
    try:
        stat, p = scipy_stats.wilcoxon(x, y)
    except ValueError:
        return {"n": int(len(x)), "statistic": None, "p_value": None}
    return {"n": int(len(x)), "statistic": float(stat), "p_value": float(p)}


def mcnemar(a_solved: np.ndarray, b_solved: np.ndarray) -> dict:
    """Exact two-sided McNemar on the solved/unsolved 2x2 table."""
    a = np.asarray(a_solved, dtype=bool)
    b = np.asarray(b_solved, dtype=bool)
    n_ab = int(np.sum(a & ~b))
    n_ba = int(np.sum(~a & b))
    n_discordant = n_ab + n_ba
    if n_discordant == 0:
        return {"n_ab": 0, "n_ba": 0, "p_value": 1.0}
    p = 2.0 * scipy_stats.binomtest(min(n_ab, n_ba), n_discordant, 0.5).pvalue
    p = min(p, 1.0)
    return {"n_ab": n_ab, "n_ba": n_ba, "p_value": float(p)}


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Reject flags (True = reject null), Holm-Bonferroni adjusted."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    reject = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[i] <= threshold:
            reject[i] = True
        else:
            break
    return reject
