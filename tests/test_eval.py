"""Eval metric and statistic tests with hand-computed values."""

import numpy as np
import pandas as pd

from satsel.eval import metrics, stats


class TestMetrics:
    def test_par2(self):
        times = pd.Series([1.0, 30.0, 4.0])  # solved, unsolved, solved
        solved = pd.Series([True, False, True])
        out = metrics.par2_series(times, solved, feat_overhead_s=pd.Series([0.1, 0.1, 0.1]))
        assert list(out) == [1.1, 60.1, 4.1]

    def test_oracle_gap(self):
        p = pd.Series([1.0, 10.0, 3.0])
        b = pd.Series([0.5, 0.5, 0.5])  # VBS-solvable, sum(b)=1.5
        gap = metrics.oracle_gap(p, b, cap=10.0)
        assert abs(gap - (1.0 + 5.0 + 3.0) / 1.5) < 1e-9

    def test_classification_accuracy_excludes_ties(self):
        pred = pd.Series(["a", "a", "b", "a"])
        opt = pd.Series(["a", "b", "b", "b"])
        tie = pd.Series([0, 1, 0, 0])
        # non-tie rows 0,2,3: correct on 0 and 2 -> 2/3
        assert abs(metrics.classification_accuracy(pred, opt, tie) - 2 / 3) < 1e-9


class TestStats:
    def test_bootstrap_ci_excludes_zero(self):
        rng = np.random.default_rng(0)
        n = 200
        diffs = rng.normal(0.5, 0.2, n)  # consistently positive
        fam = np.array([f"f{i % 4}" for i in range(n)])
        res = stats.paired_bootstrap(diffs, fam, n_resamples=2000, seed=1)
        assert res["ci_excludes_zero"] is True
        assert res["ci_low"] > 0

    def test_wilcoxon_both_solved(self):
        a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        b = np.array([2.0, 3.0, 4.0, np.nan, 1.0])
        res = stats.wilcoxon_both_solved(a, b)
        assert res["n"] == 4  # nan pair excluded

    def test_mcnemar(self):
        a = np.array([True, True, False, False, True, False, True, False])
        b = np.array([True, False, True, False, True, True, False, False])
        res = stats.mcnemar(a, b)
        # discordant: (a T b F) rows: index1,index6 ; (a F b T) rows: index2,index5
        assert res["n_ab"] == 2 and res["n_ba"] == 2

    def test_holm(self):
        p = [0.001, 0.02, 0.03, 0.8]
        reject = stats.holm_bonferroni(p, alpha=0.05)
        # sorted: 0.001 (0.0125) reject; 0.02 (0.0167) not reject -> all others not
        assert reject == [True, False, False, False]
