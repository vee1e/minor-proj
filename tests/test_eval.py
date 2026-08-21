"""Eval metric and statistic tests with hand-computed values, plus an
end-to-end run_evaluation/write_report smoke test on a synthetic dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from satsel.eval import metrics, stats
from satsel.models.classical import SOLVERS


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
        assert res["means"].shape == (2000,)
        assert res["p_two_sided"] < 0.01

    def test_bootstrap_p_value_null(self):
        rng = np.random.default_rng(3)
        n = 300
        diffs = rng.normal(0.0, 1.0, n)  # no effect
        fam = np.array([f"f{i % 3}" for i in range(n)])
        res = stats.paired_bootstrap(diffs, fam, n_resamples=2000, seed=2)
        assert res["ci_excludes_zero"] is False
        assert 0.0 <= res["p_two_sided"] <= 1.0
        assert res["p_two_sided"] > 0.05
        # p is twice the opposite-side fraction of the same resamples
        m = res["means"]
        opp = np.mean(m <= 0) if res["mean_diff"] > 0 else np.mean(m >= 0)
        assert abs(res["p_two_sided"] - min(1.0, 2 * opp)) < 1e-12

    def test_bootstrap_p_value_sign_negative(self):
        diffs = np.full(50, -1.0) + np.linspace(-0.1, 0.1, 50)
        fam = np.array(["a"] * 50)
        res = stats.paired_bootstrap(diffs, fam, n_resamples=500, seed=0)
        assert res["mean_diff"] < 0
        assert res["p_two_sided"] == 0.0
        assert res["ci_high"] < 0

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


# --------------------------------------------------------------------------
# synthetic dataset for run_evaluation
# --------------------------------------------------------------------------

def _synthetic_dataset(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    families = ["random3sat", "colouring"]
    for i in range(n):
        fam = families[i % 2]
        tau = 60.0 if fam == "random3sat" else 120.0
        split = "train" if i < n * 0.5 else ("val" if i < n * 0.7 else "test")
        # two regimes so labels are learnable: easy -> minisat best, hard -> kissat best
        hard = rng.random() < 0.5
        times = {}
        solved = {}
        for s in SOLVERS:
            base = rng.uniform(0.5, 5.0) if hard else rng.uniform(0.05, 0.5)
            if hard and s == "kissat404":
                base *= 0.3
            if not hard and s == "minisat22":
                base *= 0.3
            ok = rng.random() > 0.05
            times[s] = base if ok else tau
            solved[s] = ok
        ok_solvers = [s for s in SOLVERS if solved[s]]
        best = min(ok_solvers, key=lambda s: times[s]) if ok_solvers else ""
        best_t = times[best] if ok_solvers else tau
        row = {
            "formula_id": f"f{i:03d}",
            "family": fam,
            "split": split,
            "path": f"/nonexistent/f{i:03d}.cnf",
            "nvars": 100,
            "nclauses": 400,
            "vbs_solved": bool(ok_solvers),
            "best_solver": best,
            "best_time_s": float(best_t),
            "tie_flag": int(i % 7 == 0),
            "feat_cost_s": 0.01,
            "f_syn_nvars": 100.0 + rng.normal(),
            "f_syn_ratio": (4.0 + rng.normal(0, 0.1)) + (1.0 if hard else 0.0),
            "f_gr_vv_deg_mean": rng.normal(),
            "f_probe_runtime_s": 0.005,
            "f_probe_solved": 0.0,
        }
        for s in SOLVERS:
            row[f"{s}_time_s"] = float(times[s])
            row[f"{s}_solved"] = bool(solved[s])
            w = max(1.0, min(times[s] / best_t, 10.0)) if ok_solvers else 1.0
            row[f"{s}_weight"] = float(w)
        rows.append(row)
    return pd.DataFrame(rows).set_index("formula_id")


@pytest.fixture
def tiny_cfg(tmp_path):
    ds = tmp_path / "datasets"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    for p in (ds, models, reports):
        p.mkdir()
    df = _synthetic_dataset()
    df.to_parquet(ds / "dataset.parquet")
    (ds / "dataset_summary.json").write_text(json.dumps({
        "n_id": len(df), "n_ood": 0, "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()), "n_test": int((df["split"] == "test").sum()),
        "families": {}, "warnings": ["synthetic corpus"]}))
    cfg = {
        "paths": {"datasets": ds, "models": models, "reports": reports},
        "tau_tiers": {"random3sat": 60, "colouring": 120},
        "max_timeout": 300,
        "bootstrap_resamples": 10000,
        "alpha": 0.05,
        "random_baseline_draws": 3,
        "seeds": {"random_baseline": 2026, "split": 42},
        "runs_per_pair": 3,
    }
    return cfg


class TestRunEvaluation:
    def test_run_evaluation_without_models(self, tiny_cfg, monkeypatch):
        monkeypatch.setenv("SATEL_SKIP_GNN", "1")
        from satsel.eval import evaluate, report

        res = evaluate.run_evaluation(tiny_cfg)
        # legacy keys
        for k in ("systems", "random_mean_sd", "oracle_gap_selector", "chosen_selector",
                  "comparisons", "feat_overhead_used"):
            assert k in res
        names = {r["system"] for r in res["systems"]}
        assert {"best_average", "fallback", "majority", "selector", "classical",
                "satzilla", "vbs", "random"} <= names
        # new contract keys
        for k in ("per_system_choice_hist", "classification", "per_family", "label_stats",
                  "feature_ablation", "data_quality_warnings", "bootstrap_resamples_used"):
            assert k in res
        # cap applies: n_test < 200
        assert res["bootstrap_resamples_used"] == 2000
        assert "capped" in res["bootstrap_cap_note"]
        # selector with no models is fallback-only -> a single-solver histogram
        hist = res["per_system_choice_hist"]["selector"]
        assert sum(hist.values()) == res["n_test"] and len(hist) == 1
        sel_row = next(r for r in res["systems"] if r["system"] == "selector")
        assert sel_row["is_fallback_only"] is True
        # vbs is the lower bound on PAR-2
        vbs = next(r for r in res["systems"] if r["system"] == "vbs")["par2"]
        assert all(r["par2"] >= vbs - 1e-9 for r in res["systems"])
        # comparisons carry a real two-sided bootstrap p and Holm flag
        assert len(res["comparisons"]) == 4
        for c in res["comparisons"]:
            assert 0.0 <= c["bootstrap_p"] <= 1.0
            assert c["n_resamples"] == 2000
            assert "holm_reject" in c
        # label stats
        ls = res["label_stats"]
        assert ls["n_test"] == res["n_test"]
        assert 0 < ls["majority_share"] <= 1.0
        assert ls["tie_rate"] is not None
        assert ls["informative_rate"] is None  # column absent -> None, no crash
        # per-family has both families and every system
        assert set(res["per_family"]) == {"random3sat", "colouring"}
        assert "selector" in res["per_family"]["random3sat"]
        # classification block for each predictive system present
        assert set(res["classification"]) >= {"selector", "classical", "satzilla"}
        blk = res["classification"]["selector"]
        assert blk["accuracy"] is not None and 0.0 <= blk["accuracy"] <= 1.0
        assert blk["cost_weighted_accuracy"] is not None
        # ablation ran for all three cumulative groups
        groups = [e["group"] for e in res["feature_ablation"]]
        assert groups == ["syntactic", "syntactic+graph", "all"]
        fitted = [e for e in res["feature_ablation"] if e.get("val_par2") is not None]
        assert len(fitted) == 3
        assert res["feature_ablation"][0]["n_features"] == 2
        assert res["feature_ablation"][2]["n_features"] == 5
        # warnings pulled from dataset_summary + our own
        joined = "\n".join(res["data_quality_warnings"])
        assert "[dataset_summary] synthetic corpus" in joined
        assert "n_test=" in joined
        assert "capped" in joined

        # the report renders and results.json round-trips through json
        path = report.write_report(tiny_cfg, res, evaluate.run_ood(tiny_cfg), None)
        md = open(path, encoding="utf-8").read()
        for section in ("Data quality", "Label distribution", "Portfolio complementarity",
                        "Systems on the test set", "Planned comparisons", "Per-family",
                        "Classification accuracy", "Feature-cost ablation",
                        "Out-of-distribution", "Reproducibility"):
            assert section in md
        assert md.index("Data quality") < md.index("Systems on the test set")
        round_trip = json.loads((tiny_cfg["paths"]["reports"] / "results.json").read_text())
        assert round_trip["per_system_choice_hist"]["selector"] == hist

    def test_run_evaluation_with_portfolio_fallback(self, tiny_cfg, monkeypatch):
        monkeypatch.setenv("SATEL_SKIP_GNN", "1")
        from satsel.eval import evaluate

        (tiny_cfg["paths"]["models"] / "portfolio.json").write_text(json.dumps({
            "best_average_solver": "glucose4",
            "marginal_contribution": {"glucose4": 0.1}}))
        (tiny_cfg["paths"]["models"] / "theta.json").write_text(json.dumps({
            "theta": 0.8, "fallback_solver": "cadical153"}))
        res = evaluate.run_evaluation(tiny_cfg)
        assert res["best_average_solver"] == "glucose4"
        assert res["theta"] == 0.8
        assert list(res["per_system_choice_hist"]["best_average"]) == ["glucose4"]
        assert list(res["per_system_choice_hist"]["selector"]) == ["cadical153"]
        assert res["portfolio"]["marginal_contribution"] == {"glucose4": 0.1}
