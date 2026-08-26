"""Data module tests: label derivation and cluster-safe splits on a synthetic
profiling DB."""

import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from satsel.core.config import load_config
from satsel.data import build as dbuild
from satsel.data.labels import derive_labels

SOLVERS = ["minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
           "maplecm", "kissat404"]
TAU = {"random3sat": 60.0, "colouring": 120.0, "scheduling": 180.0,
       "ood_scheduling": 300.0}


def _make_env(tmp_path):
    cfg = load_config()
    c = copy.deepcopy(cfg)
    c["paths"] = dict(cfg["paths"])
    for k in ("raw", "manifests", "features", "runs", "datasets", "models", "reports"):
        c["paths"][k] = tmp_path / k
    c["paths"]["db"] = tmp_path / "profiling.db"
    (tmp_path / "manifests").mkdir(parents=True)
    rng = np.random.default_rng(7)
    man = []
    rows = []
    for i in range(10):
        fam = ["random3sat", "random3sat", "colouring", "colouring", "scheduling",
               "scheduling", "ood_scheduling", "ood_scheduling", "random3sat",
               "colouring"][i]
        fid = f"f{i}"
        hashv = "hashA" if i in (0, 1) else f"h{i}"
        man.append(dict(formula_id=fid, family=fam, source="generated", name=fid,
                        nvars=50 + i, nclauses=200 + i, canonical_hash=hashv,
                        renamesafe_hash="r", is_ood=fam == "ood_scheduling", path="x"))
        for s in SOLVERS:
            if fid == "f5":
                for r in range(3):
                    rows.append((fid, s, fam, r, "timeout", TAU[fam], TAU[fam]))
                continue
            if fid == "f8":
                # lucky single finish: 1 solved + 2 timeouts must NOT count as solved
                if s == "minisat22":
                    rows.append((fid, s, fam, 0, "solved", 0.5, TAU[fam]))
                    rows.append((fid, s, fam, 1, "timeout", TAU[fam], TAU[fam]))
                    rows.append((fid, s, fam, 2, "timeout", TAU[fam], TAU[fam]))
                else:
                    for r in range(3):
                        rows.append((fid, s, fam, r, "solved", 2.0 + r * 0.1, TAU[fam]))
                continue
            if fid == "f9":
                # all solvers within 5%: a tie, not informative
                for r in range(3):
                    rows.append((fid, s, fam, r, "solved", 1.0 + 0.01 * SOLVERS.index(s), 5.0))
                continue
            t = 0.03 + (SOLVERS.index(s) * 0.02) + rng.uniform(0, 0.005)
            for r in range(3):
                rows.append((fid, s, fam, r, "solved", t, 5.0))
    pd.DataFrame(man).to_parquet(tmp_path / "manifests" / "formulas.parquet", index=False)
    con = sqlite3.connect(c["paths"]["db"])
    con.execute("CREATE TABLE runs(formula_id TEXT, solver TEXT, family TEXT, run_idx INT,"
                " status TEXT, runtime_s REAL, tau_s REAL, wall_s REAL, message TEXT,"
                " presolved INT, presolve_time_s REAL)")
    con.executemany("INSERT INTO runs VALUES(?,?,?,?,?,?,?,NULL,'',0,NULL)", rows)
    con.commit()
    con.close()
    return c


def test_labels(tmp_path):
    c = _make_env(tmp_path)
    labels = derive_labels(c)
    assert len(labels) == 10
    assert not labels.loc["f5", "vbs_solved"]
    assert set(labels.loc["f5", [f"{s}_weight" for s in SOLVERS]]) == {1.0}
    assert not labels.loc["f5", "informative"]
    assert labels.loc["f0", "best_solver"] == "minisat22"
    assert labels.loc["f0", "minisat22_weight"] == 1.0
    # regret columns: 0 for the best solver, positive for the slowest
    assert labels.loc["f0", "minisat22_regret_s"] == 0.0
    assert labels.loc["f0", "kissat404_regret_s"] == pytest.approx(
        labels.loc["f0", "kissat404_time_s"] - labels.loc["f0", "best_time_s"])
    # f0: best 30ms, worst ~150ms -> spread 5x -> informative, not a tie
    assert labels.loc["f0", "informative"]
    assert labels.loc["f0", "tie_flag"] == 0


def test_median_rule_and_ties(tmp_path):
    c = _make_env(tmp_path)
    labels = derive_labels(c)
    # f8: minisat finished once out of three runs -> median is censored -> not
    # solved; glucose4 (3 finishes at ~2.1s) is the label.
    assert not labels.loc["f8", "minisat22_solved"]
    assert labels.loc["f8", "minisat22_time_s"] == TAU["random3sat"]
    assert labels.loc["f8", "best_solver"] == "glucose4"
    assert labels.loc["f8", "glucose4_time_s"] == pytest.approx(2.1)
    assert labels.loc["f8", "informative"]  # a timeout vs a finish is a huge gap
    # f9: everything within 6% -> tie, not informative
    assert labels.loc["f9", "tie_flag"] == 1
    assert not labels.loc["f9", "informative"]
    assert labels.loc["f9", "time_spread"] == pytest.approx(1.06, abs=1e-6)


def test_tie_config_keys(tmp_path):
    c = _make_env(tmp_path)
    assert c["tie_floor_ms"] == 20
    assert c["tie_margin"] == pytest.approx(1.1)
    assert c["label_min_spread"] == pytest.approx(1.5)
    c2 = copy.deepcopy(c)
    c2["label_min_spread"] = 100.0
    labels = derive_labels(c2)
    assert not labels["informative"].any()


def test_build_dataset_cluster_safe(tmp_path):
    c = _make_env(tmp_path)
    dbuild.build_dataset(c)
    ds = dbuild.load_dataset(c)
    splits = dbuild.load_splits(c)
    assert set(ds["split"]) == {"train", "val", "test", "ood"}
    hashes = {part: {ds.loc[f, "canonical_hash"] for f in splits[part]}
              for part in ("train", "val", "test")}
    assert not (hashes["train"] & hashes["val"])
    assert not (hashes["val"] & hashes["test"])
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"]) | set(splits["ood"])
    assert all_ids == set(ds.index)
    assert all(ds.loc[f, "is_ood"] for f in splits["ood"])
    # contract columns for the model stage
    assert "informative" in ds.columns
    assert all(f"{s}_regret_s" in ds.columns for s in SOLVERS)


def test_dataset_summary_diagnostics(tmp_path):
    import json

    c = _make_env(tmp_path)
    dbuild.build_dataset(c)
    summ = json.loads((tmp_path / "datasets" / "dataset_summary.json").read_text())
    for key in ("label_distribution", "label_majority_share", "tie_rate",
                "informative_rate", "n_train_usable", "warnings",
                "family_median_best_time_s", "n_id", "families"):
        assert key in summ, key
    assert summ["label_distribution"]["minisat22"] >= 5
    assert 0.0 <= summ["label_majority_share"] <= 1.0
    assert summ["label_majority_share"] > 0.6
    assert any("dominated by one solver" in w for w in summ["warnings"])
    assert any("n_train_usable" in w for w in summ["warnings"])
    assert isinstance(summ["n_train_usable"], int)
    assert set(summ["family_median_best_time_s"]) == set(summ["families"])
