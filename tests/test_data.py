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
    for i in range(8):
        fam = ["random3sat", "random3sat", "colouring", "colouring", "scheduling",
               "scheduling", "ood_scheduling", "ood_scheduling"][i]
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
            t = 0.01 + (SOLVERS.index(s) * 0.02) + rng.uniform(0, 0.01)
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
    assert len(labels) == 8
    assert not labels.loc["f5", "vbs_solved"]
    assert set(labels.loc["f5", [f"{s}_weight" for s in SOLVERS]]) == {1.0}
    assert labels.loc["f0", "best_solver"] == "minisat22"
    assert labels.loc["f0", "minisat22_weight"] == 1.0


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
