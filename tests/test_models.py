"""Model module tests on synthetic data."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from satsel.core.config import load_config
from satsel.models import classical as cl
from satsel.models import select as sel

SOLVERS = ["minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
           "maplecm", "kissat404"]


def _synthetic_dataset(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(150):
        split = "train" if i < 100 else ("val" if i < 120 else "test")
        # family drives the best solver so the model can learn
        fam = ["random3sat", "colouring", "planning"][i % 3]
        if fam == "random3sat":
            best = "kissat404"
        elif fam == "colouring":
            best = "cadical153"
        else:
            best = "glucose4"
        row = {"formula_id": f"g{i}", "family": fam, "split": split,
               "vbs_solved": True, "tie_flag": 0, "best_solver": best,
               "best_time_s": 0.05, "nvars": 50 + i, "nclauses": 200 + i,
               "path": ""}
        for k in range(10):
            row[f"f_f{k}"] = float(rng.normal(0, 1)) + (1.0 if fam == "random3sat" else 0.0)
        for s in SOLVERS:
            t = 0.05 if s == best else 0.2 + rng.uniform(0, 0.1)
            row[f"{s}_time_s"] = t
            row[f"{s}_solved"] = True
            row[f"{s}_weight"] = 1.0 if s == best else 4.0
        rows.append(row)
    ds = pd.DataFrame(rows).set_index("formula_id")
    cfg = load_config()
    c = copy.deepcopy(cfg)
    c["paths"] = dict(cfg["paths"])
    c["paths"]["models"] = tmp_path / "models"
    return ds, c


def test_train_classical(tmp_path):
    ds, c = _synthetic_dataset(tmp_path)
    port = sel.prune_portfolio(ds, c)
    assert port["best_average_solver"] in SOLVERS
    state = cl.train_classical(ds, c, seed=1)
    assert state["model_type"] in ("rf", "xgb", "svm")
    sol, conf = cl.predict_classical(state, {f"f_f{k}": 0.0 for k in range(10)})
    assert sol in SOLVERS and 0.0 <= conf <= 1.0
    theta = cl.calibrate_theta(ds, c, state, port["best_average_solver"])
    assert 0.0 <= theta["theta"] <= 1.0
    assert {pt["theta"] for pt in theta["curve"]} >= {0.0, 1.0}
    assert json.loads((c["paths"]["models"] / "theta.json").read_text())["theta"] == theta["theta"]


def test_satzilla(tmp_path):
    ds, c = _synthetic_dataset(tmp_path)
    st = cl.fit_satzilla_baseline(ds, c, seed=1)
    pred = cl.predict_satzilla(st, {f"f_f{k}": 0.0 for k in range(10)})
    assert pred in SOLVERS
