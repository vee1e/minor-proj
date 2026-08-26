"""Model module tests on synthetic data."""

import copy
import json

import numpy as np
import pandas as pd

from satsel.core.config import load_config
from satsel.models import classical as cl
from satsel.models import select as sel

SOLVERS = ["minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
           "maplecm", "kissat404"]
FAMILY_BEST = {"random3sat": "kissat404", "colouring": "cadical153",
               "planning": "glucose4", "scheduling": "minisat22"}
N_FEATS = 10


def _synthetic_dataset(tmp_path, n=160, with_new_cols=True):
    """~120 train rows, 4 classes; features carry a family signal."""
    rng = np.random.default_rng(0)
    fams = list(FAMILY_BEST)
    rows = []
    for i in range(n):
        split = "train" if i < int(0.75 * n) else ("val" if i < int(0.9 * n) else "test")
        fam = fams[i % len(fams)]
        best = FAMILY_BEST[fam]
        best_t = float(rng.uniform(0.5, 3.0))
        row = {"formula_id": f"g{i}", "family": fam, "split": split,
               "vbs_solved": True, "tie_flag": 0, "best_solver": best,
               "best_time_s": best_t, "nvars": 50 + i, "nclauses": 200 + i, "path": ""}
        for k in range(N_FEATS):
            row[f"f_f{k}"] = float(rng.normal(0, 1)) + (2.0 if k == fams.index(fam) else 0.0)
        if i % 17 == 0:
            row["f_f0"] = np.nan  # exercise imputation
        for s in SOLVERS:
            t = best_t if s == best else best_t * float(rng.uniform(2.0, 6.0))
            solved = not (s != best and i % 13 == 0)
            row[f"{s}_time_s"] = t
            row[f"{s}_solved"] = solved
            row[f"{s}_weight"] = min(t / best_t, 10.0)
            if with_new_cols:
                row[f"{s}_regret_s"] = t - best_t
        if with_new_cols:
            row["informative"] = True
        rows.append(row)
    ds = pd.DataFrame(rows).set_index("formula_id")
    cfg = load_config()
    c = copy.deepcopy(cfg)
    c["paths"] = dict(cfg["paths"])
    c["paths"]["models"] = tmp_path / "models"
    return ds, c


def test_prune_portfolio_contract(tmp_path):
    ds, c = _synthetic_dataset(tmp_path)
    port = sel.prune_portfolio(ds, c)
    assert port["best_average_solver"] in SOLVERS
    assert set(port["complementarity"]) == set(SOLVERS)
    assert all(set(v) == set(SOLVERS) for v in port["complementarity"].values())
    assert all(port["complementarity"][s][s] == 0.0 for s in SOLVERS)
    assert set(port["marginal_contribution"]) == set(SOLVERS)
    # a solver that is never best contributes nothing to the VBS
    assert port["marginal_contribution"]["lingeling"] == 0.0
    assert port["marginal_contribution"]["kissat404"] > 0.0
    saved = json.loads((c["paths"]["models"] / "portfolio.json").read_text())
    assert saved["solvers"] == port["solvers"]


def test_train_classical(tmp_path):
    ds, c = _synthetic_dataset(tmp_path)
    port = sel.prune_portfolio(ds, c)
    state = cl.train_classical(ds, c, seed=1)
    assert state["model_type"] in ("rf", "xgb", "svm", "regret")
    assert state["n_train_usable"] == int((ds["split"] == "train").sum())

    diag_path = c["paths"]["models"] / "model_diagnostics.json"
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text())
    for key in ("chosen_model", "collapsed", "n_train_usable", "train_label_hist",
                "candidates", "warnings"):
        assert key in diag
    assert set(diag["train_label_hist"]) == set(FAMILY_BEST.values())
    assert diag["chosen_model"] == state["model_type"]
    assert len(diag["candidates"]) >= 3
    for cand in diag["candidates"].values():
        assert "val_par2" in cand and "val_pred_hist" in cand and "val_accuracy" in cand
    # the signal is learnable: the model must not collapse to one class
    assert diag["collapsed"] is False
    assert diag["chosen_val_accuracy"] > 0.6

    sol, conf = cl.predict_classical(state, {f"f_f{k}": 0.0 for k in range(N_FEATS)})
    assert sol in SOLVERS and 0.0 <= conf <= 1.0
    # missing features are imputed rather than crashing
    sol2, conf2 = cl.predict_classical(state, {})
    assert sol2 in SOLVERS and 0.0 <= conf2 <= 1.0
    # family signal is picked up
    feats = {f"f_f{k}": (2.0 if k == 0 else 0.0) for k in range(N_FEATS)}
    assert cl.predict_classical(state, feats)[0] == "kissat404"

    theta = cl.calibrate_theta(ds, c, state, port["best_average_solver"])
    assert 0.0 <= theta["theta"] <= 1.0
    assert {pt["theta"] for pt in theta["curve"]} >= {0.0, 1.0}
    assert json.loads((c["paths"]["models"] / "theta.json").read_text())["theta"] == theta["theta"]


def test_train_classical_without_new_columns(tmp_path):
    """Defensive path: no 'informative' / regret columns in the dataset."""
    ds, c = _synthetic_dataset(tmp_path, with_new_cols=False)
    state = cl.train_classical(ds, c, seed=2)
    diag = json.loads((c["paths"]["models"] / "model_diagnostics.json").read_text())
    assert any("informative" in w for w in diag["warnings"])
    sol, conf = cl.predict_classical(state, {f"f_f{k}": 0.0 for k in range(N_FEATS)})
    assert sol in SOLVERS and 0.0 <= conf <= 1.0


def test_regret_selector_direct():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    C = np.stack([np.exp(X[:, 0]), np.exp(-X[:, 0]), np.full(60, 5.0)], axis=1)
    m = cl._RegretSelector(n_estimators=30, random_state=0).fit(X, C)
    p = m.predict_proba(X[:5])
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1.0)
    pred = m.predict(np.array([[3.0, 0, 0, 0], [-3.0, 0, 0, 0]]))
    assert list(pred) == [1, 0]


def test_satzilla(tmp_path):
    ds, c = _synthetic_dataset(tmp_path)
    st = cl.fit_satzilla_baseline(ds, c, seed=1)
    pred = cl.predict_satzilla(st, {f"f_f{k}": 0.0 for k in range(N_FEATS)})
    assert pred in SOLVERS
    assert cl.predict_satzilla(st, {}) in SOLVERS
