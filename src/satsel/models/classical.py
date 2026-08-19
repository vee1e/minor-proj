"""Classical cost-weighted selectors and the SATzilla-style baseline.

Models predict the best solver (cost-weighted multiclass classification).
Model choice and the fallback threshold `theta` are tuned on validation PAR-2
only; the test set is never touched during training or tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SOLVERS = ("minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
           "maplecm", "kissat404")


def _feature_cols(dataset: pd.DataFrame) -> list[str]:
    cols = [c for c in dataset.columns if c.startswith("f_")]
    if not cols:
        raise ValueError("no f_* feature columns in dataset; run the features step first")
    return cols


def _train_rows(dataset: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    feats = _feature_cols(dataset)
    df = dataset[(dataset["split"] == "train") & dataset["vbs_solved"] &
                 (dataset["tie_flag"] == 0)].copy()
    if df.empty:
        raise ValueError("no usable training rows")
    X = df[feats].fillna(0.0).to_numpy(dtype=float)
    y = df["best_solver"].to_numpy()
    w = np.array([df.loc[i, f"{s}_weight"] for i, s in zip(df.index, y, strict=True)], dtype=float)
    return df, X, y, w


def _encode(y):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def _predict_pipe(model, features: dict, feats: list[str], le) -> str:
    x = np.array([[features.get(c, 0.0) for c in feats]], dtype=float)
    return str(le.inverse_transform([model.predict(x)[0]])[0])


def _predict_proba(model, features: dict, feats: list[str], le) -> tuple[str, float]:
    x = np.array([[features.get(c, 0.0) for c in feats]], dtype=float)
    p = model.predict_proba(x)[0]
    idx = int(np.argmax(p))
    return str(le.classes_[idx]), float(p[idx])


def _par2_for(dataset: pd.DataFrame, predict_fn, subset: str) -> float:
    feats = _feature_cols(dataset)
    sub = dataset[dataset["split"] == subset]
    if sub.empty:
        return float("inf")
    total = 0.0
    for _, row in sub.iterrows():
        fdict = {c: (0.0 if pd.isna(row[c]) else float(row[c])) for c in feats}
        pred = predict_fn(fdict)
        if pred not in SOLVERS:
            pred = "kissat404"
        t = float(row[f"{pred}_time_s"])
        total += t if bool(row[f"{pred}_solved"]) else 2.0 * t
    return total / len(sub)


def train_classical(dataset: pd.DataFrame, cfg: dict, seed: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    _, X, y, w = _train_rows(dataset)
    feats = _feature_cols(dataset)
    y_enc, le = _encode(y)

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X, y_enc, sample_weight=w)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        random_state=seed, n_jobs=-1, objective="multi:softprob")
    xgb.fit(X, y_enc, sample_weight=w)
    svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
    svm.fit(X, y_enc, sample_weight=w)

    candidates = {
        "rf": rf,
        "xgb": xgb,
        "svm": svm,
    }
    best_type, best_par2 = None, float("inf")
    for mt, model in candidates.items():
        vp = _par2_for(dataset,
                       lambda f, m=model: _predict_pipe(m, f, feats, le), "val")
        if vp < best_par2:
            best_type, best_par2 = mt, vp

    model = candidates[best_type]
    state = {
        "model_type": best_type,
        "pipeline": model,
        "feature_names": feats,
        "solver_list": list(le.classes_),
        "label_encoder": le,
        "train_seed": int(seed),
        "val_par2": float(best_par2),
    }
    out = Path(cfg["paths"]["models"]) / "classical.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, out)
    return state


def predict_classical(model_state: dict, features: dict[str, float]) -> tuple[str, float]:
    try:
        model = model_state["pipeline"]
        feats = model_state["feature_names"]
        le = model_state["label_encoder"]
        return _predict_proba(model, features, feats, le)
    except Exception:  # noqa: BLE001
        return "kissat404", 0.0


def fit_satzilla_baseline(dataset: pd.DataFrame, cfg: dict, seed: int) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    feats = _feature_cols(dataset)
    train = dataset[(dataset["split"] == "train") & dataset["vbs_solved"]].copy()
    if train.empty:
        raise ValueError("no usable train rows for satzilla")
    regs: dict[str, object] = {}
    for s in SOLVERS:
        sub = train[train[f"{s}_solved"]].copy()
        if sub.empty:
            regs[s] = None
            continue
        X = sub[feats].fillna(0.0).to_numpy(dtype=float)
        w = sub[f"{s}_weight"].to_numpy(dtype=float)
        y = np.log1p(sub[f"{s}_time_s"].to_numpy(dtype=float))
        r = RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1)
        r.fit(X, y, sample_weight=w)
        regs[s] = r
    state = {"regressors": regs, "feature_names": feats, "solver_list": list(SOLVERS)}
    out = Path(cfg["paths"]["models"]) / "satzilla.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, out)
    return state


def predict_satzilla(baseline_state: dict, features: dict[str, float]) -> str:
    feats = baseline_state["feature_names"]
    x = np.array([[features.get(c, 0.0) for c in feats]], dtype=float)
    best = baseline_state["solver_list"][0]
    best_t = float("inf")
    for s, reg in baseline_state["regressors"].items():
        if reg is None:
            continue
        t = np.expm1(float(reg.predict(x)[0]))
        if t < best_t:
            best_t = t
            best = s
    return best


def calibrate_theta(dataset: pd.DataFrame, cfg: dict, model_state: dict,
                    fallback_solver: str) -> dict:
    feats = model_state["feature_names"]
    le = model_state["label_encoder"]
    val = dataset[dataset["split"] == "val"].copy()
    model = model_state["pipeline"]
    thetas = sorted(set([0.0, 1.0] + [float(t) for t in cfg.get("theta_grid", [])]))

    preds: list[tuple[str, float]] = []
    for _, row in val.iterrows():
        fdict = {c: (0.0 if pd.isna(row[c]) else float(row[c])) for c in feats}
        preds.append(_predict_proba(model, fdict, feats, le))

    curve = []
    best = (float("inf"), 0.0)
    for theta in thetas:
        total = 0.0
        cov = 0
        for (s, conf), (_, row) in zip(preds, val.iterrows(), strict=False):
            chosen = s if conf >= theta else fallback_solver
            if conf >= theta:
                cov += 1
            t = float(row[f"{chosen}_time_s"])
            total += t if bool(row[f"{chosen}_solved"]) else 2.0 * t
        par2 = total / max(len(val), 1)
        curve.append({"theta": theta, "val_par2": par2, "coverage": cov / max(len(val), 1)})
        if par2 < best[0]:
            best = (par2, theta)

    result = {
        "theta": float(best[1]),
        "curve": curve,
        "best_on": "val",
        "fallback_solver": fallback_solver,
    }
    out = Path(cfg["paths"]["models"]) / "theta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
