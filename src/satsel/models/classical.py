"""Classical cost-weighted selectors and the SATzilla-style baseline.

Four candidate selectors are trained on the usable training rows:

* ``rf``     - RandomForest classifier over the best-solver label
* ``xgb``    - XGBoost classifier (labels re-encoded densely per fit)
* ``svm``    - RBF SVM with probability outputs
* ``regret`` - one RandomForest regressor per solver on log1p(PAR-2 cost);
               the selector picks argmin predicted cost. This candidate
               cannot collapse to the majority class because it never
               looks at the argmax label directly.

Every candidate is an sklearn ``Pipeline`` (median imputation + standard
scaling + estimator). Classifiers are trained with sample weights that
combine a cost weight (how expensive a wrong choice is on this instance)
with inverse-class-frequency balancing. Hyper-parameters come from a tiny
grid searched with stratified inner K-fold CV scored by negative mean PAR-2
of the *predicted* solver (not accuracy). The final model is chosen by
validation PAR-2; the winning classifier is Platt-calibrated when class
counts allow. Diagnostics (including a collapse detector) are written to
``<models>/model_diagnostics.json``. The test split is never touched here.
"""

from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

SOLVERS = ("minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
           "maplecm", "kissat404")

MIN_USABLE_ROWS = 30
DEFAULT_FALLBACK = "cadical153"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _feature_cols(dataset: pd.DataFrame) -> list[str]:
    cols = [c for c in dataset.columns if c.startswith("f_")]
    if not cols:
        raise ValueError("no f_* feature columns in dataset; run the features step first")
    return cols


def _solvers_in(dataset: pd.DataFrame) -> list[str]:
    present = [s for s in SOLVERS if f"{s}_time_s" in dataset.columns]
    return present or list(SOLVERS)


def _feature_dict(row: pd.Series, feats: list[str]) -> dict[str, float]:
    return {c: (np.nan if pd.isna(row[c]) else float(row[c])) for c in feats}


def _x_from_dict(features: dict, feats: list[str]) -> np.ndarray:
    vals = []
    for c in feats:
        v = features.get(c, np.nan)
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.array([vals], dtype=float)


def _x_from_df(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return df[feats].to_numpy(dtype=float)


def _cost_matrix(df: pd.DataFrame, solvers: list[str]) -> np.ndarray:
    """Per-row PAR-2 cost of choosing each solver (n x K)."""
    cols = []
    for s in solvers:
        t = df[f"{s}_time_s"].to_numpy(dtype=float)
        finite = t[np.isfinite(t)]
        fill = float(finite.max()) if finite.size else 1.0
        t = np.where(np.isfinite(t), t, fill)
        if f"{s}_solved" in df.columns:
            ok = df[f"{s}_solved"].fillna(False).astype(bool).to_numpy()
            t = np.where(ok, t, 2.0 * t)
        cols.append(t)
    return np.stack(cols, axis=1)


def _regret_matrix(df: pd.DataFrame, solvers: list[str]) -> np.ndarray:
    """Regret (s) of each solver vs the best on each row (n x K)."""
    cols = []
    best = df["best_time_s"].to_numpy(dtype=float) if "best_time_s" in df.columns else None
    for s in solvers:
        if f"{s}_regret_s" in df.columns:
            r = df[f"{s}_regret_s"].to_numpy(dtype=float)
        elif best is not None:
            r = df[f"{s}_time_s"].to_numpy(dtype=float) - best
        else:
            r = np.zeros(len(df))
        cols.append(np.nan_to_num(np.maximum(r, 0.0)))
    return np.stack(cols, axis=1)


def _usable_train_rows(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """split=='train' & vbs_solved & informative (fallback ~tie_flag)."""
    warns: list[str] = []
    is_train = dataset["split"] == "train"
    base = dataset[is_train & dataset["vbs_solved"].fillna(False).astype(bool)]
    if "informative" in dataset.columns:
        mask = base["informative"].fillna(False).astype(bool)
    elif "tie_flag" in dataset.columns:
        mask = base["tie_flag"].fillna(1).astype(int) == 0
        warns.append("'informative' column missing; using tie_flag==0 as the informative filter")
    else:
        mask = pd.Series(True, index=base.index)
        warns.append("neither 'informative' nor 'tie_flag' present; "
                     "using all vbs_solved train rows")
    df = base[mask].copy()
    if len(df) < MIN_USABLE_ROWS:
        warns.append(f"only {len(df)} informative training rows (<{MIN_USABLE_ROWS}); "
                     f"falling back to all {len(base)} vbs_solved train rows")
        df = base.copy()
    if df.empty:
        raise ValueError("no usable training rows")
    return df, warns


def _sample_weights(df: pd.DataFrame, y_enc: np.ndarray, solvers: list[str],
                    cost_clip: float) -> tuple[np.ndarray, np.ndarray]:
    """Cost weight (clamped regret ratio) x inverse class frequency, mean 1."""
    best_t = df["best_time_s"].to_numpy(dtype=float) if "best_time_s" in df.columns else None
    regret = _regret_matrix(df, solvers)
    if best_t is None:
        best_t = np.ones(len(df))
    denom = np.maximum(best_t, 1e-3)
    # mean over alternatives of T(i,s)/T(i,s*) = 1 + regret/best
    ratio = 1.0 + regret / denom[:, None]
    k = max(ratio.shape[1] - 1, 1)
    cost_w = (ratio.sum(axis=1) - 1.0) / k
    cost_w = np.clip(np.nan_to_num(cost_w, nan=1.0), 1.0, cost_clip)
    counts = np.bincount(y_enc, minlength=int(y_enc.max()) + 1).astype(float)
    inv = np.where(counts > 0, len(y_enc) / (counts * max((counts > 0).sum(), 1)), 0.0)
    class_w = inv[y_enc]
    w = cost_w * class_w
    w = w / max(w.mean(), 1e-12)
    return w, cost_w


def _hist(labels, classes: list[str]) -> dict[str, int]:
    out = {c: 0 for c in classes}
    for v in labels:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
class _DenseXGB(BaseEstimator, ClassifierMixin):
    """XGBClassifier that tolerates missing classes by re-encoding densely."""

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.1,
                 subsample=0.9, random_state=0, n_jobs=1):
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_total_ = int(self.classes_.max()) + 1 if len(self.classes_) else 0
        dense = np.searchsorted(self.classes_, y)
        objective = "multi:softprob" if len(self.classes_) > 2 else "binary:logistic"
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            random_state=self.random_state, n_jobs=self.n_jobs, objective=objective,
            verbosity=0)
        if len(self.classes_) == 1:
            self.model_ = None
            return self
        self.model_.fit(X, dense, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.model_ is None:
            return np.ones((len(X), 1))
        p = self.model_.predict_proba(X)
        if p.ndim == 1:
            p = np.stack([1 - p, p], axis=1)
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _RegretSelector(BaseEstimator, ClassifierMixin):
    """Per-solver regressors on log1p(PAR-2 cost); selects argmin cost.

    ``fit`` receives ``y`` as an (n x K) matrix of costs. ``predict_proba``
    is a softmax over negative predicted log-costs (temperature 1) so it can
    be used as a confidence like any other candidate.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 random_state=0, n_jobs=1):
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestRegressor

        Y = np.asarray(y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("_RegretSelector expects an (n x K) cost matrix as y")
        self.classes_ = np.arange(Y.shape[1])
        self.regressors_ = []
        for k in range(Y.shape[1]):
            r = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=self.random_state, n_jobs=self.n_jobs)
            r.fit(X, np.log1p(np.maximum(Y[:, k], 0.0)), sample_weight=sample_weight)
            self.regressors_.append(r)
        return self

    def predict_log_cost(self, X):
        X = np.asarray(X)
        return np.stack([r.predict(X) for r in self.regressors_], axis=1)

    def predict_proba(self, X):
        z = -self.predict_log_cost(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmin(self.predict_log_cost(X), axis=1)]


def _make_pipeline(kind: str, seed: int, n_rows: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # parallel workers only pay off on larger datasets; on tiny ones they dominate
    n_jobs = -1 if n_rows >= 1000 else 1
    if kind == "rf":
        est = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs)
        grid = {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 3]}
    elif kind == "xgb":
        est = _DenseXGB(random_state=seed, n_jobs=n_jobs)
        grid = {"clf__max_depth": [3, 6], "clf__learning_rate": [0.1, 0.3]}
    elif kind == "svm":
        est = SVC(kernel="rbf", probability=True, random_state=seed)
        grid = {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]}
    elif kind == "regret":
        est = _RegretSelector(random_state=seed, n_jobs=n_jobs)
        grid = {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 3]}
    else:
        raise ValueError(kind)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", est),
    ])
    return pipe, grid


def _inner_cv_select(pipe, grid: dict, X: np.ndarray, y_enc: np.ndarray, target,
                     w: np.ndarray, C: np.ndarray, seed: int) -> tuple[dict, float, int]:
    """Tiny grid search with stratified K-fold scored by negative mean PAR-2.

    ``target`` is what the estimator is fitted on (label vector or cost matrix);
    ``C`` is the per-row cost of each class, used for scoring.
    """
    from sklearn.model_selection import StratifiedKFold

    counts = np.bincount(y_enc)
    min_count = int(counts[counts > 0].min())
    n_splits = min(5, min_count)
    combos = [dict(zip(grid.keys(), vals, strict=True)) for vals in product(*grid.values())]
    if n_splits < 2 or len(combos) == 1:
        return combos[0], float("nan"), n_splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y_enc))
    best_params, best_score = combos[0], -float("inf")
    for params in combos:
        cost_sum, n = 0.0, 0
        for tr, va in folds:
            m = clone(pipe).set_params(**params)
            m.fit(X[tr], target[tr], clf__sample_weight=w[tr])
            pred = np.asarray(m.predict(X[va]), dtype=int)
            cost_sum += float(C[va, pred].sum())
            n += len(va)
        score = -cost_sum / max(n, 1)
        if score > best_score:
            best_params, best_score = params, score
    return best_params, float(best_score), n_splits


def _predict_state(model_state: dict, features: dict) -> tuple[str, float]:
    model = model_state["pipeline"]
    feats = model_state["feature_names"]
    solver_list = list(model_state["solver_list"])
    x = _x_from_dict(features, feats)
    p = np.asarray(model.predict_proba(x))[0]
    classes = getattr(model, "classes_", None)
    idx = int(np.argmax(p))
    if classes is not None and len(classes) == len(p):
        idx = int(classes[idx])
    return solver_list[idx], float(p[int(np.argmax(p))])


def _eval_split(dataset: pd.DataFrame, model_state: dict, subset: str) -> dict:
    """Val PAR-2, accuracy on non-tie rows, predicted-class histogram."""
    feats = model_state["feature_names"]
    solvers = list(model_state["solver_list"])
    sub = dataset[dataset["split"] == subset]
    if sub.empty:
        return {"par2": float("inf"), "accuracy": float("nan"), "pred_hist": {}, "n": 0}
    total, correct, n_acc = 0.0, 0, 0
    preds = []
    for _, row in sub.iterrows():
        pred, _ = _predict_state(model_state, _feature_dict(row, feats))
        if f"{pred}_time_s" not in sub.columns:
            pred = DEFAULT_FALLBACK
        preds.append(pred)
        t = float(row[f"{pred}_time_s"])
        solved = bool(row.get(f"{pred}_solved", True))
        total += t if solved else 2.0 * t
        informative = bool(row["informative"]) if "informative" in sub.columns else \
            (int(row.get("tie_flag", 0)) == 0)
        if informative and bool(row.get("vbs_solved", True)):
            n_acc += 1
            correct += int(pred == str(row["best_solver"]))
    return {
        "par2": total / len(sub),
        "accuracy": (correct / n_acc) if n_acc else float("nan"),
        "pred_hist": _hist(preds, solvers),
        "n": int(len(sub)),
    }


def _par2_for(dataset: pd.DataFrame, predict_fn, subset: str) -> float:
    feats = _feature_cols(dataset)
    sub = dataset[dataset["split"] == subset]
    if sub.empty:
        return float("inf")
    total = 0.0
    for _, row in sub.iterrows():
        pred = predict_fn(_feature_dict(row, feats))
        if f"{pred}_time_s" not in sub.columns:
            pred = DEFAULT_FALLBACK
        t = float(row[f"{pred}_time_s"])
        total += t if bool(row.get(f"{pred}_solved", True)) else 2.0 * t
    return total / len(sub)


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def train_classical(dataset: pd.DataFrame, cfg: dict, seed: int) -> dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder

    warns: list[str] = []
    feats = _feature_cols(dataset)
    solvers = _solvers_in(dataset)
    df, w_ = _usable_train_rows(dataset)
    warns.extend(w_)

    le = LabelEncoder()
    y_enc = le.fit_transform(df["best_solver"].astype(str).to_numpy())
    classes = [str(c) for c in le.classes_]
    X = _x_from_df(df, feats)
    C = _cost_matrix(df, classes)
    cost_clip = float(cfg.get("cost_clip", 10))
    w, cost_w = _sample_weights(df, y_enc, classes, cost_clip)

    counts = np.bincount(y_enc, minlength=len(classes))
    label_hist = {c: int(n) for c, n in zip(classes, counts, strict=True)}
    majority_share = float(counts.max() / max(counts.sum(), 1))
    if len(classes) < 2:
        warns.append(f"only one class ({classes[0]}) among usable training rows; "
                     "selector degenerates to a constant")

    candidates: dict[str, dict] = {}
    per_candidate: dict[str, dict] = {}
    for kind in ("rf", "xgb", "svm", "regret"):
        try:
            pipe, grid = _make_pipeline(kind, seed, len(X))
            target = C if kind == "regret" else y_enc
            if kind == "svm" and len(classes) < 2:
                raise ValueError("SVM needs >= 2 classes")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, cv_score, n_splits = _inner_cv_select(
                    pipe, grid, X, y_enc, target, w, C, seed)
                model = clone(pipe).set_params(**params)
                model.fit(X, target, clf__sample_weight=w)
            state = {"pipeline": model, "feature_names": feats, "solver_list": classes}
            ev = _eval_split(dataset, state, "val")
            candidates[kind] = {"model": model, "val": ev, "params": params,
                                "cv_score": cv_score, "n_splits": n_splits}
            per_candidate[kind] = {
                "val_par2": ev["par2"], "val_accuracy": ev["accuracy"],
                "val_pred_hist": ev["pred_hist"], "best_params": params,
                "inner_cv_neg_par2": cv_score, "inner_cv_splits": n_splits,
            }
        except Exception as exc:  # noqa: BLE001
            warns.append(f"candidate {kind} failed: {type(exc).__name__}: {exc}")

    if not candidates:
        raise RuntimeError("all classical candidates failed: " + "; ".join(warns))

    best_type = min(candidates, key=lambda k: candidates[k]["val"]["par2"])
    chosen = candidates[best_type]
    model = chosen["model"]
    best_par2 = float(chosen["val"]["par2"])

    # Platt calibration of the winning classifier (not the regret selector)
    calibrated = False
    min_count = int(counts[counts > 0].min())
    if best_type != "regret" and len(classes) >= 2 and min_count >= 3:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cal = CalibratedClassifierCV(estimator=clone(model), method="sigmoid", cv=3)
                try:
                    cal.fit(X, y_enc, **{"clf__sample_weight": w})
                except TypeError:
                    cal.fit(X, y_enc)
            cal_state = {"pipeline": cal, "feature_names": feats, "solver_list": classes}
            cal_ev = _eval_split(dataset, cal_state, "val")
            if np.isfinite(cal_ev["par2"]) and cal_ev["par2"] <= best_par2 * 1.05 + 1e-9:
                model, calibrated = cal, True
                per_candidate[best_type]["calibrated_val_par2"] = cal_ev["par2"]
                per_candidate[best_type]["calibrated_val_pred_hist"] = cal_ev["pred_hist"]
                best_par2 = float(cal_ev["par2"])
            else:
                warns.append(f"calibration of {best_type} worsened val PAR-2 "
                             f"({cal_ev['par2']:.4f} vs {best_par2:.4f}); kept uncalibrated model")
        except Exception as exc:  # noqa: BLE001
            warns.append(f"calibration skipped: {type(exc).__name__}: {exc}")
    elif best_type == "regret":
        warns.append("winner is the regret selector; confidence is a softmax over "
                     "predicted log-costs (no Platt calibration)")
    else:
        warns.append(f"calibration skipped: smallest class has {min_count} rows (<3)")

    state = {
        "model_type": best_type,
        "pipeline": model,
        "feature_names": feats,
        "solver_list": classes,
        "label_encoder": le,
        "train_seed": int(seed),
        "val_par2": float(best_par2),
        "calibrated": calibrated,
        "n_train_usable": int(len(df)),
        "best_params": chosen["params"],
    }
    final_ev = _eval_split(dataset, state, "val")
    pred_hist = final_ev["pred_hist"]
    n_pred = max(sum(pred_hist.values()), 1)
    max_pred_share = max(pred_hist.values()) / n_pred if pred_hist else 0.0
    collapsed = bool(max_pred_share > 0.9 and majority_share < 0.7)
    if collapsed:
        warns.append(f"chosen model '{best_type}' looks collapsed: {max_pred_share:.0%} of val "
                     f"predictions are one class while the train majority share is "
                     f"{majority_share:.0%}")

    diagnostics = {
        "chosen_model": best_type,
        "chosen_val_par2": float(best_par2),
        "chosen_val_accuracy": final_ev["accuracy"],
        "chosen_val_pred_hist": pred_hist,
        "calibrated": calibrated,
        "collapsed": collapsed,
        "max_predicted_share": float(max_pred_share),
        "train_majority_share": majority_share,
        "n_train_usable": int(len(df)),
        "n_train_total": int(((dataset["split"] == "train")).sum()),
        "n_val": int(final_ev["n"]),
        "train_label_hist": label_hist,
        "classes": classes,
        "solvers_in_dataset": solvers,
        "cost_weight_mean": float(np.mean(cost_w)),
        "cost_weight_max": float(np.max(cost_w)),
        "candidates": per_candidate,
        "seed": int(seed),
        "warnings": warns,
    }
    out_dir = Path(cfg["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, out_dir / "classical.joblib")
    (out_dir / "model_diagnostics.json").write_text(json.dumps(_json_safe(diagnostics), indent=2))
    state["diagnostics"] = diagnostics
    return state


def predict_classical(model_state: dict, features: dict[str, float]) -> tuple[str, float]:
    try:
        return _predict_state(model_state, features)
    except Exception:  # noqa: BLE001
        return DEFAULT_FALLBACK, 0.0


def fit_satzilla_baseline(dataset: pd.DataFrame, cfg: dict, seed: int) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats = _feature_cols(dataset)
    solvers = _solvers_in(dataset)
    train = dataset[(dataset["split"] == "train") &
                    dataset["vbs_solved"].fillna(False).astype(bool)].copy()
    if train.empty:
        raise ValueError("no usable train rows for satzilla")
    regs: dict[str, object] = {}
    for s in solvers:
        ok = train[f"{s}_solved"].fillna(False).astype(bool) if f"{s}_solved" in train.columns \
            else pd.Series(True, index=train.index)
        sub = train[ok]
        if sub.empty:
            regs[s] = None
            continue
        X = _x_from_df(sub, feats)
        w = sub[f"{s}_weight"].to_numpy(dtype=float) if f"{s}_weight" in sub.columns \
            else np.ones(len(sub))
        y = np.log1p(sub[f"{s}_time_s"].to_numpy(dtype=float))
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1)),
        ])
        pipe.fit(X, y, reg__sample_weight=np.nan_to_num(w, nan=1.0))
        regs[s] = pipe
    state = {"regressors": regs, "feature_names": feats, "solver_list": solvers}
    out = Path(cfg["paths"]["models"]) / "satzilla.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, out)
    return state


def predict_satzilla(baseline_state: dict, features: dict[str, float]) -> str:
    feats = baseline_state["feature_names"]
    x = _x_from_dict(features, feats)
    best = baseline_state["solver_list"][0]
    best_t = float("inf")
    for s, reg in baseline_state["regressors"].items():
        if reg is None:
            continue
        t = float(np.expm1(float(reg.predict(x)[0])))
        if t < best_t:
            best_t, best = t, s
    return best


def calibrate_theta(dataset: pd.DataFrame, cfg: dict, model_state: dict,
                    fallback_solver: str) -> dict:
    feats = model_state["feature_names"]
    val = dataset[dataset["split"] == "val"].copy()
    thetas = sorted(set([0.0, 1.0] + [float(t) for t in cfg.get("theta_grid", [])]))

    preds: list[tuple[str, float]] = [
        predict_classical(model_state, _feature_dict(row, feats)) for _, row in val.iterrows()
    ]
    if f"{fallback_solver}_time_s" not in val.columns:
        fallback_solver = DEFAULT_FALLBACK

    curve = []
    best = (float("inf"), 0.0)
    for theta in thetas:
        total, cov = 0.0, 0
        for (s, conf), (_, row) in zip(preds, val.iterrows(), strict=False):
            use_model = conf >= theta and f"{s}_time_s" in val.columns
            chosen = s if use_model else fallback_solver
            cov += int(use_model)
            t = float(row[f"{chosen}_time_s"])
            total += t if bool(row.get(f"{chosen}_solved", True)) else 2.0 * t
        par2 = total / max(len(val), 1)
        curve.append({"theta": theta, "val_par2": par2, "coverage": cov / max(len(val), 1)})
        if par2 < best[0]:
            best = (par2, theta)

    result = {
        "theta": float(best[1]),
        "curve": curve,
        "best_on": "val",
        "fallback_solver": fallback_solver,
        "model_type": model_state.get("model_type"),
        "calibrated": bool(model_state.get("calibrated", False)),
    }
    out = Path(cfg["paths"]["models"]) / "theta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
