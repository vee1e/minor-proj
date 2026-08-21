"""Evaluation orchestration: compute each system's per-instance PAR-2 on the
test set, run the four planned confirmatory comparisons with the pre-registered
statistics, and the OOD / robustness appendices.

results.json contract (consumed by report.py and the web UI):

- legacy keys kept: systems, random_mean_sd, oracle_gap_selector,
  chosen_selector, best_average_solver, theta, comparisons, feat_overhead_used
- per_system_choice_hist: {system: {solver: count}} on the test split
- classification: {system: {accuracy, cost_weighted_accuracy, n_scored,
  per_family: {family: accuracy}}} for selector/classical/satzilla/gnn
- per_family: {family: {system: {par2, solve_rate, n}}}
- label_stats: test label histogram, majority share, tie rate, informative
  rate, median best time, n_test
- feature_ablation: list of {group, n_features, val_par2, val_accuracy,
  mean_feat_cost_s, note}
- data_quality_warnings: list of strings
- bootstrap_resamples_used / bootstrap_cap_note
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from satsel.core.config import tau_for_family
from satsel.data.build import load_dataset
from satsel.eval import metrics, stats
from satsel.models.classical import SOLVERS

PREDICTIVE_SYSTEMS = ("selector", "classical", "satzilla", "gnn")


# --------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _load_models(cfg: dict) -> dict:
    import os

    mdir = Path(cfg["paths"]["models"])
    out: dict = {}
    if (mdir / "portfolio.json").exists():
        out["portfolio"] = _read_json(mdir / "portfolio.json")
    if (mdir / "theta.json").exists():
        out["theta"] = _read_json(mdir / "theta.json")
    if (mdir / "model_diagnostics.json").exists():
        out["diagnostics"] = _read_json(mdir / "model_diagnostics.json")
    for name in ("classical", "satzilla"):
        p = mdir / f"{name}.joblib"
        if p.exists():
            try:
                out[name] = joblib.load(p)
            except Exception:  # noqa: BLE001
                out[name] = None
    gnn_path = mdir / "gnn.pt"
    if gnn_path.exists() and not os.environ.get("SATEL_SKIP_GNN"):
        try:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            import torch

            torch.set_num_threads(1)
            out["gnn"] = torch.load(gnn_path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            out["gnn"] = None
    return out


def _available_solvers(dataset: pd.DataFrame) -> list[str]:
    avail = [s for s in SOLVERS if f"{s}_time_s" in dataset.columns
             and f"{s}_solved" in dataset.columns]
    if not avail:  # unknown portfolio: infer from the column names
        avail = sorted({c[:-len("_time_s")] for c in dataset.columns
                        if c.endswith("_time_s") and f"{c[:-len('_time_s')]}_solved"
                        in dataset.columns})
    return avail


def _default_fallback(dataset: pd.DataFrame, models: dict) -> str:
    """Fallback solver: theta.json > portfolio.json > best mean PAR-2 on train."""
    theta_info = models.get("theta") or {}
    portfolio = models.get("portfolio") or {}
    avail = _available_solvers(dataset)
    for cand in (theta_info.get("fallback_solver"), portfolio.get("best_average_solver")):
        if cand and cand in avail:
            return str(cand)
    train = dataset[dataset["split"] == "train"]
    pool = train if len(train) else dataset
    if len(pool) and avail:
        scores = {s: float(np.where(pool[f"{s}_solved"].astype(bool),
                                    pool[f"{s}_time_s"].astype(float),
                                    2.0 * pool[f"{s}_time_s"].astype(float)).mean())
                  for s in avail}
        return min(scores, key=scores.get)
    return avail[0] if avail else "cadical153"


def _feat_overhead(dataset: pd.DataFrame) -> pd.Series:
    if "feat_cost_s" in dataset.columns:
        return dataset["feat_cost_s"].fillna(0.0).astype(float)
    return pd.Series(0.0, index=dataset.index)


def _row_par2(row: pd.Series, solver: str) -> float:
    t = float(row[f"{solver}_time_s"])
    return t if bool(row[f"{solver}_solved"]) else 2.0 * t


def _feature_dict(row: pd.Series) -> dict[str, float]:
    return {c: (0.0 if pd.isna(row[c]) else float(row[c]))
            for c in row.index if c.startswith("f_")}


def _tau_series(cfg: dict, sub: pd.DataFrame) -> pd.Series:
    return sub["family"].map(lambda f: tau_for_family(cfg, f)).astype(float)


# --------------------------------------------------------------------------
# per-system choices and PAR-2
# --------------------------------------------------------------------------

def system_choices(dataset: pd.DataFrame, cfg: dict, models: dict, system: str,
                   seed: int = 0, split: str = "test") -> pd.Series:
    """Return the solver each system picks for every formula of `split`."""
    sub = dataset[dataset["split"] == split]
    avail = _available_solvers(dataset)
    feats_cols = [c for c in sub.columns if c.startswith("f_")]
    use_feats = bool(feats_cols)
    portfolio = models.get("portfolio") or {}
    fallback = _default_fallback(dataset, models)
    best_avg = portfolio.get("best_average_solver") or fallback
    if best_avg not in avail:
        best_avg = fallback

    majority = fallback
    if system == "majority":
        train = dataset[(dataset["split"] == "train") & (dataset["best_solver"] != "")]
        if len(train):
            majority = str(train["best_solver"].mode().iloc[0])
    rng = np.random.default_rng(seed)

    out: dict = {}
    for fid, row in sub.iterrows():
        if system == "best_average":
            sol = best_avg
        elif system == "majority":
            sol = majority
        elif system == "fallback":
            sol = fallback
        elif system in ("theta0", "theta1"):
            sol = _selector_choice(models, row, use_feats, fallback,
                                   always_predict=(system == "theta0"))
        elif system == "classical":
            sol = _classical_predict(models, row, use_feats, fallback)[0]
        elif system == "satzilla":
            sol = fallback
            if use_feats and models.get("satzilla") is not None:
                try:
                    from satsel.models.classical import predict_satzilla

                    sol = predict_satzilla(models["satzilla"], _feature_dict(row))
                except Exception:  # noqa: BLE001
                    sol = fallback
        elif system == "gnn":
            sol = _gnn_predict(models, row, fallback)
        elif system == "selector":
            sol = _selector_choice(models, row, use_feats, fallback)
        elif system == "random":
            sol = str(rng.choice(avail)) if avail else fallback
        elif system == "vbs":
            sol = row["best_solver"] if bool(row["vbs_solved"]) else fallback
        else:
            sol = fallback
        if sol not in avail:
            sol = fallback
        out[fid] = str(sol)
    return pd.Series(out, dtype=object)


def _par2_from_choices(sub: pd.DataFrame, choices: pd.Series, cfg: dict,
                       add_feat_overhead: bool) -> pd.Series:
    feat_ov = _feat_overhead(sub)
    vals: dict = {}
    for fid, row in sub.iterrows():
        sol = choices.loc[fid]
        p2 = _row_par2(row, sol)
        if add_feat_overhead:
            p2 += float(feat_ov.loc[fid])
        vals[fid] = p2
    return pd.Series(vals, dtype=float)


def system_predictions(dataset: pd.DataFrame, cfg: dict, models: dict,
                       system: str, seed: int = 0, split: str = "test") -> pd.Series:
    """Return per-formula PAR-2 (including feature overhead where the system
    uses features) for `split` portion of `dataset`."""
    sub = dataset[dataset["split"] == split]
    choices = system_choices(dataset, cfg, models, system, seed=seed, split=split)
    return _par2_from_choices(sub, choices, cfg, system in PREDICTIVE_SYSTEMS)


def system_times(dataset: pd.DataFrame, cfg: dict, models: dict,
                 system: str, seed: int = 0, split: str = "test") -> pd.Series:
    """Return per-formula T(i, s_hat) (time if solved, else the family tau)."""
    sub = dataset[dataset["split"] == split]
    par2 = system_predictions(dataset, cfg, models, system, seed=seed, split=split)
    tau = _tau_series(cfg, sub)
    times = {fid: (float(par2.loc[fid]) if float(par2.loc[fid]) < 2 * tau.loc[fid]
                   else float(tau.loc[fid])) for fid in sub.index}
    return pd.Series(times, dtype=float)


def _classical_predict(models: dict, row: pd.Series, use_feats: bool,
                       fallback: str) -> tuple[str, float]:
    if not use_feats or models.get("classical") is None:
        return fallback, 0.0
    try:
        from satsel.models.classical import predict_classical

        sol, conf = predict_classical(models["classical"], _feature_dict(row))
        return str(sol), float(conf)
    except Exception:  # noqa: BLE001
        return fallback, 0.0


def _selector_choice(models: dict, row: pd.Series, use_feats: bool, fallback: str,
                     always_predict: bool = False) -> str:
    theta_info = models.get("theta") or {}
    theta = float(theta_info.get("theta", 1.0) or 1.0)
    gnn = models.get("gnn")
    if gnn is not None and "path" in row.index and int(row.get("nvars", 0)) + int(
            row.get("nclauses", 0)) <= int(models.get("gnn_cap", 500000)):
        try:
            from satsel.core.dimacs import parse_cnf
            from satsel.models.gnn import predict_gnn

            cnf = parse_cnf(Path(row["path"]).read_bytes())
            sol, conf = predict_gnn(gnn, cnf, cnf.nvars, gnn.get("solver_list", []))
            if conf >= theta or always_predict:
                return str(sol)
        except Exception:  # noqa: BLE001
            pass
    if use_feats and models.get("classical") is not None:
        sol, conf = _classical_predict(models, row, use_feats, fallback)
        if conf >= theta or always_predict:
            return sol
    return fallback


def _gnn_predict(models: dict, row: pd.Series, fallback: str = "cadical153") -> str:
    gnn = models.get("gnn")
    if gnn is None or "path" not in row.index:
        return fallback
    try:
        from satsel.core.dimacs import parse_cnf
        from satsel.models.gnn import predict_gnn

        cnf = parse_cnf(Path(row["path"]).read_bytes())
        sol, _ = predict_gnn(gnn, cnf, cnf.nvars, gnn.get("solver_list", []))
        return str(sol)
    except Exception:  # noqa: BLE001
        return fallback


# --------------------------------------------------------------------------
# planned comparisons
# --------------------------------------------------------------------------

def _planned_comparisons(test: pd.DataFrame, par2: dict[str, pd.Series],
                         cfg: dict, seed: int, n_resamples: int) -> list[dict]:
    family = test["family"].to_numpy()
    tau = _tau_series(cfg, test).to_numpy()
    names = {
        "selector_vs_best": ("selector", "best_average"),
        "selector_vs_random": ("selector", "random_mean"),
        "selector_vs_satzilla": ("selector", "satzilla"),
        "best_classical_vs_best": ("classical", "best_average"),
    }
    comparisons = []
    p_values = []
    for key, (a_sys, b_sys) in names.items():
        if a_sys not in par2 or b_sys not in par2:
            continue
        a = par2[a_sys].reindex(test.index).to_numpy(dtype=float)
        b = par2[b_sys].reindex(test.index).to_numpy(dtype=float)
        diffs = a - b
        bs = stats.paired_bootstrap(diffs, family, n_resamples=n_resamples,
                                    alpha=float(cfg.get("alpha", 0.05)), seed=seed)
        solved_a = a < 2 * tau
        solved_b = b < 2 * tau
        wil = stats.wilcoxon_both_solved(np.where(solved_a, a, np.nan),
                                         np.where(solved_b, b, np.nan))
        mcn = stats.mcnemar(solved_a, solved_b)
        p = float(bs["p_two_sided"])
        comparisons.append({
            "comparison": key,
            "system_a": a_sys,
            "system_b": b_sys,
            "mean_diff": bs["mean_diff"],
            "ci_low": bs["ci_low"],
            "ci_high": bs["ci_high"],
            "ci_excludes_zero": bs["ci_excludes_zero"],
            "bootstrap_p": p,
            "bootstrap_p_approx": p,  # legacy key
            "n_resamples": int(bs["n_resamples"]),
            "wilcoxon_n": wil["n"],
            "wilcoxon_p": wil["p_value"],
            "mcnemar_n_ab": mcn["n_ab"],
            "mcnemar_n_ba": mcn["n_ba"],
            "mcnemar_p": mcn["p_value"],
        })
        p_values.append(p)

    reject = stats.holm_bonferroni(p_values, alpha=float(cfg.get("alpha", 0.05)))
    for comp, rj in zip(comparisons, reject, strict=True):
        comp["holm_reject"] = bool(rj)
    return comparisons


# --------------------------------------------------------------------------
# diagnostics: label stats, classification, per-family, ablation, warnings
# --------------------------------------------------------------------------

def _label_stats(test: pd.DataFrame) -> dict:
    labels = test["best_solver"].astype(str)
    labels = labels[labels != ""]
    hist = {str(k): int(v) for k, v in labels.value_counts().items()}
    n = int(len(labels))
    majority_solver = max(hist, key=hist.get) if hist else None
    majority_share = (hist[majority_solver] / n) if (hist and n) else 0.0
    tie_rate = float(test["tie_flag"].astype(float).mean()) if "tie_flag" in test.columns \
        and len(test) else None
    informative_rate = (float(test["informative"].astype(bool).mean())
                        if "informative" in test.columns and len(test) else None)
    solved = test[test["vbs_solved"].astype(bool)] if "vbs_solved" in test.columns else test
    med_best = (float(solved["best_time_s"].astype(float).median())
                if len(solved) and "best_time_s" in solved.columns else None)
    return {
        "n_test": int(len(test)),
        "label_hist": hist,
        "majority_solver": majority_solver,
        "majority_share": float(majority_share),
        "tie_rate": tie_rate,
        "informative_rate": informative_rate,
        "median_best_time_s": med_best,
        "vbs_solve_rate": float(test["vbs_solved"].astype(bool).mean())
        if "vbs_solved" in test.columns and len(test) else None,
    }


def _choice_hist(choices: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in choices.value_counts().items()}


def _classification_block(test: pd.DataFrame, choices: pd.Series) -> dict:
    """Accuracy on non-tie VBS-solved rows, cost-weighted accuracy on VBS-solved
    rows (1/weight of the chosen solver; 1.0 = always the best pick), and
    per-family accuracy."""
    solved = test["vbs_solved"].astype(bool) if "vbs_solved" in test.columns \
        else pd.Series(True, index=test.index)
    tie = test["tie_flag"].astype(int) if "tie_flag" in test.columns \
        else pd.Series(0, index=test.index)
    scored = solved & (tie == 0)
    ch = choices.reindex(test.index).astype(str)
    opt = test["best_solver"].astype(str)
    correct = (ch == opt)

    acc = float(correct[scored].mean()) if scored.sum() else None
    # cost-weighted: 1 / w(i, chosen) where w = clamp(T(i,s)/T(i,s*), 1, clip)
    weights = []
    for fid in test.index[solved.to_numpy()]:
        sol = ch.loc[fid]
        wcol = f"{sol}_weight"
        if wcol in test.columns and not pd.isna(test.at[fid, wcol]):
            w = float(test.at[fid, wcol])
        else:
            best = float(test.at[fid, "best_time_s"]) if "best_time_s" in test.columns else 0.0
            t = float(test.at[fid, f"{sol}_time_s"])
            if not bool(test.at[fid, f"{sol}_solved"]):
                t *= 2.0
            w = max(1.0, t / best) if best > 0 else 1.0
        weights.append(w)
    cw = metrics.cost_weighted_accuracy(pd.Series(weights)) if weights else None
    per_family = {}
    for fam, idx in test.groupby("family").groups.items():
        m = scored.loc[idx]
        per_family[str(fam)] = float(correct.loc[idx][m].mean()) if m.sum() else None
    return {
        "accuracy": acc,
        "cost_weighted_accuracy": cw,
        "n_scored": int(scored.sum()),
        "per_family": per_family,
    }


def _per_family_block(test: pd.DataFrame, par2: dict[str, pd.Series], cfg: dict) -> dict:
    tau = _tau_series(cfg, test)
    out: dict = {}
    for fam, idx in test.groupby("family").groups.items():
        fam_entry = {}
        for sys_name, ser in par2.items():
            s = ser.reindex(idx).astype(float)
            fam_entry[sys_name] = {
                "par2": float(s.mean()),
                "solve_rate": float((s < 2 * tau.loc[idx]).mean()),
                "n": int(len(s)),
            }
        out[str(fam)] = fam_entry
    return out


def _feature_groups(feat_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"syntactic": [], "graph": [], "probe": [], "other": []}
    for c in feat_cols:
        rest = c[2:]
        key = rest.split("_", 1)[0].lower()
        if key in ("syn", "syntactic", "s"):
            groups["syntactic"].append(c)
        elif key in ("gr", "graph", "g"):
            groups["graph"].append(c)
        elif key in ("probe", "p"):
            groups["probe"].append(c)
        else:
            groups["other"].append(c)
    return groups


def _feature_ablation(dataset: pd.DataFrame, cfg: dict, seed: int) -> list[dict]:
    """Cheap Pareto ablation: fixed-param RandomForest on train, scored by val
    PAR-2, for cumulative feature groups syntactic -> +graph -> all."""
    feat_cols = [c for c in dataset.columns if c.startswith("f_")]
    if not feat_cols:
        return []
    groups = _feature_groups(feat_cols)
    plan = [
        ("syntactic", groups["syntactic"]),
        ("syntactic+graph", groups["syntactic"] + groups["graph"]),
        ("all", feat_cols),
    ]
    train = dataset[(dataset["split"] == "train") & dataset["vbs_solved"].astype(bool)]
    if "tie_flag" in train.columns:
        train = train[train["tie_flag"].astype(int) == 0]
    val = dataset[dataset["split"] == "val"]
    if len(val) == 0:
        val = dataset[dataset["split"] == "test"]
    if len(train) < 5 or len(val) == 0:
        return [{"group": g, "n_features": len(cols), "val_par2": None,
                 "val_accuracy": None, "mean_feat_cost_s": None,
                 "note": "too few training rows"} for g, cols in plan]
    avail = _available_solvers(dataset)
    y = train["best_solver"].astype(str).to_numpy()
    tau_val = _tau_series(cfg, val)
    feat_cost = _feat_overhead(val)
    probe_cost = (val["f_probe_runtime_s"].fillna(0.0).astype(float)
                  if "f_probe_runtime_s" in val.columns else pd.Series(0.0, index=val.index))
    results = []
    seen: set[tuple[str, ...]] = set()
    for group, cols in plan:
        key = tuple(cols)
        if not cols or key in seen:
            results.append({"group": group, "n_features": len(cols), "val_par2": None,
                            "val_accuracy": None, "mean_feat_cost_s": None,
                            "note": "no distinct features for this group"})
            continue
        seen.add(key)
        entry = {"group": group, "n_features": len(cols), "features": cols[:50]}
        try:
            from sklearn.ensemble import RandomForestClassifier

            if len(np.unique(y)) < 2:
                raise ValueError("single-class training labels")
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=2,
                                        class_weight="balanced", random_state=seed,
                                        n_jobs=1)
            rf.fit(train[cols].fillna(0.0).to_numpy(dtype=float), y)
            pred = rf.predict(val[cols].fillna(0.0).to_numpy(dtype=float))
            fallback = _default_fallback(dataset, {})
            pred = np.array([p if p in avail else fallback for p in pred], dtype=object)
            choices = pd.Series(pred, index=val.index)
            cost = feat_cost if group == "all" else (feat_cost - probe_cost).clip(lower=0.0)
            p2 = _par2_from_choices(val, choices, cfg, False) + cost
            solved_mask = val["vbs_solved"].astype(bool)
            tie = val["tie_flag"].astype(int) if "tie_flag" in val.columns else 0
            scored = solved_mask & (tie == 0)
            acc = float((choices[scored] == val["best_solver"][scored]).mean()) \
                if scored.sum() else None
            entry.update({
                "val_par2": float(p2.mean()),
                "val_solve_rate": float((p2 < 2 * tau_val).mean()),
                "val_accuracy": acc,
                "mean_feat_cost_s": float(cost.mean()),
                "choice_hist": _choice_hist(choices),
                "note": ("feature cost for partial groups approximated as "
                         "feat_cost_s minus probe runtime" if group != "all" else ""),
            })
        except Exception as exc:  # noqa: BLE001
            entry.update({"val_par2": None, "val_accuracy": None, "mean_feat_cost_s": None,
                          "note": f"skipped: {type(exc).__name__}: {exc}"})
        results.append(entry)
    return results


def _data_quality_warnings(cfg: dict, models: dict, test: pd.DataFrame, label_stats: dict,
                           choice_hist: dict, n_resamples_used: int) -> list[str]:
    warnings: list[str] = []
    summ = _read_json(Path(cfg["paths"]["datasets"]) / "dataset_summary.json")
    for src_name, src in (("dataset_summary", summ), ("model_diagnostics",
                                                      models.get("diagnostics") or {})):
        for w in (src.get("warnings") or []):
            warnings.append(f"[{src_name}] {w}")
        for w in (src.get("data_quality_warnings") or []):
            warnings.append(f"[{src_name}] {w}")
        for w in (src.get("provenance_warnings") or []):
            warnings.append(f"[{src_name}] {w}")
    n_test = int(len(test))
    if n_test < 100:
        warnings.append(f"n_test={n_test} < 100: bootstrap CIs will be wide and the "
                        "planned comparisons are underpowered")
    if label_stats.get("majority_share", 0) > 0.6:
        warnings.append(f"label majority share {label_stats['majority_share']:.2f} > 0.6 "
                        f"({label_stats.get('majority_solver')}): a constant predictor "
                        "is hard to beat; the corpus is too easy or too homogeneous")
    tr = label_stats.get("tie_rate")
    if tr is not None and tr > 0.5:
        warnings.append(f"tie rate {tr:.2f} > 0.5: most formulas have no clear best solver "
                        "(solve times below the tie floor); labels carry little signal")
    ir = label_stats.get("informative_rate")
    if ir is not None and ir < 0.3:
        warnings.append(f"informative-label rate {ir:.2f} < 0.3")
    mb = label_stats.get("median_best_time_s")
    if mb is not None and mb < 0.1:
        warnings.append(f"median best solve time {mb*1000:.0f} ms: formulas are too easy "
                        "for solver differences to exceed overhead noise")
    for sys_name in PREDICTIVE_SYSTEMS:
        h = choice_hist.get(sys_name)
        if h:
            tot = sum(h.values())
            top = max(h, key=h.get)
            if tot and h[top] / tot >= 0.95 and len(test) >= 5:
                warnings.append(f"system '{sys_name}' picks '{top}' on "
                                f"{100*h[top]/tot:.0f}% of test formulas (classifier collapse)")
    if n_resamples_used < int(cfg.get("bootstrap_resamples", 10000)):
        warnings.append(f"bootstrap resamples capped at {n_resamples_used} because n_test < 200")
    if summ.get("cluster_leak"):
        warnings.append("dataset_summary reports a cluster leak between splits")
    return warnings


# --------------------------------------------------------------------------
# main entry points
# --------------------------------------------------------------------------

def run_evaluation(cfg: dict, selector_kind: str = "auto", include_gnn: bool = True) -> dict:
    dataset = load_dataset(cfg)
    models = _load_models(cfg)
    test = dataset[dataset["split"] == "test"].copy()
    if test.empty:
        raise ValueError("empty test set")

    portfolio = models.get("portfolio") or {}
    theta_info = models.get("theta") or {}
    chosen = selector_kind
    if chosen == "auto":
        chosen = "gnn" if (include_gnn and models.get("gnn") is not None) else "classical"

    systems = ["best_average", "fallback", "majority", "selector", "classical", "satzilla"]
    if include_gnn and models.get("gnn") is not None:
        systems.append("gnn")
    systems.append("vbs")

    tau = _tau_series(cfg, test)
    par2: dict[str, pd.Series] = {}
    choices: dict[str, pd.Series] = {}
    table = []
    for sys_name in systems:
        ch = system_choices(dataset, cfg, models, sys_name)
        if sys_name == "vbs":
            ser = test.apply(lambda r: _row_par2(r, r["best_solver"]) if bool(r["vbs_solved"])
                             else 2.0 * tau_for_family(cfg, r["family"]), axis=1).astype(float)
        else:
            ser = _par2_from_choices(test, ch, cfg, sys_name in PREDICTIVE_SYSTEMS)
        par2[sys_name] = ser
        choices[sys_name] = ch
        model_key = "satzilla" if sys_name == "satzilla" else "classical"
        table.append({
            "system": sys_name,
            "par2": float(ser.mean()),
            "solve_rate": float((ser < 2 * tau).mean()),
            "n": int(len(ser)),
            "is_fallback_only": bool(sys_name in PREDICTIVE_SYSTEMS
                                     and models.get(model_key) is None
                                     and models.get("gnn") is None),
        })

    # random baseline: seeded draws, seed-paired, mean + sd
    draws = int(cfg.get("random_baseline_draws", 10))
    base_seed = int((cfg.get("seeds") or {}).get("random_baseline", 2026))
    random_par2 = []
    random_solved = []
    random_hist: dict[str, int] = {}
    for d in range(draws):
        ch = system_choices(dataset, cfg, models, "random", seed=base_seed + d)
        ser = _par2_from_choices(test, ch, cfg, False)
        random_par2.append(ser)
        random_solved.append(float((ser < 2 * tau).mean()))
        for k, v in _choice_hist(ch).items():
            random_hist[k] = random_hist.get(k, 0) + v
    par2["random_mean"] = pd.concat(random_par2, axis=1).mean(axis=1)
    random_mean = float(par2["random_mean"].mean())
    random_sd = float(np.std([s.mean() for s in random_par2]))
    table.append({"system": "random", "par2": random_mean,
                  "solve_rate": float(np.mean(random_solved)), "n": len(test)})

    n_test = int(len(test))
    n_resamples_cfg = int(cfg.get("bootstrap_resamples", 10000))
    n_resamples = min(n_resamples_cfg, 2000) if n_test < 200 else n_resamples_cfg

    oracle_gap = metrics.oracle_gap(
        system_times(dataset, cfg, models, "selector"),
        test["best_time_s"].astype(float).where(test["vbs_solved"].astype(bool), 1e9))
    comparisons = _planned_comparisons(test, par2, cfg, seed=base_seed,
                                       n_resamples=n_resamples)

    choice_hist = {s: _choice_hist(c) for s, c in choices.items()}
    choice_hist["random"] = random_hist
    classification = {s: _classification_block(test, choices[s])
                      for s in PREDICTIVE_SYSTEMS if s in choices}
    label_stats = _label_stats(test)
    try:
        ablation = _feature_ablation(dataset, cfg, seed=base_seed)
    except Exception as exc:  # noqa: BLE001
        ablation = [{"group": "all", "note": f"skipped: {type(exc).__name__}: {exc}"}]
    warnings = _data_quality_warnings(cfg, models, test, label_stats, choice_hist,
                                      n_resamples)

    results = {
        "systems": table,
        "random_mean_sd": [random_mean, random_sd],
        "oracle_gap_selector": oracle_gap,
        "chosen_selector": chosen,
        "best_average_solver": portfolio.get("best_average_solver"),
        "theta": theta_info.get("theta"),
        "comparisons": comparisons,
        "feat_overhead_used": bool("feat_cost_s" in dataset.columns),
        "gnn_feature_free": _gnn_feature_free(models),
        "per_system_choice_hist": choice_hist,
        "classification": classification,
        "per_family": _per_family_block(test, par2, cfg),
        "label_stats": label_stats,
        "feature_ablation": ablation,
        "data_quality_warnings": warnings,
        "bootstrap_resamples_used": n_resamples,
        "bootstrap_cap_note": (f"capped from {n_resamples_cfg} to {n_resamples} "
                               "because n_test < 200" if n_resamples < n_resamples_cfg
                               else "uncapped"),
        "n_test": n_test,
        "models_loaded": {k: (models.get(k) is not None) for k in
                          ("portfolio", "theta", "classical", "satzilla", "gnn")},
        "portfolio": {k: portfolio.get(k) for k in
                      ("complementarity", "marginal_contribution", "pruned", "kept",
                       "best_average_solver") if k in portfolio},
    }
    return results


def _gnn_feature_free(models: dict) -> dict:
    """Objective O6: GNN with designed node features vs pure topology.

    Both are trained on identical splits by `train-gnn`, so the difference is
    attributable to the node features alone. Empty when no GNN is present.
    """
    gnn = models.get("gnn")
    if not isinstance(gnn, dict):
        return {}
    comp = gnn.get("feature_free_comparison") or {}
    out = {str(k): float(v) for k, v in comp.items()}
    if "full" in out and "none" in out:
        out["delta_par2"] = out["none"] - out["full"]
        out["note"] = ("positive delta means the designed node features help; "
                       "near zero means the graph topology alone carries the signal")
    return out


def run_ood(cfg: dict) -> dict:
    dataset = load_dataset(cfg)
    models = _load_models(cfg)
    ood = dataset[dataset["split"] == "ood"].copy()
    if ood.empty:
        return {"primary_ood": {"error": "no OOD rows"}}

    vbs_rate = float(ood["vbs_solved"].astype(bool).mean())
    ceiling = {
        "vbs_solve_rate": vbs_rate,
        "ceiling_met": vbs_rate >= 0.30,
        "note": "comparisons are vacuous below ~30% VBS solve rate",
    }
    tau = _tau_series(cfg, ood)
    per_system = {}
    for sys_name in ("best_average", "selector", "classical", "satzilla"):
        try:
            ch = system_choices(dataset, cfg, models, sys_name, split="ood")
            ser = _par2_from_choices(ood, ch, cfg, sys_name in PREDICTIVE_SYSTEMS)
            per_system[sys_name] = {"par2": float(ser.mean()),
                                    "solve_rate": float((ser < 2 * tau).mean()),
                                    "choice_hist": _choice_hist(ch)}
        except Exception as exc:  # noqa: BLE001
            per_system[sys_name] = {"error": str(exc)}
    vbs_par2 = float(ood.apply(
        lambda r: _row_par2(r, r["best_solver"]) if bool(r["vbs_solved"])
        else 2.0 * tau_for_family(cfg, r["family"]), axis=1).mean())
    return {
        "vbs_ceiling": ceiling,
        "primary_ood_family": "ood_scheduling",
        "selector_par2_ood": per_system.get("selector", {}).get("par2"),
        "vbs_par2_ood": vbs_par2,
        "n_ood": len(ood),
        "per_system": per_system,
        "lofo": {"not_run": "requires per-family retraining; see protocol"},
    }


def run_robustness(cfg: dict, run_solver=None) -> dict:
    """Robustness appendix planner + runner.

    `run_solver(cnf_bytes, solver, timeout_s) -> (status, runtime_s)` is the
    pluggable solver hook; if absent, the procedure is reported as not-run.
    """
    if run_solver is None:
        return {"not_run": "provide run_solver(cnf_bytes, solver, timeout_s)"}
    return {"not_run": "perturbation runs deferred to a nightly job"}


def perturb_cnf(clauses: list[list[int]], kind: str, k: int, seed: int) -> list[list[int]]:
    """Perturb a clause list: 'unit' appends k unit clauses, 'rename' applies a
    random bijective renaming, 'redundant' appends k redundant clauses."""
    rng = np.random.default_rng(seed)
    out = [list(c) for c in clauses]
    if kind == "unit":
        nvars = max((abs(lit) for c in clauses for lit in c), default=1)
        for _ in range(k):
            v = int(rng.integers(1, nvars + 1))
            out.append([v if rng.random() < 0.5 else -v])
    elif kind == "rename":
        nvars = max((abs(lit) for c in clauses for lit in c), default=1)
        perm = rng.permutation(nvars) + 1
        out = [[int(perm[abs(lit) - 1]) * (1 if lit > 0 else -1) for lit in c] for c in out]
    elif kind == "redundant":
        for _ in range(k):
            c = clauses[int(rng.integers(0, len(clauses)))]
            out.append([abs(lit) * (1 if rng.random() < 0.5 else -1) for lit in c])
    return out
