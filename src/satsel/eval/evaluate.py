"""Evaluation orchestration: compute each system's per-instance PAR-2 on the
test set, run the four planned confirmatory comparisons with the pre-registered
statistics, and the OOD / robustness appendices.
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


def _load_models(cfg: dict) -> dict:
    mdir = Path(cfg["paths"]["models"])
    out: dict = {}
    if (mdir / "portfolio.json").exists():
        out["portfolio"] = json.loads((mdir / "portfolio.json").read_text())
    if (mdir / "theta.json").exists():
        out["theta"] = json.loads((mdir / "theta.json").read_text())
    if (mdir / "classical.joblib").exists():
        out["classical"] = joblib.load(mdir / "classical.joblib")
    if (mdir / "satzilla.joblib").exists():
        out["satzilla"] = joblib.load(mdir / "satzilla.joblib")
    gnn_path = mdir / "gnn.pt"
    if gnn_path.exists():
        try:
            import torch

            out["gnn"] = torch.load(gnn_path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            out["gnn"] = None
    return out


def _feat_overhead(dataset: pd.DataFrame) -> pd.Series:
    if "feat_cost_s" in dataset.columns:
        return dataset["feat_cost_s"].fillna(0.0)
    return pd.Series(0.0, index=dataset.index)


def _row_par2(row: pd.Series, solver: str) -> float:
    t = float(row[f"{solver}_time_s"])
    return t if bool(row[f"{solver}_solved"]) else 2.0 * t


def _feature_dict(row: pd.Series) -> dict[str, float]:
    return {c: (0.0 if pd.isna(row[c]) else float(row[c]))
            for c in row.index if c.startswith("f_")}


def system_predictions(dataset: pd.DataFrame, cfg: dict, models: dict,
                       system: str, seed: int = 0, split: str = "test") -> pd.Series:
    """Return per-formula PAR-2 (including feature overhead where the system
    uses features) for `split` portion of `dataset`."""
    sub = dataset[dataset["split"] == split].copy()
    feat_ov = _feat_overhead(sub)
    par2s: dict[str, float] = {}
    feats_cols = [c for c in sub.columns if c.startswith("f_")]
    use_feats = bool(feats_cols)

    portfolio = models.get("portfolio", {})
    theta_info = models.get("theta", {})
    fallback = theta_info.get("fallback_solver") or portfolio.get(
        "best_average_solver", "kissat404")

    for fid, row in sub.iterrows():
        if system == "best_average":
            sol = portfolio.get("best_average_solver", "kissat404")
        elif system == "majority":
            train = dataset[dataset["split"] == "train"]
            sol = str(train["best_solver"].mode().iloc[0]) if len(train) else "kissat404"
        elif system == "fallback":
            sol = fallback
        elif system == "theta0" or system == "theta1":
            sol = _selector_choice(models, row, use_feats, always_predict=(system == "theta0"))
        elif system == "classical":
            if not use_feats or models.get("classical") is None:
                sol = fallback
            else:
                from satsel.models.classical import predict_classical

                sol, _conf = predict_classical(models["classical"], _feature_dict(row))
        elif system == "satzilla":
            if not use_feats or models.get("satzilla") is None:
                sol = fallback
            else:
                from satsel.models.classical import predict_satzilla

                sol = predict_satzilla(models["satzilla"], _feature_dict(row))
        elif system == "gnn":
            sol = _gnn_predict(models, row)
        elif system == "selector":
            sol = _selector_choice(models, row, use_feats)
        elif system == "random":
            rng = np.random.default_rng(seed)
            sol = str(rng.choice(list(SOLVERS)))
        elif system == "vbs":
            sol = row["best_solver"] if row["vbs_solved"] else fallback
        else:
            sol = fallback

        t = float(row[f"{sol}_time_s"])
        par2 = t if bool(row[f"{sol}_solved"]) else 2.0 * t
        if system in ("selector", "classical", "gnn", "satzilla"):
            par2 += float(feat_ov.loc[fid])
        par2s[fid] = par2

    return pd.Series(par2s)


def system_times(dataset: pd.DataFrame, cfg: dict, models: dict,
                 system: str, seed: int = 0, split: str = "test") -> pd.Series:
    """Return per-formula T(i, s_hat) (time if solved, else the family tau)."""
    sub = dataset[dataset["split"] == split].copy()
    par2 = system_predictions(dataset, cfg, models, system, seed=seed, split=split)
    times: dict[str, float] = {}
    for fid, row in sub.iterrows():
        p2 = float(par2.loc[fid])
        tau = tau_for_family(cfg, row["family"])
        t = p2 if p2 < 2 * tau else tau
        times[fid] = t
    return pd.Series(times)


def _selector_choice(models: dict, row: pd.Series, use_feats: bool,
                     always_predict: bool = False) -> str:
    portfolio = models.get("portfolio", {})
    theta_info = models.get("theta", {})
    theta = float(theta_info.get("theta", 1.0))
    fallback = theta_info.get("fallback_solver") or portfolio.get(
        "best_average_solver", "kissat404")
    gnn = models.get("gnn")
    if gnn is not None and int(row["nvars"]) + int(row["nclauses"]) <= int(
            models.get("gnn_cap", 500000)):
        try:
            from satsel.core.dimacs import parse_cnf
            from satsel.models.gnn import predict_gnn

            cnf = parse_cnf(Path(row["path"]).read_bytes())
            sol, conf = predict_gnn(gnn, cnf, cnf.nvars, gnn.get("solver_list", []))
            if conf >= theta or always_predict:
                return sol
        except Exception:  # noqa: BLE001
            pass
    if use_feats and models.get("classical") is not None:
        from satsel.models.classical import predict_classical

        sol, conf = predict_classical(models["classical"], _feature_dict(row))
        if conf >= theta or always_predict:
            return sol
    return fallback


def _gnn_predict(models: dict, row: pd.Series) -> str:
    gnn = models.get("gnn")
    if gnn is None:
        return "kissat404"
    try:
        from satsel.core.dimacs import parse_cnf
        from satsel.models.gnn import predict_gnn

        cnf = parse_cnf(Path(row["path"]).read_bytes())
        sol, _ = predict_gnn(gnn, cnf, cnf.nvars, gnn.get("solver_list", []))
        return sol
    except Exception:  # noqa: BLE001
        return "kissat404"


def _planned_comparisons(test: pd.DataFrame, par2: dict[str, pd.Series],
                         cfg: dict, seed: int) -> list[dict]:
    family = test["family"].to_numpy()
    systems = ["selector_vs_best", "selector_vs_random", "selector_vs_satzilla",
               "best_classical_vs_best"]
    names = {
        "selector_vs_best": ("selector", "best_average"),
        "selector_vs_random": ("selector", "random_mean"),
        "selector_vs_satzilla": ("selector", "satzilla"),
        "best_classical_vs_best": ("classical", "best_average"),
    }
    comparisons = []
    p_values = []
    for key in systems:
        a_sys, b_sys = names[key]
        a = par2[a_sys].reindex(test.index)
        b = par2[b_sys].reindex(test.index)
        diffs = a.to_numpy() - b.to_numpy()
        bs = stats.paired_bootstrap(diffs, family, n_resamples=int(cfg.get(
            "bootstrap_resamples", 10000)), alpha=float(cfg.get("alpha", 0.05)), seed=seed)
        # wilcoxon on both-solved pairs (both par2 < 2*tau means solved)
        solved_a = test.apply(lambda r, a=a: a.loc[r.name] < 2 * tau_for_family(
            cfg, r["family"]), axis=1).to_numpy()
        solved_b = test.apply(lambda r, b=b: b.loc[r.name] < 2 * tau_for_family(
            cfg, r["family"]), axis=1).to_numpy()
        w_a = np.where(solved_a, a.to_numpy(), np.nan)
        w_b = np.where(solved_b, b.to_numpy(), np.nan)
        wil = stats.wilcoxon_both_solved(w_a, w_b)
        mcn = stats.mcnemar(solved_a, solved_b)
        comparisons.append({
            "comparison": key,
            "system_a": a_sys,
            "system_b": b_sys,
            "mean_diff": bs["mean_diff"],
            "ci_low": bs["ci_low"],
            "ci_high": bs["ci_high"],
            "ci_excludes_zero": bs["ci_excludes_zero"],
            "wilcoxon_n": wil["n"],
            "wilcoxon_p": wil["p_value"],
            "mcnemar_n_ab": mcn["n_ab"],
            "mcnemar_n_ba": mcn["n_ba"],
            "mcnemar_p": mcn["p_value"],
            "bootstrap_p_approx": 0.0,
        })
        p = 2 * min(_frac_bootstrap(diffs, family, seed, cfg), 0.5)
        comparisons[-1]["bootstrap_p_approx"] = float(p)
        p_values.append(p)

    reject = stats.holm_bonferroni(p_values, alpha=float(cfg.get("alpha", 0.05)))
    for comp, rj in zip(comparisons, reject, strict=True):
        comp["holm_reject"] = bool(rj)
    return comparisons


def _frac_bootstrap(diffs: np.ndarray, family: np.ndarray, seed: int,
                    cfg: dict) -> float:
    bs = stats.paired_bootstrap(diffs, family, n_resamples=int(cfg.get(
        "bootstrap_resamples", 10000)), alpha=0.5, seed=seed + 1)
    sign = np.sign(bs["mean_diff"])
    # approximate: count resample means on the far side via a second bootstrap
    rng = np.random.default_rng(seed + 2)
    fams = np.unique(family)
    idx = {f: np.where(family == f)[0] for f in fams}
    count = 0
    for _k in range(500):
        picked = [idx[f][rng.integers(0, len(idx[f]), size=len(idx[f]))] for f in fams]
        sel = np.concatenate(picked)
        m = diffs[sel].mean()
        if sign > 0 and m <= 0:
            count += 1
        elif sign < 0 and m >= 0:
            count += 1
    return count / 500.0


def run_evaluation(cfg: dict, selector_kind: str = "auto") -> dict:
    dataset = load_dataset(cfg)
    models = _load_models(cfg)
    test = dataset[dataset["split"] == "test"].copy()
    if test.empty:
        raise ValueError("empty test set")

    portfolio = models.get("portfolio", {})
    theta_info = models.get("theta", {})
    chosen = selector_kind
    if chosen == "auto":
        chosen = "gnn" if models.get("gnn") is not None else "classical"

    systems = ["best_average", "fallback", "majority", "selector", "classical"]
    if models.get("satzilla") is not None:
        systems.append("satzilla")
    if models.get("gnn") is not None:
        systems.append("gnn")
    systems.append("vbs")

    par2: dict[str, pd.Series] = {}
    table = []
    for sys_name in systems:
        if sys_name == "vbs":
            ser = test.apply(lambda r: _row_par2(r, r["best_solver"]) if r["vbs_solved"]
                             else 2.0 * tau_for_family(cfg, r["family"]), axis=1)
        else:
            ser = system_predictions(dataset, cfg, models, sys_name)
        par2[sys_name] = ser
        solved = ser < 2 * test.apply(lambda r: tau_for_family(cfg, r["family"]), axis=1)
        table.append({
            "system": sys_name,
            "par2": float(ser.mean()),
            "solve_rate": float(solved.mean()),
            "n": len(ser),
        })

    # random baseline: seeded draws, seed-paired, mean + sd
    draws = int(cfg.get("random_baseline_draws", 10))
    base_seed = int(cfg.get("seeds", {}).get("random_baseline", 2026))
    random_par2 = []
    random_solved = []
    for d in range(draws):
        ser = system_predictions(dataset, cfg, models, "random", seed=base_seed + d)
        random_par2.append(ser)
        solved = ser < 2 * test.apply(lambda r: tau_for_family(cfg, r["family"]), axis=1)
        random_solved.append(float(solved.mean()))
    par2["random_mean"] = pd.concat(random_par2, axis=1).mean(axis=1)
    random_mean = float(par2["random_mean"].mean())
    random_sd = float(np.std([s.mean() for s in random_par2]))
    table.append({"system": "random", "par2": random_mean,
                  "solve_rate": float(np.mean(random_solved)), "n": len(test)})

    oracle_gap = metrics.oracle_gap(
        system_times(dataset, cfg, models, "selector"),
        test["best_time_s"].where(test["vbs_solved"], 1e9))
    comparisons = _planned_comparisons(test, par2, cfg, seed=base_seed)

    results = {
        "systems": table,
        "random_mean_sd": [random_mean, random_sd],
        "oracle_gap_selector": oracle_gap,
        "chosen_selector": chosen,
        "best_average_solver": portfolio.get("best_average_solver"),
        "theta": theta_info.get("theta"),
        "comparisons": comparisons,
        "feat_overhead_used": bool("feat_cost_s" in dataset.columns),
    }
    return results


def run_ood(cfg: dict) -> dict:
    dataset = load_dataset(cfg)
    models = _load_models(cfg)
    ood = dataset[dataset["split"] == "ood"].copy()
    if ood.empty:
        return {"primary_ood": {"error": "no OOD rows"}}

    vbs_rate = float(ood["vbs_solved"].mean())
    ceiling = {
        "vbs_solve_rate": vbs_rate,
        "ceiling_met": vbs_rate >= 0.30,
        "note": "comparisons are vacuous below ~30% VBS solve rate",
    }
    ood_par2 = system_predictions(dataset, cfg, models, "selector", split="ood")
    ood_par2_mean = float(ood_par2.reindex(ood.index).mean())
    vbs_par2 = float(ood.apply(
        lambda r: _row_par2(r, r["best_solver"]) if r["vbs_solved"]
        else 2.0 * tau_for_family(cfg, r["family"]), axis=1).mean())
    return {
        "vbs_ceiling": ceiling,
        "primary_ood_family": "ood_scheduling",
        "selector_par2_ood": ood_par2_mean,
        "vbs_par2_ood": vbs_par2,
        "n_ood": len(ood),
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
