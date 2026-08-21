"""Report writer: renders the evaluation results to a markdown report +
results JSON in artifacts/reports/.

Sections (in order): data quality & warnings, label distribution, portfolio
complementarity, systems table, planned comparisons, per-family table,
classification accuracy, feature ablation, OOD, robustness, reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    try:
        if isinstance(x, bool):
            return str(x)
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x) -> str:
    return "-" if x is None else f"{100 * float(x):.1f}%"


def _hist_str(hist: dict | None, top: int = 4) -> str:
    if not hist:
        return "-"
    total = sum(hist.values()) or 1
    items = sorted(hist.items(), key=lambda kv: -kv[1])[:top]
    s = ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in items)
    return s + (" ..." if len(hist) > top else "")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out.append("")
    return out


def _config_hash(cfg: dict) -> str:
    try:
        blob = json.dumps(cfg, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return "-"


def write_report(cfg: dict, results: dict, ood: dict | None = None,
                 robustness: dict | None = None) -> str:
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)

    summ_path = Path(cfg["paths"]["datasets"]) / "dataset_summary.json"
    try:
        summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    except Exception:  # noqa: BLE001
        summ = {}

    L: list[str] = []
    L.append("# Evaluation report")
    L.append("")
    L.append("ML-driven SAT solver selection via PySAT oracle wrappers. "
             "Pre-registered protocol: `EVAL_PROTOCOL.md`.")
    L.append("")

    # 1. data quality & warnings (first, so nobody reads the numbers without them)
    L.append("## 1. Data quality & warnings")
    L.append("")
    warnings = results.get("data_quality_warnings") or []
    if warnings:
        L.append(f"{len(warnings)} warning(s). Read these before the results tables.")
        L.append("")
        for w in warnings:
            L.append(f"- WARNING: {w}")
    else:
        L.append("- No data-quality warnings raised.")
    L.append("")
    L.append(f"- ID formulas: {summ.get('n_id', '-')}, OOD formulas: {summ.get('n_ood', '-')}; "
             f"train / val / test: {summ.get('n_train', '-')} / {summ.get('n_val', '-')} / "
             f"{summ.get('n_test', results.get('n_test', '-'))}; "
             f"cluster-duplicates dropped: {summ.get('n_deduped', '-')}")
    fams = summ.get("families", {}) or {}
    if fams:
        L.append("")
        L += _table(["family", "n", "train", "val", "test", "VBS solve rate"],
                    [[fam, v.get("n", "-"), v.get("n_train", "-"), v.get("n_val", "-"),
                      v.get("n_test", "-"), _fmt(v.get("vbs_solve_rate"))]
                     for fam, v in sorted(fams.items())])
    else:
        L.append("")

    # 2. label distribution
    L.append("## 2. Label distribution (test split)")
    L.append("")
    ls = results.get("label_stats") or {}
    L.append(f"- n_test: {ls.get('n_test', '-')}; VBS solve rate: {_pct(ls.get('vbs_solve_rate'))}")
    L.append(f"- Majority label: {ls.get('majority_solver', '-')} "
             f"({_pct(ls.get('majority_share'))} of labelled rows) -- a constant predictor "
             "achieves this accuracy for free")
    L.append(f"- Tie rate: {_pct(ls.get('tie_rate'))}; informative-label rate: "
             f"{_pct(ls.get('informative_rate'))}; median best solve time: "
             f"{_fmt(ls.get('median_best_time_s'), 4)} s")
    L.append("")
    hist = ls.get("label_hist") or {}
    if hist:
        total = sum(hist.values()) or 1
        L += _table(["best solver", "count", "share"],
                    [[k, v, _pct(v / total)] for k, v in
                     sorted(hist.items(), key=lambda kv: -kv[1])])

    # 3. portfolio complementarity
    L.append("## 3. Portfolio complementarity")
    L.append("")
    pf = results.get("portfolio") or {}
    comp = pf.get("complementarity")
    marg = pf.get("marginal_contribution")
    if pf.get("best_average_solver"):
        L.append(f"- Single best solver (train PAR-2): {pf.get('best_average_solver')}")
    if isinstance(pf.get("kept"), list):
        L.append(f"- Kept after pruning: {', '.join(pf['kept'])}")
    if isinstance(pf.get("pruned"), list) and pf["pruned"]:
        L.append(f"- Pruned (dominated): {', '.join(pf['pruned'])}")
    if isinstance(marg, dict) and marg:
        L.append("")
        L.append("Marginal contribution = VBS PAR-2 increase when the solver is removed "
                 "from the portfolio (higher = more complementary).")
        L.append("")
        def _marg_key(kv):
            v = kv[1]
            if isinstance(v, (int, float)):
                return -float(v)
            return -float((v or {}).get("par2_delta", 0) or 0)

        rows = []
        for s, v in sorted(marg.items(), key=_marg_key):
            if isinstance(v, dict):
                rows.append([s] + [_fmt(v.get(k)) for k in ("par2_delta", "unique_best_share")])
            else:
                rows.append([s, _fmt(v), "-"])
        L += _table(["solver", "marginal PAR-2 delta", "unique-best share"], rows)
    elif isinstance(comp, dict) and comp:
        L.append("")
        L.append("```json")
        L.append(json.dumps(comp, indent=2, default=str)[:4000])
        L.append("```")
        L.append("")
    else:
        L.append("- (portfolio.json has no complementarity block; run the train step)")
        L.append("")

    # 4. systems table
    L.append("## 4. Systems on the test set")
    L.append("")
    ch = results.get("per_system_choice_hist") or {}
    rows = []
    for row in results.get("systems", []):
        name = row["system"]
        note = " (fallback only: no model loaded)" if row.get("is_fallback_only") else ""
        rows.append([name + note, _fmt(row.get("par2")), _pct(row.get("solve_rate")),
                     _hist_str(ch.get(name))])
    L += _table(["system", "PAR-2", "solve rate", "choice histogram"], rows)
    rms = results.get("random_mean_sd", [None, None])
    L.append(f"- Random baseline (seeded, {cfg.get('random_baseline_draws', 10)} draws): "
             f"mean {_fmt(rms[0])} SD {_fmt(rms[1])}")
    L.append(f"- Oracle gap (selector): {_fmt(results.get('oracle_gap_selector'))}")
    L.append(f"- Chosen selector: {results.get('chosen_selector')}; theta: "
             f"{_fmt(results.get('theta'), 2)}; best single solver: "
             f"{results.get('best_average_solver')}")
    L.append(f"- Feature overhead included in predictive systems' PAR-2: "
             f"{results.get('feat_overhead_used')}")
    L.append("")
    L.append("A selector whose choice histogram is ~100% one solver has collapsed to a "
             "constant predictor; compare it with the majority baseline row.")
    L.append("")

    # 5. planned comparisons
    L.append("## 5. Planned comparisons (Holm-Bonferroni adjusted)")
    L.append("")
    L.append(f"Bootstrap resamples used: {results.get('bootstrap_resamples_used', '-')} "
             f"({results.get('bootstrap_cap_note', '')}). Stratified by family.")
    L.append("")
    rows = []
    for c in results.get("comparisons", []):
        ci = f"[{_fmt(c.get('ci_low'))}, {_fmt(c.get('ci_high'))}]"
        rows.append([c["comparison"], _fmt(c.get("mean_diff")), ci,
                     str(c.get("ci_excludes_zero")),
                     _fmt(c.get("bootstrap_p", c.get("bootstrap_p_approx")), 4),
                     f"{_fmt(c.get('wilcoxon_p'), 4)} (n={c.get('wilcoxon_n', '-')})",
                     _fmt(c.get("mcnemar_p"), 4), str(c.get("holm_reject"))])
    L += _table(["comparison", "mean diff", "95% CI", "CI excl. 0", "bootstrap p",
                 "Wilcoxon p", "McNemar p", "Holm reject"], rows)
    L.append("Mean diff is system_a PAR-2 minus system_b PAR-2; negative means system_a "
             "(the selector) is faster. The bootstrap p is two-sided: twice the fraction "
             "of stratified resample means on the opposite side of zero from the observed mean.")
    L.append("")

    # 6. per-family table
    L.append("## 6. Per-family results")
    L.append("")
    pfam = results.get("per_family") or {}
    if pfam:
        systems = []
        for fam_entry in pfam.values():
            for s in fam_entry:
                if s not in systems:
                    systems.append(s)
        header = ["family", "n"] + [f"{s} PAR-2 (solve)" for s in systems]
        rows = []
        for fam, entry in sorted(pfam.items()):
            n = next(iter(entry.values())).get("n", "-") if entry else "-"
            rows.append([fam, n] + [f"{_fmt(entry.get(s, {}).get('par2'))} "
                                    f"({_pct(entry.get(s, {}).get('solve_rate'))})"
                                    for s in systems])
        L += _table(header, rows)
    else:
        L.append("- (no per-family block)")
        L.append("")

    # 7. classification accuracy
    L.append("## 7. Classification accuracy")
    L.append("")
    cls = results.get("classification") or {}
    if cls:
        fams_all: list[str] = []
        for blk in cls.values():
            for f in (blk.get("per_family") or {}):
                if f not in fams_all:
                    fams_all.append(f)
        header = ["system", "accuracy (non-tie)", "cost-weighted acc.", "n scored"] + fams_all
        rows = []
        for s, blk in cls.items():
            rows.append([s, _pct(blk.get("accuracy")), _fmt(blk.get("cost_weighted_accuracy")),
                         blk.get("n_scored", "-")] +
                        [_pct((blk.get("per_family") or {}).get(f)) for f in fams_all])
        L += _table(header, rows)
        L.append(f"Majority-class accuracy for reference: {_pct(ls.get('majority_share'))}. "
                 "Cost-weighted accuracy is the mean of 1/w(i, chosen), w = clamped runtime "
                 "ratio to the best solver (1.0 = always optimal).")
    else:
        L.append("- (no classification block)")
    L.append("")

    # 8. feature ablation
    L.append("## 8. Feature-cost ablation (Pareto)")
    L.append("")
    abl = results.get("feature_ablation") or []
    if abl:
        rows = []
        for e in abl:
            rows.append([e.get("group", "-"), e.get("n_features", "-"),
                         _fmt(e.get("val_par2")), _pct(e.get("val_accuracy")),
                         _fmt(e.get("mean_feat_cost_s"), 4),
                         _hist_str(e.get("choice_hist")), e.get("note", "") or ""])
        L += _table(["feature group", "# features", "val PAR-2", "val accuracy",
                     "mean feature cost (s)", "choice histogram", "note"], rows)
        L.append("Fixed-parameter RandomForest (100 trees, depth 8, balanced classes) refit "
                 "per cumulative group on train, scored on val. Lower PAR-2 at lower "
                 "feature cost is Pareto-better.")
    else:
        L.append("- (no feature columns; ablation skipped)")
    L.append("")

    L.append("### 8b. Feature-free GNN comparison (objective O6)")
    L.append("")
    ff = results.get("gnn_feature_free") or {}
    if ff and "full" in ff and "none" in ff:
        rows = [
            ["designed node features", _fmt(ff.get("full"))],
            ["feature-free (pure topology)", _fmt(ff.get("none"))],
            ["difference", _fmt(ff.get("delta_par2"))],
        ]
        L += _table(["GNN variant", "best val PAR-2 (s)"], rows)
        delta = ff.get("delta_par2")
        if delta is not None:
            if delta > 0:
                L.append(f"Designed node features reduce validation PAR-2 by "
                         f"{_fmt(delta)} s, so expert knowledge is contributing "
                         f"beyond what message passing recovers from topology alone.")
            elif delta < 0:
                L.append(f"The feature-free variant is {_fmt(-delta)} s *better*, so the "
                         f"designed node features are not earning their place and the "
                         f"graph structure alone carries the signal.")
            else:
                L.append("No measurable difference between the two variants.")
        L.append("")
        L.append("Both variants are trained by `satsel train-gnn` on identical splits "
                 "with identical seeds and architecture, so the gap is attributable to "
                 "the node features alone. This is the feature-engineering-bottleneck "
                 "question raised by Zhang et al. (2026), measured on our own data "
                 "rather than cited.")
    else:
        L.append("- (no GNN trained, or the node-feature ablation was disabled; "
                 "run `satsel train-gnn` to populate this)")
    L.append("")

    # 9. OOD
    L.append("## 9. Out-of-distribution")
    L.append("")
    if ood and "vbs_ceiling" in ood:
        ceiling = ood.get("vbs_ceiling", {})
        L.append(f"- OOD family: {ood.get('primary_ood_family')}; n = {ood.get('n_ood')}")
        L.append(f"- OOD VBS solve rate: {_pct(ceiling.get('vbs_solve_rate'))} "
                 f"(ceiling met: {ceiling.get('ceiling_met')})")
        L.append(f"- Selector PAR-2 on OOD: {_fmt(ood.get('selector_par2_ood'))}; "
                 f"VBS PAR-2 on OOD: {_fmt(ood.get('vbs_par2_ood'))}")
        per_sys = ood.get("per_system") or {}
        if per_sys:
            L.append("")
            L += _table(["system", "OOD PAR-2", "solve rate", "choice histogram"],
                        [[s, _fmt(v.get("par2")), _pct(v.get("solve_rate")),
                          _hist_str(v.get("choice_hist"))] for s, v in per_sys.items()
                         if isinstance(v, dict) and "error" not in v])
        L.append(f"- LOFO: {ood.get('lofo')}")
    elif ood:
        L.append(f"- {ood}")
    else:
        L.append("- (not run)")
    L.append("")

    # 10. robustness
    L.append("## 10. Robustness appendix")
    L.append("")
    L.append(f"- {robustness or 'not run'}")
    L.append("")

    # 11. reproducibility
    L.append("## 11. Reproducibility")
    L.append("")
    seeds = cfg.get("seeds") or {}
    L.append(f"- Config hash (sha256, first 12): `{_config_hash(cfg)}`; schema version "
             f"{cfg.get('schema_version', '-')}, feature schema "
             f"{cfg.get('feature_schema_version', '-')}")
    L.append(f"- Seeds: split={seeds.get('split', '-')}, random_baseline="
             f"{seeds.get('random_baseline', '-')}, generators={seeds.get('generators', '-')}, "
             f"model_train={seeds.get('model_train', '-')}")
    L.append(f"- n_test = {results.get('n_test', ls.get('n_test', '-'))}; bootstrap resamples "
             f"= {results.get('bootstrap_resamples_used', '-')}; alpha = {cfg.get('alpha', '-')}; "
             f"runs per (instance, solver) = {cfg.get('runs_per_pair', '-')}")
    ml = results.get("models_loaded") or {}
    if ml:
        L.append("- Models loaded: " + ", ".join(f"{k}={'yes' if v else 'no'}"
                                                 for k, v in ml.items()))
    L.append("- Deviations from the pre-registered protocol: see `EVAL_PROTOCOL.md` "
             "section 8 (deviations log).")
    L.append("")

    out_path = reports / "report.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    (reports / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                          encoding="utf-8")
    if ood is not None:
        (reports / "ood.json").write_text(json.dumps(ood, indent=2, default=str),
                                          encoding="utf-8")
    return str(out_path)
