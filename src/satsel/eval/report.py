"""Report writer: renders the evaluation results to a markdown report +
results JSON in artifacts/reports/."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{float(x):.{nd}f}"


def write_report(cfg: dict, results: dict, ood: dict | None = None,
                 robustness: dict | None = None) -> str:
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append("ML-driven SAT solver selection via PySAT oracle wrappers.")
    lines.append("Pre-registered protocol: `EVAL_PROTOCOL.md`.")
    lines.append("")

    # 1. dataset summary
    summ_path = Path(cfg["paths"]["datasets"]) / "dataset_summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
    else:
        summ = {}
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append(f"- ID formulas: {summ.get('n_id', '-')}, "
                 f"OOD formulas: {summ.get('n_ood', '-')}")
    lines.append("- Train / val / test: "
                 f"{summ.get('n_train', '-')} / {summ.get('n_val', '-')} / "
                 f"{summ.get('n_test', '-')}")
    lines.append(f"- Cluster-duplicates dropped: {summ.get('n_deduped', '-')}")
    lines.append("")
    fams = summ.get("families", {})
    lines.append("| family | n | train | val | test | VBS solve rate |")
    lines.append("|---|---|---|---|---|---|")
    for fam in sorted(fams):
        v = fams[fam]
        lines.append(f"| {fam} | {v.get('n','-')} | {v.get('n_train','-')} | "
                     f"{v.get('n_val','-')} | {v.get('n_test','-')} | "
                     f"{_fmt(v.get('vbs_solve_rate',0))} |")
    lines.append("")

    # 2. systems table
    lines.append("## 2. Systems on the test set")
    lines.append("")
    lines.append("| system | PAR-2 | solve rate |")
    lines.append("|---|---|---|")
    for row in results.get("systems", []):
        lines.append(f"| {row['system']} | {_fmt(row['par2'])} | {_fmt(row['solve_rate'])} |")
    lines.append("")
    rms = results.get('random_mean_sd', [0, 0])
    lines.append(f"- Random baseline (seeded): mean {_fmt(rms[0])} SD {_fmt(rms[1])}")
    lines.append(f"- Oracle gap (selector): {_fmt(results.get('oracle_gap_selector',0))}")
    lines.append(f"- Chosen selector: {results.get('chosen_selector')}")
    lines.append("")

    # 3. planned comparisons
    lines.append("## 3. Planned comparisons (Holm-Bonferroni adjusted)")
    lines.append("")
    lines.append("| comparison | mean diff | 95% CI | excludes 0 | "
                 "Wilcoxon p | McNemar p | reject |")
    lines.append("|---|---|---|---|---|---|---|")
    for comp in results.get("comparisons", []):
        ci = f"[{_fmt(comp.get('ci_low'))}, {_fmt(comp.get('ci_high'))}]"
        lines.append(f"| {comp['comparison']} | {_fmt(comp.get('mean_diff'))} | {ci} | "
                     f"{comp.get('ci_excludes_zero')} | "
                     f"{_fmt(comp.get('wilcoxon_p'))} | {_fmt(comp.get('mcnemar_p'))} | "
                     f"{comp.get('holm_reject')} |")
    lines.append("")
    lines.append("Mean diff is system_a PAR-2 minus system_b PAR-2; a negative "
                 "diff means the selector is faster.")
    lines.append("")

    # 4. OOD
    lines.append("## 4. Out-of-distribution")
    lines.append("")
    if ood:
        ceiling = ood.get("vbs_ceiling", {})
        lines.append(f"- OOD family VBS solve rate: {_fmt(ceiling.get('vbs_solve_rate',0))} "
                     f"(ceiling met: {ceiling.get('ceiling_met')})")
        lines.append(f"- Selector PAR-2 on OOD: {_fmt(ood.get('selector_par2_ood'))}")
        lines.append(f"- VBS PAR-2 on OOD: {_fmt(ood.get('vbs_par2_ood'))}")
        lofo = ood.get("lofo", {})
        lines.append(f"- LOFO: {lofo}")
    else:
        lines.append("- (not run)")
    lines.append("")

    # 5. robustness
    lines.append("## 5. Robustness appendix")
    lines.append("")
    lines.append(f"- {robustness or 'not run'}")
    lines.append("")

    # 6. deviations
    lines.append("## 6. Deviations from the pre-registered protocol")
    lines.append("")
    lines.append("See `EVAL_PROTOCOL.md` section 8 (deviations log).")
    lines.append("")

    out_path = reports / "report.md"
    out_path.write_text("\n".join(lines))
    (reports / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return str(out_path)
