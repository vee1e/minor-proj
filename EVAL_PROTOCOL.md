# PRE-REGISTERED EVALUATION PROTOCOL

Project: ML-Driven SAT Solver Selection via PySAT Oracle Wrappers Repo root: `/Users/verma/Projects/minor-proj`

**Status: DRAFT — to be frozen with a timestamp in the git history BEFORE profiling (M3) begins.** Any change after freezing is a documented deviation appended to §8 below.

---

## 1. Research question & primary endpoint

**Question.** Does an ML-driven selector (classical or GNN) achieve lower PAR-2 and higher solve rate than the single best-average solver, uniform random selection, and a SATzilla-style regression baseline, measured on the complete held-out test set (all-timeout instances included) with feature/orchestration overhead included?

**Primary endpoint (confirmatory).** Relative PAR-2 reduction of the **GNN selector** vs the **single best-average solver** on the in-distribution test set. Claim made only if the family-stratified paired-bootstrap 95% CI (10,000 resamples) on per-instance PAR-2 differences excludes 0 **and** the Holm–Bonferroni-adjusted test (below) is significant.

**Secondary endpoints.** PAR-2 vs random and SATzilla baselines; solve rate (McNemar); oracle gap; OOD PAR-2 (LOFO + primary OOD family); feature overhead.

## 2. Null hypotheses & planned comparisons

The 4 planned confirmatory comparisons (each Holm–Bonferroni adjusted, family-wise α = 0.05):

1. GNN vs single best-average solver (primary)
2. GNN vs uniform random (seeded)
3. GNN vs SATzilla-style regression (reimplemented on identical data)
4. Best classical (RF/XGB/SVM, selected on validation) vs single best-average solver

**Never significance-tested vs VBS** — VBS is a ceiling, compared via oracle gap only (its paired differences are one-sided by construction, so signed-rank is meaningless).

## 3. Pre-specified configuration (frozen in `config/default.yaml`)

| Key | Value |
|---|---|
| `tau_s` | per-family tiers: random 60, colouring 120, planning 120, scheduling 180, industrial 300, crypto/OOD 300 |
| `tau_screen` | 30 s |
| `tau_pres` | 5 s |
| `tau_feat` | `max(0.05·mean_best_solver_median, 0.20 s)`, frozen after pilot |
| `max_timeout` (oracle) | 300 s |
| `runs_per_pair` | 3 (median-of-solved; see label rule) |
| OOD family | **scheduling (generated)** — primary; LOFO across all 6 families as headline |
| `theta` grid | swept on validation; reported as full `PAR-2(theta)` curve |
| seeds | 5 per system; random baseline 10 draws, seed-paired |
| feature set version | `feature_schema_version` recorded in artifacts |
| portfolio | candidates: minisat22, glucose4, cadical153, maplesat, lingeling, maplecm, kissat404 (pruned → `artifacts/portfolio.json`) |
| GNN cap | `n + m ≤ 5e5` (oversized → classical path) |

## 4. Metric definitions

- **PAR-2(i)** = `T(i, s_hat) + t_feat(i)`; if unsolved within `tau` → `2·tau`. PAR-2 = mean over the **complete test set**.
- **Solve rate** = fraction of test instances solved within `tau`.
- **Oracle gap** = `Σ T(i,s_hat) / Σ T(i,s*)` over VBS-solvable instances, per-instance ratios capped at 10.
- **Classification acc** = fraction predicted == optimal, interpreted against `1 − flip_fraction` ceiling; plus cost-weighted accuracy.
- **Label rule** (decoupled from metric): timed-out run = censored `tau`; `solved(i,s) = (median of ok runs) < tau`; when solved `T(i,s)` = median of solved runs; `2·tau` only in the PAR-2 metric.
- **Ties**: `T_2nd ≤ 1.2·T_1st` or `T_1st < 50 ms` → `tie_flag=1`; excluded from accuracy, kept in PAR-2/oracle-gap.

## 5. Statistical tests

- **Primary:** paired bootstrap on per-instance PAR-2 differences, 10,000 resamples, stratified by family; claim iff 95% CI excludes 0; report effect size (mean diff + VBS-normalized %).
- **Wilcoxon signed-rank:** sensitivity analysis ONLY on pairs where both systems solved within `tau` (no censoring ties).
- **Solve rate:** McNemar's test on the solved/unsolved contingency table.
- **Multiple comparisons:** Holm–Bonferroni over the 4 planned comparisons (first threshold `0.05/4 = 0.0125`).
- **Seeds:** ≥5 independent seeds; report mean±SD PAR-2; seed variance folded into the CI.
- **Model selection:** inner criterion = validation PAR-2; nested CV is diagnostic only; final numbers from the 85/15 hold-out; test set never touched during tuning, `theta` selection, or calibration.

## 6. Baselines

(i) single best-average solver; (ii) uniform random (seed-paired); (iii) SATzilla-style regression on identical data; (iv) majority-class/zero-rule; (v) `theta=0` and `theta=1` endpoints of the coverage curve; (vi) representation-vs-architecture controls (GNN on node features only; RF on graph features); (vii) VBS (ceiling, no significance test); "no-features global-best" null.

## 7. OOD protocol

- **Primary OOD family:** generated scheduling (hardness-controllable, license-clean, structurally distinct).
- **Headline OOD metric:** leave-one-family-out (LOFO) mean PAR-2 across all 6 families + CI.
- **Crypto:** hardness-capped hard-suite appendix, reported separately; demoted from headline (mostly all-solver timeouts → degenerate).
- Before any OOD claim, report the VBS solve rate on the OOD family (if VBS solves <30%, comparisons are vacuous — state this ceiling first).
- Leakage: canonical-hash + GBD isohash dedup; cluster-based splits; OOD never shares a cluster/hash with ID (asserted in CI); nearest-ID distance per OOD instance recorded.

## 8. Robustness appendix (red-team)

Perturb the test set: append `k` unit clauses; random variable renaming; add `t` redundant clauses; duplicate-then-trim. Report PAR-2 degradation vs the pre-solver fallback. No selector robustness is claimed; the fallback is the only guarantee.

## 9. Deviations log

| Date | Deviation | Reason | Author |
|---|---|---|---|
| — | (none yet) | | |
