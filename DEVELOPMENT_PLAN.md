# Spec & Development Plan

## ML-Driven SAT Solver Selection via PySAT Oracle Wrappers

Minor Project · School of Computer Engineering · MIT Manipal

Team: Aryendra Pandey (230907498) · Priyanshu Sharma (230907576) · Ansh Jaiswal (230959004) · Lakshit Verma (230959210) Guide: Dr. Manohara Pai Source: `CP_synopsis_revised.pdf` (Revised Draft v2.0, 15 Mar 2026) Repo: `/Users/verma/Projects/minor-proj`

---

## 1. What this project builds

A tool that reads a logic formula (a CNF file, one of the standard SAT input formats) and predicts which SAT solver from a fixed set will solve it fastest, then runs that solver.

CNF (conjunctive normal form): a formula written as an AND of clauses, each clause an OR of literals. SAT (satisfiability): is there a truth assignment that makes the formula true?

Solvers are treated as black-box oracles (outside programs called by name), never modified. Machine learning runs only at the selection layer, never inside solver loops. This is required because a solver's inner loop is milliseconds-fast; extra neural work in there would slow it down. `[from your synopsis §2.2]`

The success measure: lower PAR-2 (mean solve time, counting a timeout as double the limit) and more instances solved than three baselines — the single best solver, random choice, and a SATzilla-style predictor — measured on the full test set, including instances no solver could finish.

---

## 2. Goals (from the synopsis §5)

| # | Goal |
|---|---|
| O1 | List which formula features predict solver speed (syntactic, graph, search-probe). |
| O2 | Extract features within a fixed time budget; show the cost-vs-accuracy trade-off. |
| O3 | Build a dataset of ≥2,000 formulas, each profiled with runtimes. |
| O4 | Show which solvers are worth keeping; drop any that never win. |
| O5 | Train classical baselines (Random Forest, XGBoost, SVM) with cost-weighted training. |
| O6 | Train the main model, a graph neural network (GNN; a neural net that reads a graph), and compare to the baselines. |
| O7 | Package it as one `SolverOracle` call. |
| O8 | Evaluate per the pre-registered protocol (`EVAL_PROTOCOL.md`). |

---

## 3. How it works

Eight steps, run in order:

```
1 get formulas ─> 2 extract features ─> 3 profile solvers ─> 4 build dataset
                                                                │
8 evaluate <── 7 SolverOracle <── 6 train GNN <── 5 train classical models
```

1. **Get formulas.** Download from SATLIB and the Global Benchmark Database (GBD), plus generate some with fixed random seeds. At least 6 formula families, ≥2,000 total, ≥200 set aside for out-of-distribution (OOD) testing (a family never seen in training).
2. **Extract features.** Turn each formula into a number vector (≤120 numbers) within a hard time budget. If the budget runs out, drop the slow feature groups and mark the missing values.
3. **Profile solvers.** Time each solver on each formula. Run each (formula, solver) pair several times and use the median. A timed-out run is recorded as "censored at the limit", not as a raw time.
4. **Build dataset.** Combine features + runtimes into one table. Split into train/validation/test (70/15/15) so no near-duplicate formula crosses a split.
5. **Train classical models.** Random Forest, XGBoost, SVM. Each trains with per-sample cost weights, so a costly mistake (picking a slow solver for an easy formula) is penalised more.
6. **Train the GNN.** The main model, on a graph version of each formula, using an expected-cost loss.
7. **SolverOracle.** One public call: feed formula bytes in, get a typed result with full provenance, never an exception.
8. **Evaluate.** Per `EVAL_PROTOCOL.md`: PAR-2, solve rate, OOD, bootstrap confidence intervals, adjusted significance tests.

---

## 4. The one public API

```python
result = oracle.solve(cnf_bytes, timeout_s=300)
```

`result` is a frozen data object with: `status` (SAT/UNSAT/TIMEOUT/CRASH/ERROR), `satisfiable`, `assignment` (the satisfying values, optional), `solver_used`, `solver_version`, `runtime_ms`, timing breakdown (`parse_ms`, `feature_ms`, `infer_ms`, `solve_ms`), `features`, `confidence`, `theta_used`, `fallback_used`, `fallback_reason`, `error`.

Guarantees:

- Total time is capped: wall time ≤ `timeout_s + overhead`. Enforced by running the work in a child process that the caller can kill.
- Never raises. Every failure comes back as a typed result with `status=ERROR`, and no stack trace or file path leaks out.
- Takes content bytes, not a file path, so a caller can't read arbitrary files. A path variant checks the path is inside an allowed folder and rejects symlinks.
- A bounded pre-check enforces max file size, max variables, max clauses, and that the header numbers match the actual content — before allocating memory.
- One fresh solver instance per call (solvers are not thread-safe).
- Worst case: the result is no worse than the fallback solver on any input. Confidence is never claimed to guarantee correctness.

---

## 5. Cost-weighted training (the key idea)

A "mistake" is picking a solver slower than the best one. Weight that mistake by the runtime ratio:

```
w(i, s) = clamp( T(i,s) / T(i,s*), 1, 10 )    # T = median runtime; s* = fastest solver
```

`clamp` (limit to a range) to 10 stops a sub-millisecond best time from dominating the loss, and each batch is normalised (divided by its mean). `[from reviewer: methodology]`

The three classical models consume the weight differently: XGBoost multiplies its gradient, Random Forest re-weights its bootstrap samples, SVM rescales per-sample `C`. So the spec says per-model `sample_weight`, not "cost-weighted cross-entropy" for all.

No inverse-frequency (rarity-based) weighting for the PAR-2 objective — the cost weight already encodes importance.

The GNN uses an expected-cost loss, which is smooth and trainable:

```
L = average over batch of  sum over solvers of  p(s | formula) · w(i, s)
```

Instances that time out everywhere are excluded from the loss (their best runtime is undefined) but stay in the test-set PAR-2.

---

## 6. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | CPython 3.11 (pinned) | Solver library needs it |
| SAT solvers | one pip package, `python-sat == 1.9.dev15` | Ships MiniSat22, Glucose4, CaDiCaL, Maplesat, Lingeling, MapleCM, Kissat404 in one wheel; no system installs. Note the name is `glucose4`, not `glucose41`; `kissat404`, not `pysat-kissat` (that package doesn't exist). |
| Env | `uv` + `uv.lock` | Reproducible installs |
| Features | vendored extractor we write | The ready-made SATfeatPy is unmaintained, Linux-only, and has no license `[unverified: repo status as of review]` |
| Classical ML | `scikit-learn`, `xgboost` | Standard |
| GNN | `PyTorch` + `PyTorch Geometric` | Standard; kept as an optional install |
| Data | `numpy`, `pandas`, `pyarrow`, `sqlite3` | Wide tables, no dict columns |
| Stats | `scipy` | Bootstrap, Wilcoxon, McNemar |
| Test | `pytest` (`quick`/`slow` markers) | Fast fake-solver tests in CI, real solver tests nightly |
| Lint/type | `ruff`, `pyright` | |

---

## 7. Repo layout

```
minor-proj/
├── DEVELOPMENT_PLAN.md        # this file
├── EVAL_PROTOCOL.md            # pre-registered evaluation, frozen before profiling
├── README.md                   # how to install and run
├── config/default.yaml         # frozen settings
├── src/
│   ├── core/                   # shared contracts: DIMACS parser, schemas, config
│   ├── bench/                  # step 1: download + generate formulas
│   ├── features/               # step 2: feature extraction
│   ├── profiling/              # step 3: solver timing
│   ├── data/                   # step 4: dataset build + splits
│   ├── models/                 # steps 5–6: classical + GNN
│   ├── oracle/                 # step 7: SolverOracle
│   ├── eval/                   # step 8: metrics + stats + report
│   └── cli.py                  # command-line runner
├── tests/
├── scripts/
└── artifacts/                  # outputs; gitignored
```

One rule governs imports: `core ← bench ← features/profiling ← data ← models ← oracle/eval`. A module may only import from `core` or earlier stages. This lets four people work in parallel without tangled imports. The exact schemas and signatures are frozen in milestone M0, before coding starts.

---

## 8. Data

### 8.1 Formula sources

| Family | Source | Count |
|---|---|---|
| Random 3-SAT | SATLIB | 400 |
| Random 3-SAT (large) | own generator | 400 |
| Graph colouring | SATLIB `flat*` | 300 |
| Block-world planning | SATLIB (7) + GBD planning | 200 |
| Industrial hardware | GBD + Velev VLIW | 400 |
| Scheduling | GBD + own generator | 300 |
| **OOD — scheduling (primary)** | own generator, seeds unique to this family | 200 |
| Crypto (appendix) | SAT-2021 crypto, hardness-capped | ≤200 |

Why scheduling for OOD: the crypto instances are mostly timeouts for every solver, which would make the OOD number meaningless and blow the compute budget. `[unverified: based on reviewer's check of the crypto track]`

### 8.2 Label rule

- A run is a timeout if CPU time ≥ the limit.
- A timed-out run is recorded as "censored at the limit" (a marker), not as double the limit.
- `solved(i,s)` = median of the finished runs is under the limit. When solved, `T(i,s)` = that median.
- The double-limit `2·tau` appears only in the PAR-2 metric.
- Tie rule: if the second-best solver is within 1.2× of the best, or the best is under 50 ms, mark it a tie. Ties are excluded from accuracy, kept in PAR-2.

### 8.3 No data leakage

Leakage (a test formula that's secretly like a training one) would inflate the score. So, before any split: hash every formula canonically, join on GBD's isomorphism hash, cluster near-duplicates, and never let a cluster or an OOD formula cross a split. This is checked by CI.

### 8.4 Licence

SAT Competition instances are mostly not free to republish. The dataset ships links + hashes for those, and the team's own generated formulas under CC-BY-4.0. No solver binaries are committed.

---

## 9. Profiling

The naive plan (7 solvers × 3 runs × ~2,200 formulas × 300 s worst case) is too slow. Instead, three phases:

1. Presolve: 1 run of Kissat, 5 s, to tag hardness.
2. Screen: all solvers × 1 run, 30 s. Formulas solved in ≤30 s never see the long limit.
3. Escalate: only pairs that timed out at 30 s get 2 more runs at the family limit. Near-winners get 2 short runs; clear losers keep 1 run.

Each run runs in a child process, killed with the whole process group on timeout (a Python signal can't stop a blocked C solver). Results go into one SQLite database with checkpointing, so a crash resumes instead of re-running 300 s jobs.

Expected cost: ~45–50 core-days ≈ 3–4 wall-clock days on three 8-core machines. On one laptop it's 7–10 days — too long, so profiling starts early, right after a 100-formula pilot.

---

## 10. Evaluation (frozen in `EVAL_PROTOCOL.md`)

- **Primary test:** paired bootstrap (10,000 resamples, stratified by family) on per-formula PAR-2 differences. Claim a win only if the 95% confidence interval excludes zero.
- **Wilcoxon signed-rank:** only as a check on pairs where both systems solved within the limit (avoids censoring ties). Never against the VBS (virtual best solver; the always-perfect oracle) — it's a ceiling, compared via oracle gap only.
- **Solve rate:** McNemar's test.
- **Multiple comparisons:** Holm–Bonferroni across the 4 planned comparisons (first threshold 0.05/4 = 0.0125).
- **Seeds:** ≥5 per system; seed variance folded into the interval.
- **Model choice:** tune on validation PAR-2 only; never touch the test set during tuning or threshold (`theta`) selection. Report the full threshold curve, with "always predict" and "always fallback" endpoints.
- **OOD:** leave-one-family-out mean across all 6 families, plus the primary scheduling family. Report the VBS solve rate on the OOD family first — if it's under ~30%, the comparison is meaningless.
- **Robustness appendix:** perturb the test formulas (add unit clauses, rename variables, add redundant clauses) and report PAR-2 change vs the fallback.

Baselines: single best solver, seeded random, SATzilla-style regression, majority class, threshold endpoints, representation-vs-architecture controls, VBS (ceiling).

---

## 11. Timeline

| Phase | Work | Time |
|---|---|---|
| M0 | Env, solver check, freeze contracts + config | 2–3 days |
| M1 | Get + verify formulas | 1.5 weeks |
| M2a | Pilot profile 100 formulas, freeze budget, power check | 2–3 days |
| M2b | Feature pipeline + extraction | 1 week |
| M3 | **Full profiling** (critical path) | 3–4 days on 3 machines |
| M4 | Classical models | 1.5 weeks |
| M5 | GNN | 2 weeks |
| M6 | SolverOracle | 1 week |
| M7 | Evaluation + report | 1 week |
| M8 | Polish + release | buffer |

Critical path is M2a → M3 → M7. Start profiling before features finish.

---

## 12. Main risks

| Risk | Fix |
|---|---|
| Profiling too slow | Screening, 3 machines, checkpointing, start early |
| OOD meaningless | Scheduling family (controllable hardness) + LOFO; report VBS ceiling first |
| Data leakage | Canonical-hash dedup, cluster splits, CI check |
| Cost weight exploding | Clamp to 10, normalise per batch |
| Test-set contamination | Tune on validation only, frozen protocol |
| Licensing | Pointer-only for SAT-Comp, CC-BY-4.0 for own, no binaries |

---

## 13. Claims we will not make

The synopsis overclaimed. Reworded to what the project can actually back:

- "first in the PySAT literature" → "to the best of our knowledge"
- "expected statistically significant reduction" → "we will test whether it reduces PAR-2"
- "first apples-to-apples comparison" → "a shared dataset with a fixed protocol"
- "compared directly against the feature-free upper bound" → "a self-implemented feature-free GNN proxy; external paper discussed, not reproduced"

Every PAR-2 claim gets a bootstrap interval and an adjusted test.

---

## 14. Sources

Synopsis: `CP_synopsis_revised.pdf` (all project content). Reviewer findings marked `[from reviewer: …]`. Solver-package facts `[unverified]` pending the M0 machine check.
