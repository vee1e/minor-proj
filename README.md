# satsel — ML-driven SAT solver selection

Reads a CNF file (the standard SAT input format: an AND of clauses, each clause an OR of literals), predicts which SAT solver from a fixed portfolio will solve it fastest, then runs that solver. Solvers are treated as black-box oracles, never modified. Machine learning runs only at the selection layer, never inside solver loops.

Full spec: `DEVELOPMENT_PLAN.md`. Pre-registered evaluation: `EVAL_PROTOCOL.md`. Step-by-step guide: `RUNBOOK.md`.

## Install

**On a new machine, run the bootstrap script.** It is the supported path and
handles the things that silently break a fresh checkout:

```
python scripts/bootstrap.py              # solvers + classical ML + dev tools
python scripts/bootstrap.py --gnn        # ... plus PyTorch (CPU build)
python scripts/bootstrap.py --gnn --cuda # ... plus PyTorch built for CUDA
python scripts/bootstrap.py --check      # verify an existing setup, install nothing
```

It finds or installs `uv`, fetches the pinned Python 3.11 (see
`.python-version`), syncs the lockfile, and then *verifies* that every
dependency imports. `--check` exits non-zero when something is missing, so it
works as a pre-flight gate before a long run.

Four failure modes it exists to prevent, all of which have bitten us:

- `uv` installs somewhere off PATH (on Windows,
  `%APPDATA%\Python\Python3xx\Scripts`) and appears "not installed".
- The system Python is newer than 3.11 and `uv sync` picks the wrong one.
- **On Windows, `pip install torch` from PyPI silently gives a CPU-only
  build.** Training then runs on the CPU with no warning. Use `--cuda`.
- A CUDA build too old for the GPU installs cleanly and only fails at runtime
  with "no kernel image is available". `--check` reports the GPU's compute
  capability against the kernels the installed torch actually carries.

If you prefer to drive `uv` directly:

```
uv sync --extra dev              # base: solvers + classical ML
uv sync --extra dev --extra gnn  # adds PyTorch + PyTorch Geometric (heavy)
```

This installs `python-sat == 1.9.dev15`, which ships MiniSat22, Glucose4,
CaDiCaL, Maplesat, Lingeling, MapleCM, and Kissat404 in one wheel. No system
installs.

### Choosing the compute device

The GNN honours `SATSEL_DEVICE` (`cpu`, `cuda`, `cuda:1`, or `auto`, the
default) and the `gnn_device` config key. Everything else in the pipeline —
including all solver profiling — is CPU-only by nature and ignores it.

## Quick start

Build everything end to end:

```
uv run satsel all            # full corpus (profiling takes hours)
uv run satsel all --small    # pilot corpus: ~80 per family + 40 OOD, minutes
uv run satsel all --tiny     # CI smoke test only; too small to learn from
```

`all` runs bench -> features -> profile -> build -> train -> train-gnn
(skip with `--no-gnn`) -> evaluate, and prints the data-quality warnings from
`dataset_summary.json` after `build` and again at the very end so they cannot
be missed.

Or run each stage separately, in this order:

```
uv run satsel bench        # 1. download + generate formulas
uv run satsel features     # 2. extract feature vectors
uv run satsel profile      # 3. time solvers (the long step)
uv run satsel build        # 4. assemble dataset + splits
uv run satsel train        # 5. classical models + theta calibration
uv run satsel train-gnn    # 6. GNN (optional extra)
uv run satsel evaluate     # 7. metrics + stats + report
```

The report lands at `artifacts/reports/report.md`. To check the health of
whatever has been produced so far without training anything:

```
uv run satsel diagnose     # label distribution, warnings, model diagnostics,
                           # per-system choice histogram
```

## Why the selector used to always pick MiniSat, and what changed

An early run of the pipeline produced a "selector" that chose MiniSat for every
formula. That was not a modelling insight; it was four compounding bugs in how
the training signal was produced. In plain terms:

1. **Timing measured clause loading, not solving (R1).** The stopwatch started
   before the Python loop that feeds clauses into the solver object. On the
   small generated formulas every solver finishes in milliseconds, so the
   "runtime" was dominated by per-solver constant overhead, and the solver with
   the lightest Python wrapper (MiniSat) "won" nearly every time. Timing now
   wraps only `solve()` (`perf_counter`) and also records PySAT's own
   `time()` / `time_accum()` when available.

2. **The tie rule threw away almost all training rows (R2).** Labels treated
   any formula where the fastest two solvers were within 50 ms or a 1.2x margin
   as a tie, and ties were excluded from training. On an easy corpus that was
   nearly everything, so the classifier saw a handful of rows and collapsed to
   the majority class. The solved-check also used the *minimum* run while the
   label used the *median*, which is inconsistent. Both are fixed, and `build`
   now reports tie rate, informative rate, usable training rows and the label
   distribution, with explicit warnings.

3. **The corpus was far too easy (R3).** The old `--small` corpus
   (30/10/20/... formulas) and the default generator sizes produced instances
   that every solver finished in under 0.1 s. There was no solver
   differentiation to learn. `--small` is now a ~80-per-family pilot corpus
   (40 OOD), and the old sizes live under `--tiny` for CI smoke tests only.

4. **The classical models hid the collapse (R4).** SVM was trained on
   unscaled features, there was no class balancing, no cross-validation, no
   calibration, and model choice was made on a tiny validation set. Nothing
   printed the predicted-class distribution, so "always MiniSat" was
   invisible. Training now uses scaling, balanced cost-weighted loss, nested
   CV with Platt calibration, and writes `artifacts/models/model_diagnostics.json`.

`satsel diagnose` prints the label distribution, the data-quality warnings,
and the per-system choice histogram so a degenerate selector is obvious at a
glance -- no dashboard required.

## Where the benchmark instances come from

`satsel bench` builds the corpus from **real SATLIB instances** downloaded from
the canonical UBC mirror, topped up by the project's seeded generators for the
families SATLIB does not cover.

```
uv run satsel sources           # what is available, and what is cached
uv run satsel sources --probe   # check every archive is reachable
uv run satsel sources --fetch   # download them all now (~4 MB, then works offline)
```

**2,210 real instances are available** — enough alone to meet the N >= 2,000
target in the synopsis:

| family | archives | instances |
| --- | --- | ---: |
| random3sat | uf/uuf at n = 150–250, plus CBS | 1,999 |
| colouring | flat200-479, flat125-301 | 200 |
| planning | logistics, blocksworld | 11 |

Generators then cover `random3sat_large`, `industrial`, `crypto`, `scheduling`
and the held-out `ood_scheduling` family.

A family's quota is split across its archives **by weight**, not evenly. The
`uf/uuf250` sets sit on the 4.26 phase transition where solvers disagree most
and are weighted highest; `uf150` and `CBS` solve in milliseconds on everything
and are weighted lowest. Satisfiable and unsatisfiable sets at each size carry
equal weight deliberately — UNSAT is where CDCL solvers separate most, since a
refutation must exhaust the search space rather than stumbling onto an answer.
Allocation uses largest-remainder apportionment so that balance stays exact.

Archives are cached in `artifacts/raw/_cache/`; copy that folder to an offline
machine and `bench` runs there with no network at all.

Provenance is recorded, not assumed: the manifest carries a `source` column and
`dataset_summary.json` reports `source_distribution`, `n_real_instances` and
`real_instance_share`. If nothing downloads, `provenance_warnings` says so
rather than letting a fully synthetic corpus be described as a benchmark
collection.

Two notes on why this needed fixing. Earlier builds pointed at
`www.satlib.org`, which has been a parked domain for years, so every download
failed silently. And real SATLIB files end with a `%` terminator line that our
DIMACS parser rejected — so even a successful download was discarded as
malformed. Both are fixed, with a regression test for the terminator.

## The GNN selector

`src/satsel/models/gnn.py` implements the GraSS tripartite architecture named
in the synopsis: `var` / `lit` / `clause` node types with four relations,
heterogeneous convolution (`HeteroConv` + `SAGEConv`), sinusoidal positional
encodings on clause nodes, and a runtime-sensitive expected-cost loss.

Giving literals their own nodes is the point, not decoration. The previous
implementation built a variable-clause bipartite graph and indexed edges by
`abs(lit)`, discarding the sign of every occurrence — so `(x OR y)` and
`(NOT x OR NOT y)` produced identical graphs. `tests/test_gnn.py` pins that
behaviour as a regression test.

```
uv run satsel train-gnn --epochs 40      # honours SATSEL_DEVICE
```

Training reports a per-layer ablation, the histogram of solvers it chose on
validation, and a warning if it collapsed onto a single solver.

## Profiling notes

`profile` is the long pole (2300 formulas x 7 solvers, three phases). It shows a live heartbeat with rate and ETA every N seconds:

```
uv run satsel profile --workers 11 --heartbeat 15
```

Every completed formula is checkpointed into `artifacts/runs/profiling.db`, so Ctrl-C or a crash is safe: re-running the same command skips what is already done and continues. Easy families are processed first so progress is visible immediately. `--limit N` profiles a few formulas first if you want to see it working, and `--no-resume` forces re-profiling.

A worst-case formula takes about 17 minutes (5s presolve + 7 x 30s screen + 2 x 60s escalation per timed-out solver), but most formulas solve in under a second, so the bulk of the corpus is fast.

## The one public API

After training, use the selector from Python:

```python
from satsel.oracle import SolverOracle

oracle = SolverOracle()
with open("formula.cnf", "rb") as fh:
    result = oracle.solve(fh.read(), timeout_s=300)

print(result.status)        # SAT / UNSAT / TIMEOUT / CRASH / ERROR
print(result.solver_used)   # which solver ran
```

`result` is a frozen object with `status`, `satisfiable`, `assignment`, `solver_used`, `solver_version`, `runtime_ms`, `parse_ms`, `feature_ms`, `infer_ms`, `solve_ms`, `features`, `confidence`, `theta_used`, `fallback_used`, `fallback_reason`, and `error`.

Guarantees: never raises (every failure is a typed `ERROR` result), total time is capped by running work in a killable child process, it takes content bytes rather than paths (a path variant rejects symlinks and out-of-folder paths), and inputs are size-checked before any memory is allocated.

There is also a command-line form: `uv run satsel oracle --input formula.cnf`.

## Layout

```
src/satsel/
  core/        DIMACS parser, schemas, config, canonical hashing, subprocess helpers
  bench/       step 1: download + generate formulas
  features/    step 2: feature extraction
  profiling/   step 3: solver timing
  data/        step 4: dataset build + splits
  models/      steps 5-6: classical + GNN
  oracle/      step 7: SolverOracle
  eval/        step 8: metrics + stats + report
  cli.py       command-line runner (also `python -m satsel.cli`)
tests/
config/default.yaml   # frozen settings
artifacts/            # outputs, gitignored
```

Import rule: `core <- bench <- features/profiling <- data <- models <- oracle/eval`. A module may only import from `core` or earlier stages.

## Tests

```
uv run pytest          # everything except the slow real-solver tests
uv run pytest -m slow  # real solver tests, nightly
```
