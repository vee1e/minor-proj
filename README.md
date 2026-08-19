# satsel — ML-driven SAT solver selection

Reads a CNF file (the standard SAT input format: an AND of clauses, each clause an OR of literals), predicts which SAT solver from a fixed portfolio will solve it fastest, then runs that solver. Solvers are treated as black-box oracles, never modified. Machine learning runs only at the selection layer, never inside solver loops.

Full spec: `DEVELOPMENT_PLAN.md`. Pre-registered evaluation: `EVAL_PROTOCOL.md`. Step-by-step guide: `RUNBOOK.md`.

## Install

Python 3.11 pinned. Uses `uv`:

```
uv sync                      # base: solvers + classical ML
uv sync --extra gnn          # adds PyTorch + PyTorch Geometric (optional, heavy)
```

This installs `python-sat == 1.9.dev15`, which ships MiniSat22, Glucose4, CaDiCaL, Maplesat, Lingeling, MapleCM, and Kissat404 in one wheel. No system installs.

## Quick start

Build everything end to end:

```
uv run satsel all
```

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

The report lands at `artifacts/reports/report.md`.

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
  cli.py       command-line runner
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
