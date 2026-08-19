# Runbook: how to build and use satsel

A plain-language walkthrough. Read it top to bottom the first time; after that you only need the step that matches where you are.

## What this project does

`satsel` reads a logic formula (a CNF file, the standard SAT input format) and predicts which SAT solver from a fixed set of seven will solve it fastest. It then runs that solver. The prediction comes from a machine-learning model trained on how fast each solver runs on thousands of formulas.

The full spec lives in `DEVELOPMENT_PLAN.md`. The pre-registered evaluation rules live in `EVAL_PROTOCOL.md`.

## What you need before you start

- macOS or Linux.
- `uv` (the Python package manager). If you do not have it: `brew install uv`, or follow the instructions at https://docs.astral.sh/uv.
- About 2 GB of free disk space.
- Time. The slowest step (profiling) takes about 1.5 to 2.5 hours on 11 cores. Every other step takes minutes or seconds.

You do not need a SAT solver installed. The project installs seven of them (MiniSat22, Glucose4, CaDiCaL, Maplesat, Lingeling, MapleCM, Kissat404) as a single pip package called `python-sat`.

## 1. Install

```
cd minor-proj
uv sync
```

This creates a virtual environment and installs every dependency. Add the neural-network parts with:

```
uv sync --extra gnn
```

The GNN (graph neural network, the main model) needs the `gnn` extra. Skip it and the classical models still work; you just cannot train the GNN.

Check the install with:

```
uv run pytest -q
```

You should see a line like `34 passed, 3 deselected`.

## 2. Run the whole pipeline

One command builds everything, in order:

```
uv run satsel all
```

It runs these steps, in this order:

1. `bench` — download formulas if the network allows, otherwise generate them from fixed random seeds. Writes 2300 CNF files.
2. `features` — turn every formula into a 63-number feature vector.
3. `profile` — time every solver on every formula. This is the long step.
4. `build` — assemble the dataset and split it into train/validation/test.
5. `train` — train the classical models (Random Forest, XGBoost, SVM) plus the SATzilla baseline, and pick the fallback threshold.
6. `train-gnn` — train the graph neural network (skip with `--no-gnn`).
7. `evaluate` — run the pre-registered statistics and write the report.

When it finishes, open the report:

```
open artifacts/reports/report.md
```

## 3. Run each step by itself

Everything `satsel all` does can be run separately. Run them in this order because each step reads the output of the previous one.

### 3.1 bench: get the formulas

```
uv run satsel bench
```

Tries to download real benchmark instances from SATLIB and GBD. Those servers are often unreachable, and one old host is known to be hijacked, so the downloader is a bounded probe: if it gets nothing it logs a warning and the project's own generators fill every family to its target. The result is always 2300 formulas (200 of them held out as the out-of-distribution family).

### 3.2 features: turn formulas into numbers

```
uv run satsel features --workers 11
```

Each formula becomes a fixed vector of 63 numbers: clause lengths, variable occurrences, graph statistics, and a short solver probe. Extraction respects a time budget; if a group would exceed it, those features are marked missing.

### 3.3 profile: time every solver (the slow part)

```
uv run satsel profile --workers 11 --heartbeat 15
```

This is the step that takes hours. It times all seven solvers on all 2300 formulas in three phases: a 5-second presolve, a 30-second screen on every solver, then longer runs only for pairs that timed out at screen.

Things to know:

- **You will see live progress.** Every 15 seconds (the `--heartbeat` value) it prints how many formulas are done, the rate, and an ETA. It prints a line for every finished formula too. Easy families go first, so progress is visible almost immediately.
- **Ctrl-C is safe.** Every completed formula is written to the database the moment it finishes. Interrupt, re-run the same command, and it skips what is already done and continues. You can also just close the terminal and run it again later.
- **It is resumable across days.** Nothing is lost on exit.
- `--limit 5` profiles just a few formulas first, so you can watch it work before committing to the full run.
- A worst-case formula (one no solver can finish) takes about 17 minutes. Most formulas solve in under a second, which is why the whole corpus still finishes in a couple of hours.

### 3.4 build: assemble the dataset

```
uv run satsel build
```

Reads the profiling database, computes the label for every formula (which solver was fastest, what each solver's time was, the cost weight for each mistake), removes near-duplicate formulas, and splits the rest into train/validation/test (70/15/15) so that no duplicate crosses a split. Writes a summary to `artifacts/datasets/dataset_summary.json`.

### 3.5 train: the classical models

```
uv run satsel train
```

Trains Random Forest, XGBoost, and SVM classifiers with cost-weighted training (a costly mistake, like picking a slow solver for an easy formula, is penalised more). Picks the best on validation PAR-2, calibrates the fallback threshold, and trains the SATzilla-style baseline. Output goes to `artifacts/models/`.

### 3.6 train-gnn: the main model

```
uv run satsel train-gnn --epochs 40
```

Trains the graph neural network. It reads each formula as a bipartite graph of variables and clauses and trains with an expected-cost loss. Formulas larger than a size cap are routed to the classical path at prediction time.

### 3.7 evaluate: the pre-registered statistics

```
uv run satsel evaluate
```

Runs the exact protocol from `EVAL_PROTOCOL.md`: PAR-2 (mean solve time with timeouts counted double), solve rate, the stratified bootstrap confidence interval, Wilcoxon and McNemar tests, Holm-Bonferroni adjustment across the four planned comparisons, the out-of-distribution check, and the robustness appendix. Writes `artifacts/reports/report.md` and `results.json`.

## 4. Use the one public API

After training, the trained selector is wrapped in a single call. Save a CNF file and run:

```
uv run satsel oracle --input my_formula.cnf --timeout 300
```

Or call it from Python:

```python
from satsel.oracle import SolverOracle

oracle = SolverOracle()
with open("my_formula.cnf", "rb") as fh:
    result = oracle.solve(fh.read(), timeout_s=300)

print(result.status)          # SAT / UNSAT / TIMEOUT / CRASH / ERROR
print(result.solver_used)     # which solver ran
print(result.runtime_ms)      # total wall time in milliseconds
print(result.confidence)      # how sure the selector was
```

Guarantees you can rely on:

- **It never raises.** Any failure comes back as a typed result with `status = "ERROR"` and a short message. No stack trace leaks out.
- **Total time is capped.** All work runs in a child process that is killed when the budget is exceeded.
- **It takes content bytes, not file paths.** A path-based variant exists but rejects symlinks and anything outside the allowed folder.
- **Inputs are size-checked** before any solver memory is allocated.

## 5. Tests

```
uv run pytest              # everything except the slow solver tests
uv run pytest -m slow      # the real-solver tests (nightly)
```

The quick suite runs in seconds and does not need real solvers. The slow suite runs the actual solvers on small formulas.

## 6. Project layout

```
minor-proj/
├── DEVELOPMENT_PLAN.md   the spec
├── EVAL_PROTOCOL.md       the pre-registered evaluation rules
├── README.md              this project's front door
├── RUNBOOK.md             you are here
├── config/default.yaml    frozen settings (timeouts, seeds, paths)
├── src/satsel/
│   ├── core/              DIMACS parser, schemas, config, subprocess helper
│   ├── bench/             step 1: get formulas
│   ├── features/          step 2: feature extraction
│   ├── profiling/         step 3: solver timing
│   ├── data/              step 4: dataset build + splits
│   ├── models/            steps 5-6: classical + GNN
│   ├── oracle/            step 7: the public API
│   ├── eval/              step 8: metrics + stats + report
│   └── cli.py             the `satsel` command
├── tests/
└── artifacts/             all outputs; gitignored
```

One rule governs the code: a module may only import from `core` or earlier stages (`core <- bench <- features/profiling <- data <- models <- oracle/eval`).

## 7. Troubleshooting

**`manifest not found`** when running `features` or `profile`. Run `uv run satsel bench` first.

**`no f_* feature columns in dataset`** when running `train`. Run `uv run satsel features` and then `uv run satsel build` again.

**`torch` / `torch_geometric` errors** when running `train-gnn`. You installed without the GNN extra. Run `uv sync --extra gnn`.

**Profiling seems stuck at 0 done for minutes.** That is normal only for the hardest family. Easy families are processed first now, so if it is still at zero after a couple of minutes, check that you are on a recent version of the code and that the heartbeat is printing. If you see heartbeat lines but no completions, you are profiling genuinely hard formulas; Ctrl-C and re-run to resume later.

**The report says a comparison "not run".** The GNN comparison needs `train-gnn` to have run. The robustness appendix needs a solver hook that is only wired in nightly jobs.

## 8. Where the numbers land

Outputs are written under `artifacts/`, which is gitignored:

- `artifacts/raw/` — the CNF files
- `artifacts/manifests/` — the formula manifest (parquet + json)
- `artifacts/features/` — the feature vectors
- `artifacts/runs/` — the profiling database
- `artifacts/datasets/` — the dataset, splits, labels, summary
- `artifacts/models/` — the trained models and the theta curve
- `artifacts/reports/` — the evaluation report
