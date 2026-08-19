"""Profiling engine: three-phase timing with SQLite checkpointing.

Phases (EVAL_PROTOCOL.md):
  P0 presolve   kissat, tau_pres (5s)   -> hardness tag
  P1 screen     all solvers x 1 run, tau_screen (30s)
  P2 escalate   pairs that timed out at screen: 2 more runs at the family tau
                pairs that solved at screen: 2 short runs (small cap)

Every run is a subprocess killed by whole-process-group SIGKILL at the wall
clock; solver-internal budgets are never relied on. Results are written to one
SQLite database so a crash resumes instead of re-running long jobs.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from satsel.core.proc import ProcResult, run_subprocess
from satsel.core.schemas import RUN_CRASH, RUN_ERROR, RUN_SOLVED, RUN_TIMEOUT
from satsel.core.solvers import ALLOWED_SOLVERS

PRESOLVE_SOLVER = "kissat404"


def _worker_cmd(worker: Path) -> list[str]:
    return [sys.executable, "-m", "satsel.profiling.worker"]


def run_one(cnf_path: str, solver: str, timeout_s: float, run_idx: int,
            formula_id: str, seed: int = 0) -> dict[str, Any]:
    """Run one (formula, solver) in a subprocess; return a run record dict."""
    argv = _worker_cmd(Path("unused")) + [
        "--cnf", cnf_path, "--solver", solver, "--timeout", str(timeout_s),
        "--seed", str(seed), "--formula-id", formula_id,
    ]
    res: ProcResult = run_subprocess(argv, timeout_s=timeout_s + 5.0)
    status = RUN_ERROR
    runtime_s = float("nan")
    message = ""
    satisfiable = None
    try:
        payload = json.loads(res.stdout.decode().strip())
        if payload.get("ok"):
            status = payload["status"]
            runtime_s = float(payload["runtime_s"])
            message = payload.get("message", "")
            satisfiable = payload.get("satisfiable")
    except Exception as exc:  # noqa: BLE001
        message = f"bad worker output: {exc}"
    if res.timed_out or status == RUN_ERROR and res.returncode == -9:
        status = RUN_TIMEOUT
        runtime_s = timeout_s
    if status == RUN_TIMEOUT:
        runtime_s = timeout_s
    if status in (RUN_CRASH, RUN_ERROR):
        runtime_s = float("nan")
    return {
        "formula_id": formula_id, "solver": solver, "run_idx": run_idx,
        "status": status, "runtime_s": runtime_s,
        "tau_s": timeout_s, "wall_s": res.wall_s, "message": message,
        "satisfiable": satisfiable,
    }


def profile_formula(cnf_path: str, formula_id: str, family: str, *,
                    tau_screen: float, tau_pres: float, tau_family: float,
                    solvers: tuple[str, ...], seed_base: int = 0) -> list[dict[str, Any]]:
    """Run the three phases for one formula. Returns run records (already
    median-free; label derivation happens in the data stage)."""
    records: list[dict[str, Any]] = []
    try:
        payload = run_one(cnf_path, PRESOLVE_SOLVER, tau_pres, -1, formula_id, seed=seed_base)
    except Exception:  # noqa: BLE001
        payload = {"status": RUN_ERROR, "runtime_s": float("nan")}
    presolved = payload.get("status") in (RUN_SOLVED,)
    presolve_time = payload.get("runtime_s", float("nan"))

    screen: dict[str, dict[str, Any]] = {}
    for s in solvers:
        rec = run_one(cnf_path, s, tau_screen, 0, formula_id, seed=seed_base + 1)
        records.append(rec)
        screen[s] = rec

    min_screen = min((r.get("runtime_s") or float("inf")) for r in screen.values()
                     if r.get("status") == RUN_SOLVED) if any(
                         r.get("status") == RUN_SOLVED for r in screen.values()) else float("inf")
    if min_screen == float("inf"):
        min_screen = tau_screen

    for s in solvers:
        rec = screen[s]
        if rec.get("status") == RUN_SOLVED:
            short_cap = min(tau_screen, max(2.0, 3.0 * (rec.get("runtime_s") or 0.0)))
            for r in (1, 2):
                records.append(run_one(cnf_path, s, short_cap, r, formula_id,
                                          seed=seed_base + 1 + r))
        elif rec.get("status") == RUN_TIMEOUT:
            for r in (1, 2):
                records.append(run_one(cnf_path, s, tau_family, r, formula_id,
                                          seed=seed_base + 1 + r))
        else:
            # crash/error at screen: one retry at family tau
            records.append(run_one(cnf_path, s, tau_family, 1, formula_id, seed=seed_base + 3))

    for r in records:
        r["family"] = family
        r["presolved"] = presolved
        r["presolve_time_s"] = presolve_time
    return records


def _profile_one_worker(item, tau_screen: float, tau_pres: float,
                        tau_tiers: dict, solvers: tuple, max_timeout: float):
    fid, path, fam = item
    tau_family = float(tau_tiers.get(fam, max_timeout))
    recs = profile_formula(path, fid, fam, tau_screen=tau_screen,
                           tau_pres=tau_pres, tau_family=tau_family, solvers=solvers)
    return fid, recs


def run_batch(formulas: list[tuple[str, str, str]], cfg: dict[str, Any],
              max_workers: int | None = None, *, resume: bool = True,
              progress: callable | None = None,
              heartbeat_every_s: float = 30.0,
              limit: int | None = None) -> int:
    """Profile a list of (formula_id, path, family) tuples into the runs DB.

    Returns the number of formulas newly profiled. Completed formulas are
    skipped when `resume` is true.

    A time-based heartbeat thread prints live progress (done/total, rate,
    ETA, per-formula completions) to stderr every `heartbeat_every_s`
    seconds, whether or not a formula has finished, so progress is visible
    even while workers are mid-formula and even when stdout is piped.
    Ctrl-C cancels pending work, shuts the pool down promptly, and returns
    with whatever completed (already checkpointed to the DB).
    """
    import sqlite3
    import sys
    import threading
    import time

    db_path = Path(cfg["paths"]["db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    solvers = tuple(cfg.get("portfolio", ALLOWED_SOLVERS))
    tau_screen = float(cfg["tau_screen"])
    tau_pres = float(cfg["tau_pres"])
    tau_tiers = cfg["tau_tiers"]
    max_timeout = float(cfg.get("max_timeout", 300))

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    _init_db(conn)
    done = {r[0] for r in conn.execute("SELECT DISTINCT formula_id FROM runs")}
    conn.close()

    todo = [(fid, path, fam) for fid, path, fam in formulas if not resume or fid not in done]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print("[profiler] nothing to do (all formulas already profiled)", file=sys.stderr)
        return 0

    workers = max_workers or max(1, _default_workers())
    total = len(todo)
    start = time.monotonic()
    state = {"n_done": 0, "last": ""}
    print(f"[profiler] profiling {total} formulas with {workers} workers "
          f"(tau_screen={tau_screen}s, resume={'on' if resume else 'off'}, "
          f"heartbeat every {heartbeat_every_s}s)",
          file=sys.stderr, flush=True)

    def _heartbeat():
        while True:
            time.sleep(heartbeat_every_s)
            elapsed = time.monotonic() - start
            n = state["n_done"]
            rate = n / elapsed if elapsed > 0 else 0.0
            eta = (total - n) / rate / 60 if rate > 0 else float("inf")
            print(f"[profiler] {n}/{total} done ({100*n/total:.1f}%), "
                  f"rate {rate*60:.1f}/min, elapsed {elapsed/60:.1f}m, "
                  f"ETA {eta:.1f}m, last: {state['last'] or 'none yet'}",
                  file=sys.stderr, flush=True)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    interrupted = False
    ex: ProcessPoolExecutor | None = None
    try:
        ex = ProcessPoolExecutor(max_workers=workers)
        futs = {ex.submit(_profile_one_worker, item, tau_screen, tau_pres,
                          tau_tiers, solvers, max_timeout): item for item in todo}
        try:
            for fut in as_completed(futs):
                fid, recs = fut.result()
                _write_records(db_path, fid, recs)
                state["n_done"] += 1
                state["last"] = fid
                if progress:
                    progress(state["n_done"], total, fid)
                print(f"[profiler] + {fid}  ({state['n_done']}/{total})",
                      file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            interrupted = True
            for fut in futs:
                fut.cancel()
    finally:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)

    elapsed = time.monotonic() - start
    if interrupted:
        print(f"[profiler] interrupted after {state['n_done']}/{total} formulas "
              f"({elapsed/60:.1f}m); progress is checkpointed, re-run to resume",
              file=sys.stderr, flush=True)
    else:
        print(f"[profiler] finished {state['n_done']}/{total} in {elapsed/60:.1f}m "
              f"({100*state['n_done']/total:.1f}%)", file=sys.stderr, flush=True)
    return state["n_done"]


def _default_workers() -> int:
    import os

    return os.cpu_count() or 1


def _init_db(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            formula_id TEXT, solver TEXT, family TEXT, run_idx INT,
            status TEXT, runtime_s REAL, tau_s REAL, wall_s REAL,
            message TEXT, presolved INT, presolve_time_s REAL,
            PRIMARY KEY (formula_id, solver, run_idx)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_fid ON runs(formula_id)")
    conn.commit()


def _write_records(db_path: Path, formula_id: str, records: list[dict[str, Any]]) -> None:
    import sqlite3

    for attempt in range(5):
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                for r in records:
                    conn.execute(
                        "INSERT OR REPLACE INTO runs (formula_id, solver, family, run_idx,"
                        " status, runtime_s, tau_s, wall_s, message, presolved, presolve_time_s)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (r["formula_id"], r["solver"], r.get("family", ""), r["run_idx"],
                         r["status"], r.get("runtime_s"), r.get("tau_s"), r.get("wall_s"),
                         r.get("message", ""), int(r.get("presolved", False)),
                         r.get("presolve_time_s")),
                    )
            conn.close()
            return
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
