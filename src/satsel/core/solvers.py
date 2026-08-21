"""Single-process solver execution used by profiling workers and the oracle.

Solver names are validated against a fixed allowlist so a caller can never pass
an arbitrary module path into pysat. A fresh solver instance per call: solvers
are not thread-safe.

Timing contract (important for label quality):
- ``runtime_s``   wall time of ONLY the ``solve()`` call (perf_counter). The
                  Python-side ``add_clause`` loop is NOT included, otherwise
                  easy formulas are labelled by per-solver construction
                  overhead rather than by search performance.
- ``solve_cpu_s`` solver-reported CPU time for the last call (pysat
                  ``Solver.time()``), NaN when the backend does not expose it.
- ``load_s``      time spent constructing the solver and adding clauses.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from pysat.solvers import Solver

from satsel import __version__
from satsel.core.schemas import (
    RUN_CRASH,
    RUN_ERROR,
    RUN_SOLVED,
    RUN_TIMEOUT,
)

# Frozen allowlist. Adding a solver here is a config change (documented).
ALLOWED_SOLVERS = (
    "minisat22",
    "glucose4",
    "cadical153",
    "maplesat",
    "lingeling",
    "maplecm",
    "kissat404",
)

SOLVER_VERSION = f"pysat-{__version__}"


class SolverError(Exception):
    pass


@dataclass
class SolveOutcome:
    status: str  # RUN_SOLVED / RUN_TIMEOUT / RUN_CRASH / RUN_ERROR
    satisfiable: bool | None
    runtime_s: float            # wall time of solve() only
    message: str = ""
    solve_cpu_s: float = float("nan")   # solver-reported CPU time (pysat .time())
    load_s: float = 0.0                 # construct + add_clause time


def validate_solver(name: str) -> None:
    if name not in ALLOWED_SOLVERS:
        raise SolverError(f"unknown or disallowed solver {name!r}")


def _solver_cpu_time(slv) -> float:
    """pysat exposes ``time()`` (last call) and ``time_accum()`` on most
    backends; return NaN when neither works."""
    for attr in ("time", "time_accum"):
        fn = getattr(slv, attr, None)
        if fn is None:
            continue
        try:
            v = float(fn())
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(v) and v >= 0.0:
            return v
    return float("nan")


def solve_in_process(
    clauses: tuple[tuple[int, ...], ...],
    nvars: int,
    solver: str,
    timeout_s: float,
    *,
    seed: int = 0,
) -> SolveOutcome:
    """Solve `clauses` with a fresh solver instance. Uses a pysat time budget
    plus a wall-clock guard; reports censored timeouts rather than exceptions.

    Only the ``solve()`` call is timed as ``runtime_s``; clause loading is
    reported separately as ``load_s``."""
    validate_solver(solver)

    load_start = time.perf_counter()
    try:
        # use_timer=True makes pysat populate time()/time_accum() (solver CPU
        # time of the last call); most backends reject a `seed` kwarg.
        try:
            slv = Solver(name=solver, use_timer=True, seed=seed)
        except TypeError:
            slv = Solver(name=solver, use_timer=True)
    except Exception as exc:  # noqa: BLE001
        return SolveOutcome(status=RUN_ERROR, satisfiable=None, runtime_s=0.0,
                            message=f"solver construct failed: {exc}",
                            load_s=time.perf_counter() - load_start)

    solve_start = load_start
    load_s = 0.0
    try:
        for clause in clauses:
            slv.add_clause(list(clause))
        try:
            slv.time_budget = timeout_s
        except Exception:
            pass  # time_budget may not be exposed on every solver build
        load_s = time.perf_counter() - load_start

        solve_start = time.perf_counter()
        res = slv.solve()
        elapsed = time.perf_counter() - solve_start
        cpu = _solver_cpu_time(slv)

        if res is None:
            return SolveOutcome(status=RUN_TIMEOUT, satisfiable=None, runtime_s=elapsed,
                                solve_cpu_s=cpu, load_s=load_s)
        if res is True:
            return SolveOutcome(status=RUN_SOLVED, satisfiable=True, runtime_s=elapsed,
                                solve_cpu_s=cpu, load_s=load_s)
        return SolveOutcome(status=RUN_SOLVED, satisfiable=False, runtime_s=elapsed,
                            solve_cpu_s=cpu, load_s=load_s)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - solve_start
        if elapsed >= timeout_s:
            return SolveOutcome(status=RUN_TIMEOUT, satisfiable=None, runtime_s=elapsed,
                                load_s=load_s)
        return SolveOutcome(
            status=RUN_CRASH,
            satisfiable=None,
            runtime_s=elapsed,
            message=f"solver raised {type(exc).__name__}: {exc}",
            load_s=load_s,
        )
    finally:
        try:
            slv.delete()
        except Exception:  # noqa: BLE001
            pass


# Solvers whose pysat bindings implement interrupt() / solve_limited(). Verified
# empirically against python-sat 1.9.dev15: CaDiCaL and Lingeling raise
# NotImplementedError("Limited solve is currently unsupported"), and a CaDiCaL
# probe given a 0.15 s budget ran for 5.8 s because nothing could stop it. Only
# solvers on this list may be used for an in-process, time-budgeted probe.
INTERRUPTIBLE_SOLVERS = ("minisat22", "glucose4", "maplesat", "maplecm")


@dataclass(frozen=True)
class ProbeOutcome:
    """Result of a short, budgeted, in-process probe run."""

    ok: bool
    solved: bool
    satisfiable: bool | None
    runtime_s: float
    interrupted: bool
    stats: dict[str, int]
    message: str = ""


def supports_interrupt(name: str) -> bool:
    return name in INTERRUPTIBLE_SOLVERS


def solve_probe(
    clauses: tuple[tuple[int, ...], ...],
    nvars: int,
    solver: str,
    budget_s: float,
    *,
    seed: int = 0,
) -> ProbeOutcome:
    """Run a short probe in this process, stopped by the solver's own interrupt.

    Used only for feature extraction, never for profiling. Profiling keeps its
    subprocess-per-run design because a wall-clock SIGKILL is the only thing
    that reliably stops a solver that has stopped cooperating; a probe is
    different, because the budget is small and the solver is asked to stop
    rather than made to.

    Doing this in-process removes ~0.45 s per instance of Python interpreter
    startup plus a temp-file round trip, which on easy formulas exceeded the
    solving time it was meant to measure and broke the tau_feat budget the
    evaluation protocol promises.

    Returns ok=False (rather than raising) if the solver cannot be interrupted,
    so the caller can mark the probe group missing and carry on.
    """
    if not supports_interrupt(solver):
        return ProbeOutcome(
            ok=False, solved=False, satisfiable=None, runtime_s=0.0,
            interrupted=False, stats={},
            message=f"{solver} does not support interrupt; refusing to probe")

    from threading import Timer

    slv = None
    timer = None
    try:
        try:
            slv = Solver(name=solver, bootstrap_with=[list(c) for c in clauses],
                         use_timer=True)
        except TypeError:
            slv = Solver(name=solver, bootstrap_with=[list(c) for c in clauses])

        fired = {"hit": False}

        def _stop() -> None:
            fired["hit"] = True
            try:
                slv.interrupt()
            except Exception:  # noqa: BLE001  # nothing useful to do from a timer thread
                pass

        timer = Timer(max(budget_s, 0.01), _stop)
        timer.daemon = True
        timer.start()

        start = time.perf_counter()
        res = slv.solve_limited(expect_interrupt=True)
        elapsed = time.perf_counter() - start

        timer.cancel()
        try:
            slv.clear_interrupt()
        except Exception:  # noqa: BLE001
            pass

        stats: dict[str, int] = {}
        try:
            stats = {str(k): int(v) for k, v in dict(slv.accum_stats()).items()}
        except Exception:  # noqa: BLE001  # not every build reports statistics
            stats = {}

        return ProbeOutcome(
            ok=True,
            solved=res is not None,
            satisfiable=(None if res is None else bool(res)),
            runtime_s=min(elapsed, budget_s) if res is None else elapsed,
            interrupted=bool(fired["hit"] and res is None),
            stats=stats,
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(ok=False, solved=False, satisfiable=None, runtime_s=0.0,
                            interrupted=False, stats={},
                            message=f"{type(exc).__name__}: {exc}")
    finally:
        if timer is not None:
            timer.cancel()
        if slv is not None:
            try:
                slv.delete()
            except Exception:  # noqa: BLE001
                pass
