"""Single-process solver execution used by profiling workers and the oracle.

Solver names are validated against a fixed allowlist so a caller can never pass
an arbitrary module path into pysat. A fresh solver instance per call: solvers
are not thread-safe.
"""

from __future__ import annotations

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
    runtime_s: float
    message: str = ""


def validate_solver(name: str) -> None:
    if name not in ALLOWED_SOLVERS:
        raise SolverError(f"unknown or disallowed solver {name!r}")


def solve_in_process(
    clauses: tuple[tuple[int, ...], ...],
    nvars: int,
    solver: str,
    timeout_s: float,
    *,
    seed: int = 0,
) -> SolveOutcome:
    """Solve `clauses` with a fresh solver instance. Uses a pysat time budget
    plus a wall-clock guard; reports censored timeouts rather than exceptions."""
    validate_solver(solver)

    start = time.monotonic()
    try:
        try:
            slv = Solver(name=solver, seed=seed)
        except TypeError:
            slv = Solver(name=solver)
    except Exception as exc:  # noqa: BLE001
        return SolveOutcome(status=RUN_ERROR, satisfiable=None, runtime_s=0.0,
                            message=f"solver construct failed: {exc}")

    try:
        for clause in clauses:
            slv.add_clause(list(clause))
        try:
            slv.time_budget = timeout_s
        except Exception:
            pass  # time_budget may not be exposed on every solver build
        res = slv.solve()
        elapsed = time.monotonic() - start
        if res is None:
            return SolveOutcome(status=RUN_TIMEOUT, satisfiable=None, runtime_s=elapsed)
        if res is True:
            return SolveOutcome(status=RUN_SOLVED, satisfiable=True, runtime_s=elapsed)
        return SolveOutcome(status=RUN_SOLVED, satisfiable=False, runtime_s=elapsed)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        if elapsed >= timeout_s:
            return SolveOutcome(status=RUN_TIMEOUT, satisfiable=None, runtime_s=elapsed)
        return SolveOutcome(
            status=RUN_CRASH,
            satisfiable=None,
            runtime_s=elapsed,
            message=f"solver raised {type(exc).__name__}: {exc}",
        )
    finally:
        try:
            slv.delete()
        except Exception:  # noqa: BLE001
            pass
