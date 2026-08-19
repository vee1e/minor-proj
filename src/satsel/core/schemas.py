"""Shared data contracts frozen at M0.

These are the inter-stage schemas. Everything downstream consumes these
dataclasses; nothing else may define its own row layout for the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Status values for a single (formula, solver) profiling outcome.
RUN_SOLVED = "solved"
RUN_TIMEOUT = "timeout"
RUN_CRASH = "crash"
RUN_ERROR = "error"

# Oracle result statuses.
ST_SAT = "SAT"
ST_UNSAT = "UNSAT"
ST_TIMEOUT = "TIMEOUT"
ST_CRASH = "CRASH"
ST_ERROR = "ERROR"

# Feature value kinds (for the missing-value marker scheme).
FEAT_SYNTACTIC = "syntactic"
FEAT_GRAPH = "graph"
FEAT_PROBE = "probe"
FEAT_MISSING = "missing"  # marker for values dropped by the time budget


@dataclass(frozen=True)
class FormulaMeta:
    """Identity + provenance of one CNF formula."""

    formula_id: str
    family: str
    source: str  # generated | satlib | gbd
    name: str
    nvars: int
    nclauses: int
    canonical_hash: str
    renamesafe_hash: str
    path: str = ""
    is_ood: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.nvars + self.nclauses


@dataclass(frozen=True)
class RunRecord:
    """Outcome of profiling one (formula, solver) pair."""

    formula_id: str
    solver: str
    status: str  # RUN_*
    runtime_s: float  # median of finished runs; tau if censored
    runs: tuple[float, ...]
    censored: bool
    tau_s: float
    solver_version: str = ""


@dataclass(frozen=True)
class Label:
    """Derived label for one formula: best solver, per-solver runtimes."""

    formula_id: str
    family: str
    best_solver: str
    best_time_s: float
    times: dict[str, float]  # solver -> runtime_s (tau if unsolved)
    solved: dict[str, bool]
    weights: dict[str, float]  # solver -> cost weight w(i,s)
    tie_flag: bool
    vbs_solved: bool


@dataclass(frozen=True)
class SolverOracleResult:
    """The one public result object. Frozen; every field is populated for
    every outcome, including failures."""

    status: str
    satisfiable: bool | None
    assignment: tuple[int, ...] | None
    solver_used: str | None
    solver_version: str | None
    runtime_ms: int
    parse_ms: int
    feature_ms: int
    infer_ms: int
    solve_ms: int
    features: tuple[float, ...] | None
    confidence: float | None
    theta_used: float | None
    fallback_used: bool
    fallback_reason: str | None
    error: str | None

    @property
    def total_ms(self) -> int:
        return self.parse_ms + self.feature_ms + self.infer_ms + self.solve_ms

    def to_dict(self) -> dict[str, Any]:
        d = {
            "status": self.status,
            "satisfiable": self.satisfiable,
            "solver_used": self.solver_used,
            "solver_version": self.solver_version,
            "runtime_ms": self.runtime_ms,
            "parse_ms": self.parse_ms,
            "feature_ms": self.feature_ms,
            "infer_ms": self.infer_ms,
            "solve_ms": self.solve_ms,
            "features": list(self.features) if self.features is not None else None,
            "confidence": self.confidence,
            "theta_used": self.theta_used,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
        }
        if self.assignment is not None:
            d["assignment"] = list(self.assignment)
        return d


def oracle_error(message: str) -> SolverOracleResult:
    return SolverOracleResult(
        status=ST_ERROR,
        satisfiable=None,
        assignment=None,
        solver_used=None,
        solver_version=None,
        runtime_ms=0,
        parse_ms=0,
        feature_ms=0,
        infer_ms=0,
        solve_ms=0,
        features=None,
        confidence=None,
        theta_used=None,
        fallback_used=False,
        fallback_reason=None,
        error=message,
    )
