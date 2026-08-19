"""SolverOracle: the one public API.

Guarantees (DEVELOPMENT_PLAN.md §4):
- never raises; every failure is a typed SolverOracleResult with status=ERROR
- total time capped: all work runs in a child process the caller can kill
- takes content bytes, not paths (a path variant rejects symlinks and
  out-of-folder paths)
- size pre-checks before any solver memory is allocated
- one fresh solver instance per call
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from satsel.core.config import load_config
from satsel.core.proc import run_subprocess
from satsel.core.schemas import (
    ST_ERROR,
    ST_TIMEOUT,
    SolverOracleResult,
    oracle_error,
)

MODEL_FILE_NAMES = ("portfolio.json", "theta.json", "classical.joblib", "gnn.pt")


class SolverOracle:
    """Predict the fastest solver for a CNF, then run it."""

    def __init__(self, cfg: dict | None = None, model_dir: str | Path | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        if model_dir is not None:
            self.cfg["paths"]["models"] = Path(model_dir)
        self._model_cache: dict[str, object] = {}

    # -- public ------------------------------------------------------------

    def solve(self, cnf_bytes: bytes, timeout_s: float = 300.0) -> SolverOracleResult:
        try:
            if not isinstance(cnf_bytes, (bytes, bytearray)):
                return oracle_error("input must be bytes")
            max_bytes = int(self.cfg["oracle"]["max_bytes"])
            if len(cnf_bytes) > max_bytes:
                return oracle_error("input exceeds the size cap")
            return self._run_worker(bytes(cnf_bytes), timeout_s)
        except Exception:  # noqa: BLE001
            return oracle_error("internal failure")

    def solve_path(self, path: str | Path, timeout_s: float = 300.0) -> SolverOracleResult:
        try:
            p = Path(path)
            allowed = self._allowed_root()
            real = p.resolve(strict=False)
            try:
                abs_path = p.absolute().resolve(strict=False)
            except OSError:
                abs_path = p.absolute()
            if real != abs_path:
                return oracle_error("symlinks are not allowed")
            if not self._inside(real, allowed):
                return oracle_error("path is outside the allowed folder")
            if not real.is_file():
                return oracle_error("file does not exist")
            return self.solve(real.read_bytes(), timeout_s)
        except Exception:  # noqa: BLE001
            return oracle_error("internal failure")

    # -- internals ---------------------------------------------------------

    def _allowed_root(self) -> Path:
        # repo root = parent of the package's src dir; fall back to cwd.
        pkg = Path(__file__).resolve().parents[3]
        return pkg if pkg.is_dir() else Path.cwd()

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _worker_argv(self) -> list[str]:
        return [sys.executable, "-m", "satsel.oracle.worker"]

    def _run_worker(self, cnf_bytes: bytes, timeout_s: float) -> SolverOracleResult:
        overhead = float(self.cfg["oracle"].get("budget_overhead_s", 30))
        wall_cap = timeout_s + overhead
        model_dir = str(Path(self.cfg["paths"]["models"]))
        argv = self._worker_argv() + [
            "--cnf-b64", base64.b64encode(cnf_bytes).decode(),
            "--timeout", str(timeout_s),
            "--model-dir", model_dir,
        ]
        res = run_subprocess(argv, timeout_s=wall_cap, input_bytes=None)
        if res.timed_out:
            return SolverOracleResult(
                status=ST_TIMEOUT, satisfiable=None, assignment=None,
                solver_used=None, solver_version=None,
                runtime_ms=int(res.wall_s * 1000), parse_ms=0, feature_ms=0,
                infer_ms=0, solve_ms=0, features=None, confidence=None,
                theta_used=None, fallback_used=True, fallback_reason="timeout",
                error=None)
        try:
            payload = json.loads(res.stdout.decode().strip())
        except Exception:  # noqa: BLE001
            return oracle_error("worker returned unparseable output")
        if not payload.get("ok"):
            return oracle_error(str(payload.get("error", "worker failed"))[:500])
        r = payload.get("result", {})
        return SolverOracleResult(
            status=r.get("status", ST_ERROR),
            satisfiable=r.get("satisfiable"),
            assignment=tuple(r["assignment"]) if r.get("assignment") is not None else None,
            solver_used=r.get("solver_used"),
            solver_version=r.get("solver_version"),
            runtime_ms=int(r.get("runtime_ms", 0)),
            parse_ms=int(r.get("parse_ms", 0)),
            feature_ms=int(r.get("feature_ms", 0)),
            infer_ms=int(r.get("infer_ms", 0)),
            solve_ms=int(r.get("solve_ms", 0)),
            features=tuple(r["features"]) if r.get("features") is not None else None,
            confidence=r.get("confidence"),
            theta_used=r.get("theta_used"),
            fallback_used=bool(r.get("fallback_used", False)),
            fallback_reason=r.get("fallback_reason"),
            error=r.get("error"),
        )
