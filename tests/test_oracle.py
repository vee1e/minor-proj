"""Oracle tests: never-raises, path safety, and (slow) real solve."""

import copy
import json

import pytest

from satsel.core.config import load_config
from satsel.oracle.oracle import SolverOracle

DUMMY_MODELS = {
    "portfolio.json": {
        "solvers": ["minisat22", "glucose4", "cadical153", "maplesat",
                    "lingeling", "maplecm", "kissat404"],
        "best_average_solver": "kissat404",
        "mean_best_solver_median": 0.5,
    },
    "theta.json": {"theta": 0.5, "fallback_solver": "kissat404"},
}


@pytest.fixture
def oracle(tmp_path):
    for name, content in DUMMY_MODELS.items():
        (tmp_path / name).write_text(json.dumps(content))
    cfg = load_config()
    c = copy.deepcopy(cfg)
    c["paths"] = dict(cfg["paths"])
    c["paths"]["models"] = tmp_path
    return SolverOracle(c)


class TestOracleQuick:
    def test_never_raises_on_garbage(self, oracle):
        assert oracle.solve(b"garbage").status == "ERROR"
        assert oracle.solve(b"").status == "ERROR"

    def test_path_safety(self, oracle):
        r = oracle.solve_path("/etc/passwd")
        assert r.status == "ERROR"

    def test_size_cap(self, oracle):
        r = oracle.solve(b"p cnf 1 1\n1 0\n" * 10_000_000)
        assert r.status == "ERROR"

    def test_typed_result(self, oracle):
        r = oracle.solve(b"garbage")
        assert hasattr(r, "status") and hasattr(r, "error")


@pytest.mark.slow
class TestOracleSlow:
    def test_real_solve(self, oracle):
        from satsel.bench import generators as g

        c = g.gen_random3sat_batch(1, 1)[0]
        r = oracle.solve(c.to_dimacs(), timeout_s=10)
        assert r.status in ("SAT", "UNSAT")
        assert r.solver_used in DUMMY_MODELS["portfolio.json"]["solvers"]
