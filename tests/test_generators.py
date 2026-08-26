"""Generator round-trip tests (quick) and ground-truth checks (slow)."""

import pytest

from satsel.bench import generators as g
from satsel.core.dimacs import parse_cnf


class TestGeneratorsQuick:
    def test_random3sat_roundtrip(self):
        cnfs = g.gen_random3sat_batch(5, 1)
        for c in cnfs:
            assert parse_cnf(c.to_dimacs()).clauses == c.clauses
            assert len(c.clauses) >= 1

    def test_families_consistent(self):
        for fn in (g.gen_colouring_batch, g.gen_planning_batch,
                   g.gen_scheduling_batch, g.gen_industrial_batch,
                   g.gen_crypto_batch):
            cnfs = fn(3, 1)
            assert len(cnfs) == 3
            for c in cnfs:
                assert c.nvars == max((abs(lit) for cl in c.clauses for lit in cl), default=0)

    def test_random3sat_phase_transition_sizes(self):
        cnfs = g.gen_random3sat_batch(6, 3)
        for c in cnfs:
            assert 200 <= c.nvars <= 320
            assert abs(c.nclauses / c.nvars - 4.26) < 0.02
        large = g.gen_random3sat_large_batch(6, 3)
        for c in large:
            assert 320 <= c.nvars <= 420
            assert 4.15 <= c.nclauses / c.nvars <= 4.35

    def test_deterministic(self):
        a = g.gen_colouring_batch(3, 11)
        b = g.gen_colouring_batch(3, 11)
        assert [c.clauses for c in a] == [c.clauses for c in b]
        a = g.gen_scheduling_batch(2, 11)
        b = g.gen_scheduling_batch(2, 11)
        assert [c.clauses for c in a] == [c.clauses for c in b]

    def test_generate_all_counts(self):
        out = g.generate_all({"random3sat": 5, "random3sat_large": 5,
                              "colouring": 5, "planning": 5, "industrial": 5,
                              "scheduling": 5, "ood_scheduling": 5, "crypto": 3})
        for fam, n in {"random3sat": 5, "colouring": 5, "crypto": 3}.items():
            assert len(out[fam]) == n


@pytest.mark.slow
class TestGeneratorsSlow:
    def test_planning_ground_truth(self):
        from satsel.core.solvers import solve_in_process

        cnfs = g.gen_planning_batch(6, 1)
        for c in cnfs:
            o = solve_in_process(c.clauses, c.nvars, "cadical153", 30.0)
            assert o.status == "solved"
            assert bool(o.satisfiable) == bool(c.extra["ground_truth"])

    def test_scheduling_ground_truth(self):
        from satsel.core.solvers import solve_in_process

        cnfs = g.gen_scheduling_batch(6, 1)
        for c in cnfs:
            o = solve_in_process(c.clauses, c.nvars, "cadical153", 30.0)
            assert o.status == "solved"
            assert bool(o.satisfiable) == bool(c.extra["ground_truth"])
