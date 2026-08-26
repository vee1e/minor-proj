"""Quick tests for the core package: DIMACS parsing, hashing, config."""

import pytest

from satsel.core.config import load_config
from satsel.core.dimacs import Cnf, DimacsError, parse_cnf
from satsel.core.proc import run_subprocess


class TestDimacs:
    def test_roundtrip(self):
        c = Cnf(nvars=3, nclauses=2, clauses=((1, -2, 3), (-1, 3)))
        parsed = parse_cnf(c.to_dimacs())
        assert parsed.clauses == c.clauses
        assert parsed.nvars == 3 and parsed.nclauses == 2

    def test_missing_header(self):
        with pytest.raises(DimacsError):
            parse_cnf("1 2 0\n-1 0\n")

    def test_header_mismatch(self):
        with pytest.raises(DimacsError):
            parse_cnf("p cnf 3 5\n1 0\n-1 0\n")

    def test_literal_out_of_range(self):
        with pytest.raises(DimacsError):
            parse_cnf("p cnf 2 1\n3 0\n")

    def test_caps(self):
        with pytest.raises(DimacsError):
            parse_cnf("p cnf 3 1\n1 0\n", max_vars=2)

    def test_empty_clause_ok(self):
        c = parse_cnf("p cnf 1 1\n0\n")
        assert c.clauses == ((),)

    def test_canonical_hash_order_invariant(self):
        a = Cnf(nvars=3, nclauses=2, clauses=((3, 1), (2, -3)))
        b = Cnf(nvars=3, nclauses=2, clauses=((2, -3), (1, 3)))
        assert a.canonical_hash == b.canonical_hash
        c = Cnf(nvars=3, nclauses=2, clauses=((3, 1), (2, -4)))
        assert a.canonical_hash != c.canonical_hash


class TestConfig:
    def test_load(self):
        cfg = load_config()
        assert cfg["tau_screen"] == 30
        assert cfg["paths"]["db"].suffix == ".db"

    def test_tau_tiers(self):
        cfg = load_config()
        assert cfg["tau_tiers"]["random3sat"] == 60
        assert cfg["tau_tiers"]["scheduling"] == 180


class TestProc:
    def test_returns(self):
        r = run_subprocess(["true"], timeout_s=5)
        assert r.returncode == 0 and not r.timed_out

    def test_timeout_kills(self):
        r = run_subprocess(["sleep", "30"], timeout_s=1)
        assert r.timed_out
        assert r.returncode != 0
