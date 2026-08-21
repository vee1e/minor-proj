"""Feature extraction tests."""

from satsel.bench import generators as g
from satsel.features.extractor import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    extract_features,
)


class TestFeatures:
    def test_schema_fixed_across_budgets(self):
        c = g.gen_random3sat_batch(1, 1)[0]
        full = extract_features(c, 2.0)
        tiny = extract_features(c, 0.0001)
        assert len(full.names) == len(FEATURE_NAMES)
        assert full.names == tiny.names
        assert sum(tiny.missing) > 0
        assert sum(full.missing) == 0

    def test_under_120(self):
        assert len(FEATURE_NAMES) <= 120
        assert len(FEATURE_GROUPS["syntactic"]) > 0

    def test_deterministic(self):
        c = g.gen_random3sat_batch(1, 2)[0]
        # 40ms: always enough for the syntactic+graph groups, but below the
        # 50ms probe threshold, so no probe -> fully deterministic.
        a = extract_features(c, 0.04, seed=7)
        b = extract_features(c, 0.04, seed=7)
        missing = {n for n, m in zip(a.names, a.missing, strict=True) if m}
        assert "probe_runtime_s" in missing and "gr_density" not in missing
        assert a.values == b.values


    def test_budget_respected(self):
        c = g.gen_colouring_batch(1, 1)[0]
        fv = extract_features(c, 1.0)
        assert fv.cost_s < 2.0

    def test_parses_without_clauses(self):
        from satsel.core.dimacs import Cnf

        c = Cnf(nvars=3, nclauses=0, clauses=())
        fv = extract_features(c, 2.0)
        assert len(fv.names) == len(FEATURE_NAMES)
