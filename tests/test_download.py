"""Tests for benchmark acquisition.

All offline: the allocator is pure logic, and archive unpacking is exercised
against tarballs built in the test itself. Nothing here touches the network.
"""

from __future__ import annotations

import io
import tarfile

from satsel.bench.download import (
    SATLIB_ARCHIVES,
    Archive,
    _extract_cnfs,
    allocate,
    archives_for,
    cache_status,
    total_available,
)
from satsel.core.dimacs import parse_cnf

SATLIB_CNF = b"""c a real SATLIB file, terminator and all
p cnf 3 2
1 -2 3 0
-1 2 0
%
0
"""


def _tarball(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class TestArchiveTable:
    def test_counts_are_declared(self):
        for a in SATLIB_ARCHIVES:
            assert a.n_instances > 0, f"{a.label} declares no instances"
            assert a.weight > 0, f"{a.label} has no weight"
            assert a.status in ("sat", "unsat", "mixed")

    def test_total_meets_the_synopsis_target(self):
        # C1 promises N >= 2,000; SATLIB alone should now cover it.
        assert total_available() >= 2000

    def test_family_lookup(self):
        assert {a.family for a in archives_for("random3sat")} == {"random3sat"}
        assert archives_for("nonexistent") == []

    def test_sat_and_unsat_sets_are_paired(self):
        """Each uf<size> should have a matching uuf<size> at equal weight."""
        rnd = [a for a in archives_for("random3sat") if a.label.startswith(("uf", "uuf"))]
        by_size: dict[int, list[Archive]] = {}
        for a in rnd:
            by_size.setdefault(a.nvars, []).append(a)
        for nvars, group in by_size.items():
            kinds = {a.status for a in group}
            assert kinds == {"sat", "unsat"}, f"n={nvars} is not paired: {kinds}"
            assert len({a.weight for a in group}) == 1, (
                f"n={nvars} weights differ, which would skew the SAT/UNSAT balance")


class TestAllocator:
    def test_respects_the_request(self):
        q = allocate(archives_for("random3sat"), 400)
        assert sum(q.values()) == 400

    def test_never_exceeds_what_an_archive_holds(self):
        pool = archives_for("random3sat")
        sizes = {a.label: a.n_instances for a in pool}
        q = allocate(pool, 1500)
        for label, n in q.items():
            assert n <= sizes[label], f"{label} over-allocated"

    def test_clamps_to_availability(self):
        pool = archives_for("planning")           # only 11 instances exist
        q = allocate(pool, 500)
        assert sum(q.values()) == total_available("planning")

    def test_equal_weights_get_equal_quota(self):
        """SAT/UNSAT balance depends on this; a greedy split broke it."""
        for want in (40, 137, 400, 800):
            q = allocate(archives_for("random3sat"), want)
            assert q.get("uf250-1065") == q.get("uuf250-1065"), (
                f"want={want} skewed the hardest SAT/UNSAT pair")

    def test_prefers_heavier_archives(self):
        q = allocate(archives_for("random3sat"), 200)
        assert q.get("uf250-1065", 0) > q.get("uf150-645", 0)
        assert q.get("uf250-1065", 0) > q.get("CBS_k3_n100_m403_b10", 0)

    def test_degenerate_requests(self):
        assert allocate(archives_for("random3sat"), 0) == {}
        assert allocate(archives_for("random3sat"), -5) == {}
        assert allocate([], 100) == {}

    def test_single_instance_request(self):
        q = allocate(archives_for("random3sat"), 1)
        assert sum(q.values()) == 1


class TestExtraction:
    def test_reads_satlib_terminator(self):
        """Real SATLIB files end with '%' then '0'; the parser used to raise."""
        cnf = parse_cnf(SATLIB_CNF)
        assert cnf.nvars == 3
        assert cnf.nclauses == 2

    def test_extracts_cnf_members(self):
        data = _tarball({
            "dir/a.cnf": SATLIB_CNF,
            "dir/b.cnf": SATLIB_CNF,
        })
        got = _extract_cnfs(data, limit=10)
        assert len(got) == 2
        assert {name for _, name in got} == {"a", "b"}

    def test_honours_the_limit(self):
        data = _tarball({f"x{i}.cnf": SATLIB_CNF for i in range(9)})
        assert len(_extract_cnfs(data, limit=4)) == 4

    def test_skips_non_cnf_members(self):
        data = _tarball({
            "readme.txt": b"not a formula",
            "index.html": b"<html>hijacked</html>",
            "good.cnf": SATLIB_CNF,
        })
        got = _extract_cnfs(data, limit=10)
        assert len(got) == 1 and got[0][1] == "good"

    def test_rejects_cnf_named_garbage(self):
        """A .cnf that does not parse must not enter the corpus."""
        data = _tarball({"fake.cnf": b"<html>404 not found</html>"})
        assert _extract_cnfs(data, limit=10) == []

    def test_corrupt_archive_returns_empty(self):
        assert _extract_cnfs(b"this is not a tarball", limit=10) == []


class TestCacheStatus:
    def test_reports_every_archive(self, tmp_path):
        status = cache_status(tmp_path)
        assert set(status) == {a.label for a in SATLIB_ARCHIVES}
        assert all(v is False for v in status.values())
