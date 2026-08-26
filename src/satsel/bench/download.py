"""Downloaders for real SATLIB benchmark instances.

The historical `www.satlib.org` host has been a parked domain for years, which
is why earlier builds of this module never actually downloaded anything and the
corpus was 100% synthetic. The canonical SATLIB mirror is Holger Hoos' page at
UBC, which is reachable and serves the original gzipped tarballs:

    https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/...

Each archive is fetched once, cached under `<raw>/_cache/`, and unpacked in
memory. Every extracted file must parse as DIMACS CNF or it is discarded, so a
hijacked host or an HTML error page can never contaminate the corpus.

Downloads are strictly bounded and never block a build: any failure is logged
and the bench pipeline fills the shortfall from the seeded generators. Whether
an instance came from SATLIB or a generator is recorded per formula in the
manifest `source` column, so the provenance of the dataset is auditable rather
than assumed.

GBD (the Global Benchmark Database, which fronts SAT Competition instances)
requires an API token and per-instance queries; it is probed but not used.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tarfile
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin

import requests

from satsel.core.dimacs import parse_cnf

log = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 30.0
ARCHIVE_TIMEOUT_S = 180.0
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024      # refuse anything absurd
MAX_MEMBER_BYTES = 8 * 1024 * 1024        # a single CNF over 8 MB is not ours

SATLIB_BASE = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT"


class Archive(NamedTuple):
    """One SATLIB tarball and what it is worth to this study.

    `weight` drives how a family's quota is split across its archives. An even
    split would drown the hard instances in easy ones: uf150 and CBS solve in
    milliseconds on every solver in the portfolio and so contribute no signal
    about which solver to pick, while uf250 and especially uuf250 sit right on
    the difficulty peak. `n_instances` is the true member count, measured by
    unpacking each archive, and is used for quota clamping and for `satsel
    sources` so nobody has to guess what a full fetch yields.
    """

    family: str
    path: str
    status: str          # "sat" | "unsat" | "mixed"
    label: str
    n_instances: int
    nvars: int
    weight: int


# Measured by unpacking every archive; totals 2,210 real instances, which alone
# meets the N >= 2,000 target in the synopsis. Generators then only need to
# cover the families SATLIB has no equivalent for (industrial, crypto, and the
# held-out OOD family).
#
# The uf/uuf pairs matter: uf is satisfiable, uuf is not. Equal weight on both
# keeps the corpus balanced, and the UNSAT half is where CDCL solvers separate
# most, because proving unsatisfiability requires exhausting the search space
# rather than stumbling onto a solution early.
SATLIB_ARCHIVES: tuple[Archive, ...] = (
    # Random 3-SAT at the 4.26 phase transition, hardest first.
    Archive("random3sat", "RND3SAT/uf250-1065.tar.gz", "sat", "uf250-1065", 100, 250, 5),
    Archive("random3sat", "RND3SAT/uuf250-1065.tar.gz", "unsat", "uuf250-1065", 100, 250, 5),
    Archive("random3sat", "RND3SAT/uf225-960.tar.gz", "sat", "uf225-960", 100, 225, 4),
    Archive("random3sat", "RND3SAT/uuf225-960.tar.gz", "unsat", "uuf225-960", 100, 225, 4),
    Archive("random3sat", "RND3SAT/uf200-860.tar.gz", "sat", "uf200-860", 100, 200, 3),
    Archive("random3sat", "RND3SAT/uuf200-860.tar.gz", "unsat", "uuf200-860", 99, 200, 3),
    Archive("random3sat", "RND3SAT/uf175-753.tar.gz", "sat", "uf175-753", 100, 175, 1),
    Archive("random3sat", "RND3SAT/uuf175-753.tar.gz", "unsat", "uuf175-753", 100, 175, 1),
    Archive("random3sat", "RND3SAT/uf150-645.tar.gz", "sat", "uf150-645", 100, 150, 1),
    Archive("random3sat", "RND3SAT/uuf150-645.tar.gz", "unsat", "uuf150-645", 100, 150, 1),
    # Controlled backbone size, 1,000 instances but small; an easy tier that
    # adds size diversity without being allowed to dominate.
    Archive("random3sat", "CBS/CBS_k3_n100_m403_b10.tar.gz", "sat",
            "CBS_k3_n100_m403_b10", 1000, 100, 1),
    # Graph colouring (flat graphs, 3-colourable).
    Archive("colouring", "GCP/flat200-479.tar.gz", "sat", "flat200-479", 100, 600, 3),
    Archive("colouring", "GCP/flat125-301.tar.gz", "sat", "flat125-301", 100, 375, 2),
    # Planning. Only 11 instances exist, so take all of them.
    Archive("planning", "PLANNING/logistics.tar.gz", "sat", "logistics", 4, 828, 1),
    Archive("planning", "PLANNING/blocksworld.tar.gz", "sat", "blocksworld", 7, 459, 1),
)


def archives_for(family: str) -> list[Archive]:
    return [a for a in SATLIB_ARCHIVES if a.family == family]


def total_available(family: str | None = None) -> int:
    """How many real instances a full fetch would yield."""
    pool = SATLIB_ARCHIVES if family is None else archives_for(family)
    return sum(a.n_instances for a in pool)


def _tiebreak(label: str) -> int:
    """Stable deterministic order between candidates that tie on skew too."""
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def _skew_if(eligible: list[Archive], quota: dict[str, int],
             base: dict[str, int], pick: Archive) -> int:
    """|sat - unsat| across the allocation if `pick` receives one more unit."""
    s = sum(quota[a.label] + base[a.label]
            for a in eligible if a.status == "sat")
    u = sum(quota[a.label] + base[a.label]
            for a in eligible if a.status == "unsat")
    if pick.status == "sat":
        s += 1
    elif pick.status == "unsat":
        u += 1
    return abs(s - u)


def allocate(archives: list[Archive], want: int) -> dict[str, int]:
    """Split `want` across `archives` by weight, clamped to what each holds.

    Uses largest-remainder apportionment rather than a sequential greedy split,
    so two archives with the same weight receive the same quota wherever the
    total allows it. That matters here: uf and uuf sets at a given size are
    deliberately weighted equally to keep the corpus balanced between
    satisfiable and unsatisfiable instances, and a greedy loop would quietly
    skew it toward whichever came first.

    Leftover units (from rounding, unavoidable on odd quotas) go to the tied
    archive that least disturbs the sat/unsat balance, so an odd request never
    systematically lands on the satisfiable half.

    Overflow from an archive clamped at its true size is redistributed to
    archives that still have room, so asking for more than one set can supply
    never silently loses instances.
    """
    if want <= 0 or not archives:
        return {}

    quota = {a.label: 0 for a in archives}
    remaining = want
    eligible = [a for a in archives if a.n_instances > 0]

    while remaining > 0 and eligible:
        total_w = sum(a.weight for a in eligible)
        if total_w <= 0:
            break

        # Hare quota: floor of the exact share, then hand out what is left over.
        exact = {a.label: remaining * a.weight / total_w for a in eligible}
        base = {lbl: int(val) for lbl, val in exact.items()}
        leftover = remaining - sum(base.values())
        eps = 1e-9
        for _ in range(leftover):
            rem = {a.label: exact[a.label] - base[a.label] for a in eligible}
            top = max(rem.values())
            tied = [a for a in eligible if rem[a.label] > top - eps]
            with_room = [a for a in tied
                         if quota[a.label] + base[a.label] + 1 <= a.n_instances]
            pool = with_room or tied
            pick = min(pool, key=lambda a: (_skew_if(eligible, quota, base, a),
                                            _tiebreak(a.label)))
            base[pick.label] += 1

        progressed = False
        for a in list(eligible):
            room = a.n_instances - quota[a.label]
            take = min(base[a.label], room)
            if take > 0:
                quota[a.label] += take
                remaining -= take
                progressed = True
            if quota[a.label] >= a.n_instances:
                eligible.remove(a)
        if not progressed:
            break

    return {k: v for k, v in quota.items() if v > 0}


def _get(url: str, *, https_only: bool = True,
         timeout: float = HTTP_TIMEOUT_S) -> bytes | None:
    """Fetch `url` with redirects followed manually.

    `allow_redirects=False` matters: with auto-redirects a 302 to
    `http://...` would be followed silently and the https_only check on the
    initial URL would never see the downgrade.
    """
    if https_only and not url.startswith("https://"):
        log.debug("refusing non-https url %s", url)
        return None
    current = url
    for _ in range(10):
        try:
            with requests.get(current, timeout=timeout, allow_redirects=False,
                              stream=True) as resp:
                if resp.is_redirect or resp.is_permanent_redirect:
                    loc = resp.headers.get("Location")
                    if not loc:
                        return None
                    nxt = urljoin(current, loc)
                    if https_only and not nxt.startswith("https://"):
                        log.warning("refusing redirect from %s to non-https %s",
                                    current, nxt)
                        return None
                    current = nxt
                    continue
                if resp.status_code != 200:
                    log.debug("HTTP %s for %s", resp.status_code, current)
                    return None
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        log.warning("archive %s exceeds %d bytes; abandoning",
                                    current, MAX_ARCHIVE_BYTES)
                        return None
                    chunks.append(chunk)
                return b"".join(chunks)
        except requests.RequestException as exc:
            log.debug("download failed for %s: %s", current, exc)
            return None
    log.warning("too many redirects for %s", url)
    return None


def verify_cnf_bytes(data: bytes, family: str = "") -> bool:
    """Return True if the bytes parse as a consistent DIMACS CNF."""
    try:
        parse_cnf(data)
        return True
    except Exception:  # noqa: BLE001
        return False


def _archive_cache_path(url: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()
    return cache_dir / f"{key}.tar.gz"


def _cached_archive(url: str, cache_dir: Path, *, https_only: bool) -> bytes | None:
    """Fetch `url`, using `cache_dir` so a rebuild does not re-download.

    The write is atomic (tmp file + rename) so an interrupted download can
    never leave a truncated file that later runs mistake for a valid cache
    entry.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _archive_cache_path(url, cache_dir)
    if cached.exists():
        try:
            data = cached.read_bytes()
            if data:
                log.info("using cached archive for %s", url)
                return data
        except OSError:
            pass
    data = _get(url, https_only=https_only, timeout=ARCHIVE_TIMEOUT_S)
    if data is None:
        return None
    tmp = cached.with_suffix(".tar.gz.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, cached)
    except OSError as exc:  # cache is an optimisation, not a requirement
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        log.debug("could not cache %s: %s", url, exc)
    return data


def evict_archive(url: str, cache_dir: Path) -> None:
    """Drop a cache entry whose contents turned out to be unusable, so the
    next run re-downloads instead of failing on the same corrupt file forever."""
    try:
        _archive_cache_path(url, cache_dir).unlink(missing_ok=True)
    except OSError:
        pass


def _extract_cnfs(data: bytes, limit: int) -> list[tuple[bytes, str]]:
    """Unpack a gzipped tar in memory, returning up to `limit` (bytes, name).

    Every member is validated as DIMACS CNF; anything else is skipped, so a
    redirect to an HTML page yields an empty list rather than junk formulas.
    """
    out: list[tuple[bytes, str]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf:
                if len(out) >= limit:
                    break
                if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                    continue
                name = Path(member.name).name
                if not name.lower().endswith((".cnf", ".dimacs")):
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                payload = fh.read()
                if not verify_cnf_bytes(payload):
                    log.debug("skipping non-CNF member %s", name)
                    continue
                out.append((payload, Path(name).stem))
    except (tarfile.TarError, OSError, EOFError) as exc:
        log.warning("could not unpack archive: %s", exc)
        return []
    return out


def download_satlib(family_counts: dict[str, int], out_dir: Path,
                    *, https_only: bool = True) -> list:
    """Download real SATLIB instances. Returns a list of (data, family, name).

    `family_counts` caps how many instances are taken per family. Within a
    family the quota is split by archive weight rather than evenly, so a
    request for 400 random-3SAT instances is filled mostly from the hard
    phase-transition sets rather than diluted across the easy ones.

    Unreachable archives are skipped with a warning; an empty return simply
    means the generators supply everything.
    """
    cache_dir = Path(out_dir) / "_cache"
    out: list = []

    for family in sorted({a.family for a in SATLIB_ARCHIVES}):
        want = int(family_counts.get(family, 0))
        if want <= 0:
            continue
        pool = archives_for(family)
        available = sum(a.n_instances for a in pool)
        if want > available:
            log.info("family %s: asked for %d, SATLIB holds %d; generators will "
                     "cover the remaining %d", family, want, available, want - available)
        quota = allocate(pool, want)

        for archive in pool:
            take = quota.get(archive.label, 0)
            if take <= 0:
                continue
            data = _cached_archive(f"{SATLIB_BASE}/{archive.path}", cache_dir,
                                   https_only=https_only)
            if data is None:
                log.warning("SATLIB archive unavailable, skipping: %s", archive.label)
                continue
            members = _extract_cnfs(data, take)
            if not members:
                # A cached-but-corrupt archive would otherwise fail every run;
                # evict so the next attempt re-downloads.
                log.warning("SATLIB archive %s yielded no usable CNFs; "
                            "evicting cache entry", archive.label)
                evict_archive(f"{SATLIB_BASE}/{archive.path}", cache_dir)
                continue
            for payload, name in members:
                out.append((payload, family, f"satlib-{archive.label}-{name}"))
            log.info("SATLIB %-22s %3d instances (%s, n=%d) -> %s",
                     archive.label, len(members), archive.status, archive.nvars, family)

    if not out:
        log.warning("no SATLIB instances were downloaded; corpus will be "
                    "entirely generated (see dataset_summary.json warnings)")
    return out


def prefetch(out_dir: Path, *, https_only: bool = True,
             progress=None) -> dict[str, str]:
    """Download every archive into the cache without building a corpus.

    Useful before working somewhere with no network: warm the cache here, copy
    `<raw>/_cache/` to the target machine, and `bench` will run entirely
    offline. Returns {label: status}.
    """
    cache_dir = Path(out_dir) / "_cache"
    results: dict[str, str] = {}
    for archive in SATLIB_ARCHIVES:
        if progress:
            progress(archive.label)
        data = _cached_archive(f"{SATLIB_BASE}/{archive.path}", cache_dir,
                               https_only=https_only)
        if data is None:
            results[archive.label] = "unreachable"
            continue
        members = _extract_cnfs(data, 10 ** 6)
        results[archive.label] = (f"ok ({len(members)} instances)" if members
                                  else "empty (no usable CNFs)")
    return results


def cache_status(out_dir: Path) -> dict[str, bool]:
    """Which archives are already cached locally."""
    cache_dir = Path(out_dir) / "_cache"
    out: dict[str, bool] = {}
    for archive in SATLIB_ARCHIVES:
        url = f"{SATLIB_BASE}/{archive.path}"
        out[archive.label] = _archive_cache_path(url, cache_dir).exists()
    return out


def download_gbd(family_counts: dict[str, int], out_dir: Path,
                 *, https_only: bool = True) -> list:
    """Probe the Global Benchmark Database (SAT Competition instances).

    GBD gates bulk instance access behind an API token and per-hash queries, so
    this is a reachability probe only and always returns []. Wiring it up is
    tracked as future work; SAT Competition instances are also individually far
    larger than anything else in this corpus and would dominate profiling time.
    """
    base = "https://gbd.iti.kit.edu"
    try:
        resp = requests.get(base, timeout=HTTP_TIMEOUT_S)
        if resp.status_code != 200:
            log.warning("GBD unreachable (HTTP %s); skipping", resp.status_code)
    except requests.RequestException as exc:
        log.debug("GBD unreachable: %s", exc)
    return []
