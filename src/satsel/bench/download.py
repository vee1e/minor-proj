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
import tarfile
from pathlib import Path

import requests

from satsel.core.dimacs import parse_cnf

log = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 30.0
ARCHIVE_TIMEOUT_S = 180.0
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024      # refuse anything absurd
MAX_MEMBER_BYTES = 8 * 1024 * 1024        # a single CNF over 8 MB is not ours

SATLIB_BASE = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT"

# (family, archive path, satisfiability of the set, human label).
# Ordered by value to this project: the uf/uuf250-1065 sets sit exactly on the
# 4.26 phase transition, and the UNSAT half is where CDCL solvers differ most
# (an UNSAT answer needs a full refutation, so there is no lucky early exit).
SATLIB_ARCHIVES: tuple[tuple[str, str, str, str], ...] = (
    ("random3sat", "RND3SAT/uf250-1065.tar.gz", "sat", "uf250-1065"),
    ("random3sat", "RND3SAT/uuf250-1065.tar.gz", "unsat", "uuf250-1065"),
    ("random3sat", "RND3SAT/uf225-960.tar.gz", "sat", "uf225-960"),
    ("random3sat", "RND3SAT/uuf225-960.tar.gz", "unsat", "uuf225-960"),
    ("colouring", "GCP/flat200-479.tar.gz", "sat", "flat200-479"),
    ("colouring", "GCP/flat125-301.tar.gz", "sat", "flat125-301"),
    ("planning", "PLANNING/logistics.tar.gz", "sat", "logistics"),
    ("planning", "PLANNING/blocksworld.tar.gz", "sat", "blocksworld"),
)


def _get(url: str, *, https_only: bool = True,
         timeout: float = HTTP_TIMEOUT_S) -> bytes | None:
    if https_only and not url.startswith("https://"):
        log.debug("refusing non-https url %s", url)
        return None
    try:
        with requests.get(url, timeout=timeout, allow_redirects=True,
                          stream=True) as resp:
            if resp.status_code != 200:
                log.debug("HTTP %s for %s", resp.status_code, url)
                return None
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    log.warning("archive %s exceeds %d bytes; abandoning",
                                url, MAX_ARCHIVE_BYTES)
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    except requests.RequestException as exc:
        log.debug("download failed for %s: %s", url, exc)
        return None


def verify_cnf_bytes(data: bytes, family: str = "") -> bool:
    """Return True if the bytes parse as a consistent DIMACS CNF."""
    try:
        parse_cnf(data)
        return True
    except Exception:  # noqa: BLE001
        return False


def _cached_archive(url: str, cache_dir: Path, *, https_only: bool) -> bytes | None:
    """Fetch `url`, using `cache_dir` so a rebuild does not re-download."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    cached = cache_dir / f"{key}.tar.gz"
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
    try:
        cached.write_bytes(data)
    except OSError as exc:  # cache is an optimisation, not a requirement
        log.debug("could not cache %s: %s", url, exc)
    return data


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

    `family_counts` caps how many instances are taken per family, so a bench
    run never pulls more than it needs. Unreachable archives are skipped with a
    warning; an empty return simply means the generators supply everything.
    """
    cache_dir = Path(out_dir) / "_cache"
    out: list = []

    # Spread each family's quota evenly across the archives available for it.
    # Draining the first archive would hand back an all-SAT corpus, but the
    # UNSAT sets are exactly where CDCL solvers separate: a refutation has to
    # be complete, so there is no lucky early exit to mask a weak solver.
    by_family: dict[str, list[tuple[str, str, str]]] = {}
    for family, path, sat_status, label in SATLIB_ARCHIVES:
        by_family.setdefault(family, []).append((path, sat_status, label))

    for family, archives in by_family.items():
        want = int(family_counts.get(family, 0))
        if want <= 0:
            continue
        share, extra = divmod(want, len(archives))
        for idx, (path, sat_status, label) in enumerate(archives):
            quota = share + (1 if idx < extra else 0)
            if quota <= 0:
                continue
            url = f"{SATLIB_BASE}/{path}"
            data = _cached_archive(url, cache_dir, https_only=https_only)
            if data is None:
                log.warning("SATLIB archive unavailable, skipping: %s", label)
                continue
            members = _extract_cnfs(data, quota)
            if not members:
                log.warning("SATLIB archive %s yielded no usable CNFs", label)
                continue
            for payload, name in members:
                out.append((payload, family, f"satlib-{label}-{name}"))
            log.info("SATLIB %s: %d instances (%s) for family %s",
                     label, len(members), sat_status, family)

    if not out:
        log.warning("no SATLIB instances were downloaded; corpus will be "
                    "entirely generated (see dataset_summary.json warnings)")
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
