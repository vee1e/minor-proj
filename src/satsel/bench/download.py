"""Best-effort, strictly bounded downloaders for SATLIB / GBD instances.

As of this build the historical hosts are unreliable (www.satlib.org is
hijacked, DIMACS FTP and GBD are unreachable), so downloads are only probes:
a single connectivity check plus at most a handful of file attempts, all with
short timeouts. The bench pipeline fills any shortfall from the project's own
seeded generators, so downloads never block or slow the build.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from satsel.core.dimacs import parse_cnf

log = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 5.0
MAX_DOWNLOAD_ATTEMPTS = 10


def _get(url: str, *, https_only: bool = True) -> bytes | None:
    if https_only and not url.startswith("https://"):
        return None
    try:
        with requests.get(url, timeout=HTTP_TIMEOUT_S, allow_redirects=True) as resp:
            if resp.status_code != 200:
                return None
            return resp.content
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


def download_satlib(family_counts: dict[str, int], out_dir: Path,
                    *, https_only: bool = True) -> list:
    """Bounded SATLIB probe. Returns a list of (data, family, name).

    In this environment the host answers with a hijacked page, so at most a
    couple of attempts are made and anything non-CNF is skipped. Normally
    returns [].
    """
    out: list = []
    base = "https://www.satlib.org/Benchmarks/SAT/DIMACS"
    probe = _get(f"{base}/uf250/uf250-01.cnf", https_only=https_only)
    if probe is None or not verify_cnf_bytes(probe, "random3sat"):
        log.warning("SATLIB host unreachable or untrusted; skipping SATLIB downloads")
        return out
    # Host is reachable and serving real CNFs: grab a bounded few.
    for klass in ("uf250", "uf100-430"):
        for i in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            data = _get(f"{base}/{klass}/{klass}-{i:02d}.cnf", https_only=https_only)
            if data is None or not verify_cnf_bytes(data, "random3sat"):
                continue
            out.append((data, "random3sat", f"{klass}-{i:02d}"))
            if len(out) >= MAX_DOWNLOAD_ATTEMPTS:
                return out
    return out


def download_gbd(family_counts: dict[str, int], out_dir: Path,
                 *, https_only: bool = True) -> list:
    """Bounded GBD probe. GBD is unreachable here; returns []."""
    base = "https://gbd.iti.kit.edu"
    try:
        resp = requests.get(base, timeout=HTTP_TIMEOUT_S)
        if resp.status_code != 200:
            log.warning("GBD unreachable; skipping GBD downloads")
            return []
    except requests.RequestException as exc:
        log.debug("GBD unreachable: %s", exc)
        return []
    return []
