"""Subprocess helpers.

SAT solvers are native C binaries linked into python-sat. A Python-level
signal cannot interrupt a blocked C solver, so every solver run (profiling and
oracle) executes in a child process started in its own session and, on timeout,
the whole process group is killed with SIGKILL.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class ProcResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    wall_s: float
    timed_out: bool


def run_subprocess(
    argv: list[str],
    *,
    timeout_s: float,
    input_bytes: bytes | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> ProcResult:
    """Run a command, kill the whole process group on timeout.

    Returns a ProcResult. `timed_out=True` means we killed it; the returncode
    is then -9.
    """
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if input_bytes is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=full_env,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        return ProcResult(returncode=-2, stdout=b"", stderr=str(exc).encode(),
                           wall_s=0.0, timed_out=False)

    timed_out = False
    try:
        stdout, stderr = proc.communicate(input=input_bytes, timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
    finally:
        wall = time.monotonic() - start

    return ProcResult(
        returncode=returncode,
        stdout=stdout or b"",
        stderr=stderr or b"",
        wall_s=wall,
        timed_out=timed_out,
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group so grandchildren die too."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def current_python() -> str:
    return sys.executable
