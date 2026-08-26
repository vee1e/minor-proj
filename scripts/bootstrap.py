#!/usr/bin/env python3
"""One-command environment setup for satsel. Standard library only.

Run this first on any new machine:

    python scripts/bootstrap.py              # solvers + classical ML + dev tools
    python scripts/bootstrap.py --gnn        # ... plus PyTorch (CPU build)
    python scripts/bootstrap.py --gnn --cuda # ... plus PyTorch with CUDA
    python scripts/bootstrap.py --check      # verify an existing setup only

Why this exists: `uv sync` alone is not enough to get a working checkout, and
every failure mode below has actually happened on one of our machines.

  1. `uv` installs to a directory that is not on PATH (notably
     %APPDATA%\\Python\\Python3xx\\Scripts on Windows), so `uv` is "not found"
     even though it is installed. We search the known locations.
  2. The project needs Python 3.11, but the system Python is often newer.
     `.python-version` pins it and `uv python install` fetches it.
  3. On Windows, `pip install torch` from PyPI gives a **CPU-only** build with
     no error and no warning. You discover this when training silently runs on
     the CPU. Getting CUDA requires PyTorch's own package index.
  4. torch-geometric must be built against the same torch as everything else.

The --check pass verifies imports and prints what it found, including whether
CUDA is actually usable and whether the GPU's compute capability is supported
by the installed torch. Exit code is non-zero if anything required is missing,
so this is safe to use as a CI or pre-flight gate.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_VERSION = "3.11"

# PyTorch wheel index. cu128 carries kernels for Blackwell (compute capability
# 12.0, e.g. RTX PRO 6000 / B200); older cu12x builds do not and will fail at
# runtime with "no kernel image is available for execution on the device".
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

REQUIRED = ["pysat", "numpy", "pandas", "sklearn", "scipy", "xgboost", "yaml",
            "joblib", "requests", "flask"]
OPTIONAL_GNN = ["torch", "torch_geometric"]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    GREEN = RED = YELLOW = DIM = RESET = ""


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}    {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}", flush=True)


def find_uv() -> str | None:
    """Locate the uv executable, including places installers use but PATH misses."""
    found = shutil.which("uv")
    if found:
        return found

    candidates: list[Path] = []
    home = Path.home()
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            # pip install --user puts it under the interpreter's version dir.
            candidates += sorted(Path(appdata).glob("Python/Python3*/Scripts/uv.exe"))
        localapp = os.environ.get("LOCALAPPDATA")
        if localapp:
            candidates.append(Path(localapp) / "Programs/uv/uv.exe")
            candidates += sorted(Path(localapp).glob("uv/uv.exe"))
        candidates.append(home / ".local/bin/uv.exe")
        candidates.append(home / ".cargo/bin/uv.exe")
    else:
        candidates += [home / ".local/bin/uv", home / ".cargo/bin/uv",
                       Path("/usr/local/bin/uv"), Path("/opt/homebrew/bin/uv")]

    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def install_uv() -> str | None:
    """Install uv via pip into the current interpreter, then re-locate it."""
    say("uv not found; installing it with pip ...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--upgrade", "uv"], check=True)
    except subprocess.CalledProcessError as exc:
        fail(f"could not pip install uv: {exc}")
        return None
    return find_uv()


def run(uv: str, args: list[str], *, check: bool = True) -> int:
    printable = " ".join(["uv", *args])
    say(f"{DIM}$ {printable}{RESET}")
    proc = subprocess.run([uv, *args], cwd=str(REPO_ROOT))
    if check and proc.returncode != 0:
        fail(f"command failed ({proc.returncode}): {printable}")
        sys.exit(proc.returncode)
    return proc.returncode


def probe(uv: str, want_gnn: bool) -> int:
    """Import every dependency inside the project env and report the result."""
    template = """
import importlib, sys, platform
required = __REQUIRED__
optional = __OPTIONAL__
missing = []
print("  python      {} ({})".format(sys.version.split()[0], platform.machine()))
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)[:70]))
if missing:
    for name, why in missing:
        print("  MISSING     {}  -- {}".format(name, why))
else:
    print("  deps        all {} core packages import cleanly".format(len(required)))

torch_ok = bool(optional)
for name in optional:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print("  gnn extra   {} unavailable -- {}".format(name, str(exc)[:60]))
        torch_ok = False
        break

if torch_ok:
    import torch
    print("  torch       {}".format(torch.__version__))
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        tag = "sm_{}{}".format(props.major, props.minor)
        print("  CUDA        YES -- {} ({}, {:.0f} GB)".format(
            props.name, tag, props.total_memory / 1e9))
        arches = list(torch.cuda.get_arch_list())
        if tag not in arches:
            print("  CUDA        WARNING: this torch build has no kernels for " + tag)
            print("              built for: " + ", ".join(arches))
            print("              fix: python scripts/bootstrap.py --gnn --cuda")
    else:
        print("  CUDA        no -- training will run on CPU")
        if "+cpu" in torch.__version__:
            print("              CPU-only wheel; for GPU run --gnn --cuda")
sys.exit(1 if missing else 0)
"""
    code = (template
            .replace("__REQUIRED__", repr(REQUIRED))
            .replace("__OPTIONAL__", repr(OPTIONAL_GNN if want_gnn else [])))
    proc = subprocess.run([uv, "run", "python", "-c", code], cwd=str(REPO_ROOT))
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Set up the satsel environment reproducibly.")
    ap.add_argument("--gnn", action="store_true",
                    help="also install PyTorch + PyTorch Geometric")
    ap.add_argument("--cuda", action="store_true",
                    help="install the CUDA build of PyTorch (implies --gnn)")
    ap.add_argument("--check", action="store_true",
                    help="verify an existing environment without installing")
    ap.add_argument("--cuda-index", default=CUDA_INDEX,
                    help=f"PyTorch wheel index for --cuda (default {CUDA_INDEX})")
    args = ap.parse_args()
    want_gnn = args.gnn or args.cuda

    print()
    print(f"satsel bootstrap  --  {platform.system()} {platform.release()}, "
          f"{os.cpu_count()} cores")
    print(f"repo: {REPO_ROOT}")
    print()

    uv = find_uv()
    if uv is None and not args.check:
        uv = install_uv()
    if uv is None:
        fail("uv is not installed and could not be installed automatically.")
        say("Install it manually, then re-run this script:")
        say("  Windows : winget install astral-sh.uv")
        say("  macOS   : brew install uv")
        say("  Linux   : curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1
    ok(f"uv at {uv}")

    if not (REPO_ROOT / ".python-version").exists():
        (REPO_ROOT / ".python-version").write_text(PY_VERSION + "\n")
        say(f"wrote .python-version ({PY_VERSION})")

    if not args.check:
        run(uv, ["python", "install", PY_VERSION])
        sync = ["sync", "--extra", "dev"]
        if want_gnn:
            sync += ["--extra", "gnn"]
        run(uv, sync)

        if args.cuda:
            print()
            say("installing the CUDA build of PyTorch ...")
            say(f"{DIM}(PyPI torch on Windows is CPU-only; this replaces it){RESET}")
            rc = run(uv, ["pip", "install", "--upgrade", "torch",
                          "--index-url", args.cuda_index], check=False)
            if rc != 0:
                warn("CUDA torch install failed. The CPU build is still usable.")
                warn("Try a different index, e.g. --cuda-index "
                     "https://download.pytorch.org/whl/cu126")

    print()
    print("verifying:")
    rc = probe(uv, want_gnn)

    print()
    if rc == 0:
        ok("environment ready")
        say(f"{DIM}run the pipeline with:  uv run satsel --help{RESET}")
    else:
        fail("environment incomplete -- see MISSING lines above")
    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
