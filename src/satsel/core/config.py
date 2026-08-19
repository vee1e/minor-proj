"""Configuration loading for config/default.yaml.

The config is frozen before profiling (M3) per EVAL_PROTOCOL.md. This module
only reads it; nothing in the codebase mutates it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the frozen YAML config and resolve repo-relative paths."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with path.open() as fh:
        cfg = yaml.safe_load(fh)

    paths = cfg.get("paths", {})
    resolved = {}
    for key, value in paths.items():
        resolved[key] = (REPO_ROOT / value).resolve() if isinstance(value, str) else value
    cfg["paths"] = resolved
    return cfg


def tau_for_family(cfg: dict[str, Any], family: str) -> float:
    """Per-family timeout tier; falls back to max_timeout for unknown families."""
    return float(cfg["tau_tiers"].get(family, cfg.get("max_timeout", 300)))


def feat_budget_s(cfg: dict[str, Any], mean_best_solver_median: float) -> float:
    """tau_feat = max(feat_budget_frac * mean_best_solver_median, feat_budget_floor_s)."""
    frac = float(cfg.get("feat_budget_frac", 0.05))
    floor = float(cfg.get("feat_budget_floor_s", 0.20))
    return max(frac * mean_best_solver_median, floor)
