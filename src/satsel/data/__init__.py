"""Dataset stage: derive per-formula labels from the profiling DB, dedup, and
produce cluster-safe train/val/test splits with OOD held out."""

from satsel.data import build, labels  # noqa: F401

__all__ = ["build", "labels"]
