"""Bench stage: acquire formulas (download with fallback to generators), write
them to disk, and emit a manifest consumed by later stages."""

from satsel.bench import pipeline  # noqa: F401

__all__ = ["pipeline"]
