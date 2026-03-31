"""Structural variation analysis for the C6 convergence evaluation harness.

Measures how far a configuration's realized topology deviates from the
task's intended shape, providing a distance metric and summary statistics.
"""

from __future__ import annotations

from .metrics import RunMetrics


# Canonical ordering for distance computation.
_SHAPE_INDEX: dict[str, int] = {
    "sequential": 0,
    "mixed": 1,
    "parallel": 2,
}


def topology_distance(task_shape: str, config_recommendation: str) -> float:
    """Compute a normalized distance between two topology shapes.

    Returns a float in [0.0, 1.0]:
      - 0.0 if shapes match exactly
      - 0.5 if adjacent (sequential <-> mixed, mixed <-> parallel)
      - 1.0 if maximally divergent (sequential <-> parallel)
    """
    task_idx = _SHAPE_INDEX.get(task_shape, 1)
    rec_idx = _SHAPE_INDEX.get(config_recommendation, 1)
    diff = abs(task_idx - rec_idx)
    # Max possible distance is 2 (sequential <-> parallel).
    return diff / 2.0


def variation_summary(runs: list[RunMetrics]) -> dict[str, float]:
    """Compute per-config mean structural variation from a list of runs.

    Returns a dict mapping config_id to mean structural_variation.
    """
    by_config: dict[str, list[float]] = {}
    for run in runs:
        by_config.setdefault(run.config_id, []).append(run.structural_variation)

    return {
        config_id: sum(vals) / len(vals) if vals else 0.0
        for config_id, vals in by_config.items()
    }
