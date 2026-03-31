"""Metrics collection and comparison for the C6 convergence evaluation harness.

Provides per-run metrics, aggregation across tasks within a configuration,
and cross-configuration comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunMetrics:
    """Metrics from a single task x configuration evaluation run."""

    task_id: str
    config_id: str
    success: bool
    token_cost: int
    latency_ms: float
    intervention_count: int
    convergence_rate: float
    structural_variation: float
    risk_score: float
    stage_count: int


@dataclass
class AggregateMetrics:
    """Aggregated metrics for a single configuration across all tasks."""

    config_id: str
    success_rate: float
    mean_token_cost: float
    mean_latency_ms: float
    mean_interventions: float
    mean_risk_score: float
    n_tasks: int


def collect_metrics(runs: list[RunMetrics]) -> dict[str, AggregateMetrics]:
    """Group runs by config_id and compute aggregate metrics.

    Returns a dict mapping config_id to AggregateMetrics.
    """
    by_config: dict[str, list[RunMetrics]] = {}
    for run in runs:
        by_config.setdefault(run.config_id, []).append(run)

    aggregates: dict[str, AggregateMetrics] = {}
    for config_id, config_runs in by_config.items():
        n = len(config_runs)
        successes = sum(1 for r in config_runs if r.success)
        aggregates[config_id] = AggregateMetrics(
            config_id=config_id,
            success_rate=successes / n if n > 0 else 0.0,
            mean_token_cost=sum(r.token_cost for r in config_runs) / n if n > 0 else 0.0,
            mean_latency_ms=sum(r.latency_ms for r in config_runs) / n if n > 0 else 0.0,
            mean_interventions=sum(r.intervention_count for r in config_runs) / n if n > 0 else 0.0,
            mean_risk_score=sum(r.risk_score for r in config_runs) / n if n > 0 else 0.0,
            n_tasks=n,
        )

    return aggregates


def compare_configs(aggregates: dict[str, AggregateMetrics]) -> list[dict]:
    """Produce a ranked comparison of configurations.

    Returns a list of dicts sorted by success_rate (desc), then mean_risk_score (asc).
    Each dict contains the config_id and all aggregate metric values.
    """
    rows: list[dict] = []
    for agg in aggregates.values():
        rows.append({
            "config_id": agg.config_id,
            "success_rate": round(agg.success_rate, 4),
            "mean_token_cost": round(agg.mean_token_cost, 1),
            "mean_latency_ms": round(agg.mean_latency_ms, 1),
            "mean_interventions": round(agg.mean_interventions, 2),
            "mean_risk_score": round(agg.mean_risk_score, 4),
            "n_tasks": agg.n_tasks,
        })

    # Sort: highest success rate first, lowest risk second.
    rows.sort(key=lambda r: (-r["success_rate"], r["mean_risk_score"]))
    return rows
