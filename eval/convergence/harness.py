"""Convergence evaluation harness for the C6 evaluation.

Orchestrates a full evaluation run: selects tasks and configurations,
runs the mock evaluator, collects metrics, and returns structured results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .configurations import ConfigurationSpec, get_configurations
from .credit_assignment import aggregate_credit, assign_credit
from .metrics import RunMetrics, collect_metrics, compare_configs
from .mock_evaluator import MockEvaluator
from .structural_variation import variation_summary
from .tasks import TaskDefinition, get_benchmark_tasks


@dataclass
class HarnessConfig:
    """Configuration for a convergence evaluation run."""

    seed: int = 1337
    tasks: list[str] | None = None
    configs: list[str] | None = None
    mode: str = "mock"


class ConvergenceHarness:
    """Orchestrates a convergence evaluation run.

    Creates a MockEvaluator, iterates over all selected task x config pairs,
    collects metrics, and produces a structured result dict.
    """

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config
        self._evaluator = MockEvaluator(seed=config.seed)

        # Select tasks.
        all_tasks = get_benchmark_tasks()
        if config.tasks is not None:
            self._tasks = [t for t in all_tasks if t.task_id in config.tasks]
        else:
            self._tasks = all_tasks

        # Select configurations.
        all_configs = get_configurations()
        if config.configs is not None:
            self._configs = [c for c in all_configs if c.config_id in config.configs]
        else:
            self._configs = all_configs

    @property
    def tasks(self) -> list[TaskDefinition]:
        """Return selected tasks."""
        return list(self._tasks)

    @property
    def configurations(self) -> list[ConfigurationSpec]:
        """Return selected configurations."""
        return list(self._configs)

    def run(self) -> dict[str, Any]:
        """Execute the full evaluation and return structured results.

        Returns a dict with:
          - seed: the RNG seed used
          - runs: list of per-run metric dicts
          - aggregates: per-config aggregate metrics
          - comparison: ranked comparison table
          - variation: per-config structural variation summary
          - credit: per-role credit assignment summary
          - n_tasks: number of tasks evaluated
          - n_configs: number of configurations evaluated
        """
        runs: list[RunMetrics] = []
        all_credits: list[list] = []

        for task in self._tasks:
            for config in self._configs:
                metrics = self._evaluator.evaluate(task, config)
                runs.append(metrics)

                # Credit assignment for this run.
                import random
                pair_seed = hash((self._config.seed, task.task_id, config.config_id)) & 0xFFFFFFFF
                rng = random.Random(pair_seed)
                credits = assign_credit(
                    roles=task.required_roles,
                    success=metrics.success,
                    risk_score=metrics.risk_score,
                    rng=rng,
                )
                all_credits.append(credits)

        # Collect and compare.
        aggregates = collect_metrics(runs)
        comparison = compare_configs(aggregates)
        variation = variation_summary(runs)
        credit = aggregate_credit(all_credits)

        # Serialize runs.
        run_dicts = []
        for r in runs:
            run_dicts.append({
                "task_id": r.task_id,
                "config_id": r.config_id,
                "success": r.success,
                "token_cost": r.token_cost,
                "latency_ms": r.latency_ms,
                "intervention_count": r.intervention_count,
                "convergence_rate": r.convergence_rate,
                "structural_variation": r.structural_variation,
                "risk_score": r.risk_score,
                "stage_count": r.stage_count,
            })

        # Serialize aggregates.
        agg_dicts = {}
        for config_id, agg in aggregates.items():
            agg_dicts[config_id] = {
                "config_id": agg.config_id,
                "success_rate": agg.success_rate,
                "mean_token_cost": agg.mean_token_cost,
                "mean_latency_ms": agg.mean_latency_ms,
                "mean_interventions": agg.mean_interventions,
                "mean_risk_score": agg.mean_risk_score,
                "n_tasks": agg.n_tasks,
            }

        return {
            "seed": self._config.seed,
            "runs": run_dicts,
            "aggregates": agg_dicts,
            "comparison": comparison,
            "variation": variation,
            "credit": credit,
            "n_tasks": len(self._tasks),
            "n_configs": len(self._configs),
        }
