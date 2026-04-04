"""Shared types for the biological benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from eval.utils import wilson_interval


class Variant(Enum):
    """Experimental condition."""

    BIOLOGICAL = "biological"  # Operon feature enabled
    ABLATED = "ablated"  # Feature disabled entirely
    NAIVE = "naive"  # Simple non-biological alternative


@dataclass
class TrialResult:
    """Result of a single metric within one trial."""

    variant: Variant
    metric_name: str
    success: int
    total: int
    latency_steps: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0.0

    @property
    def wilson_95(self) -> tuple[float, float]:
        return wilson_interval(self.success, self.total)


@dataclass
class BenchmarkResult:
    """Full result from one benchmark run (one scenario, one seed)."""

    name: str
    scenario_name: str
    seed: int
    config: dict
    # variant -> metric_name -> TrialResult
    variants: dict[Variant, dict[str, TrialResult]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        out: dict = {
            "name": self.name,
            "scenario": self.scenario_name,
            "seed": self.seed,
        }
        for variant, metrics in self.variants.items():
            out[variant.value] = {
                name: {
                    "success": tr.success,
                    "total": tr.total,
                    "rate": tr.rate,
                    "wilson_95": list(tr.wilson_95),
                    **({"latency_steps": tr.latency_steps} if tr.latency_steps is not None else {}),
                }
                for name, tr in metrics.items()
            }
        return out


@dataclass
class ComparisonRow:
    """One row in the comparison table: biological vs naive on a single metric."""

    metric: str
    scenario: str
    biological_rate: float
    biological_ci: tuple[float, float]
    ablated_rate: float
    ablated_ci: tuple[float, float]
    naive_rate: float
    naive_ci: tuple[float, float]
    delta_bio_vs_naive: float
    n: int

    @property
    def significant(self) -> bool:
        """CIs don't overlap => statistically significant difference."""
        return self.biological_ci[0] > self.naive_ci[1] or self.naive_ci[0] > self.biological_ci[1]


def build_comparison_table(results: list[BenchmarkResult]) -> list[ComparisonRow]:
    """Aggregate results into comparison rows.

    Groups by (scenario, metric), pools success/total across seeds,
    then computes rates and Wilson CIs for each variant.
    """
    # (scenario, metric) -> variant -> (total_success, total_total)
    pool: dict[tuple[str, str], dict[Variant, list[int]]] = {}

    for result in results:
        for variant, metrics in result.variants.items():
            for metric_name, tr in metrics.items():
                key = (result.scenario_name, metric_name)
                if key not in pool:
                    pool[key] = {}
                if variant not in pool[key]:
                    pool[key][variant] = [0, 0]
                pool[key][variant][0] += tr.success
                pool[key][variant][1] += tr.total

    rows: list[ComparisonRow] = []
    for (scenario, metric), variant_data in sorted(pool.items()):
        bio = variant_data.get(Variant.BIOLOGICAL, [0, 0])
        abl = variant_data.get(Variant.ABLATED, [0, 0])
        nai = variant_data.get(Variant.NAIVE, [0, 0])

        bio_rate = bio[0] / bio[1] if bio[1] > 0 else 0.0
        abl_rate = abl[0] / abl[1] if abl[1] > 0 else 0.0
        nai_rate = nai[0] / nai[1] if nai[1] > 0 else 0.0

        rows.append(ComparisonRow(
            metric=metric,
            scenario=scenario,
            biological_rate=bio_rate,
            biological_ci=wilson_interval(bio[0], bio[1]),
            ablated_rate=abl_rate,
            ablated_ci=wilson_interval(abl[0], abl[1]),
            naive_rate=nai_rate,
            naive_ci=wilson_interval(nai[0], nai[1]),
            delta_bio_vs_naive=bio_rate - nai_rate,
            n=bio[1],
        ))

    return rows
