"""Benchmark 1: EpiplexityMonitor vs. repetition counter + timeout.

Tests whether Bayesian two-signal stagnation detection (embedding novelty
+ perplexity) outperforms cosine-similarity repetition counting.

The key hypothesis: EpiplexityMonitor uses TWO independent signals
(novelty and perplexity) to distinguish:
- Convergence (low novelty + low perplexity) → healthy
- Stagnation (low novelty + high perplexity) → pathological

A cosine-similarity detector uses ONE signal (similarity).  It can't
distinguish convergence from stagnation without external information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from operon_ai.health.epiplexity import EpiplexityMonitor, MockEmbeddingProvider

from eval.utils import Counter
from eval.benchmarks.types import BenchmarkResult, TrialResult, Variant
from eval.benchmarks.naive.repetition_counter import RepetitionCounter
from eval.benchmarks.pathways.free_energy import SCENARIOS as FE_SCENARIOS


def _bio_matches_ground_truth(bio_status: str, ground_truth: str) -> bool:
    """Check if biological monitor status matches ground truth."""
    bio_lower = bio_status.lower()
    gt = ground_truth.lower()
    if gt in ("healthy", "exploring"):
        return bio_lower in ("healthy", "exploring", "converging")
    if gt == "converging":
        return bio_lower in ("converging", "healthy")
    if gt in ("stagnant", "critical"):
        return bio_lower in ("stagnant", "critical")
    return bio_lower == gt


def _naive_matches_ground_truth(naive_status: str, ground_truth: str) -> bool:
    """Check if naive detector status matches ground truth."""
    gt = ground_truth.lower()
    if gt in ("healthy", "exploring", "converging"):
        return naive_status == "healthy"
    if gt in ("stagnant", "critical"):
        return naive_status == "stagnant"
    return False


@dataclass
class EpiplexityBenchConfig:
    n_trials: int = 100
    window_size: int = 10
    alpha: float = 0.5
    threshold: float = 0.2
    critical_duration: int = 5
    naive_similarity_threshold: float = 0.85
    naive_timeout_steps: int = 20
    scenarios: list[str] = field(default_factory=lambda: list(FE_SCENARIOS.keys()))


def run_epiplexity_bench(
    config: EpiplexityBenchConfig,
    rng: random.Random,
) -> list[BenchmarkResult]:
    """Run the epiplexity benchmark across all scenarios."""
    provider = MockEmbeddingProvider()
    results: list[BenchmarkResult] = []

    for scenario_name in config.scenarios:
        scenario_fn = FE_SCENARIOS[scenario_name]

        # Counters per variant
        bio_accuracy = Counter("detection_accuracy")
        bio_false_pos = Counter("false_positive")
        bio_false_neg = Counter("false_negative")
        bio_converge_disc = Counter("convergence_discrimination")

        abl_accuracy = Counter("detection_accuracy")
        abl_false_pos = Counter("false_positive")
        abl_false_neg = Counter("false_negative")

        naive_accuracy = Counter("detection_accuracy")
        naive_false_pos = Counter("false_positive")
        naive_false_neg = Counter("false_negative")
        naive_converge_disc = Counter("convergence_discrimination")

        bio_latencies: list[int] = []
        naive_latencies: list[int] = []

        for trial in range(config.n_trials):
            trial_rng = random.Random(rng.randint(0, 2**32) + trial)
            steps = scenario_fn(trial_rng)

            # --- Biological variant ---
            monitor = EpiplexityMonitor(
                embedding_provider=provider,
                alpha=config.alpha,
                window_size=config.window_size,
                threshold=config.threshold,
                critical_duration=config.critical_duration,
            )
            bio_first_detection: int | None = None
            for i, step in enumerate(steps):
                result = monitor.measure(step.message, perplexity=step.perplexity)
                bio_status = result.status.value
                gt = step.ground_truth_status or "healthy"

                correct = _bio_matches_ground_truth(bio_status, gt)
                bio_accuracy.record(correct)

                is_alert = bio_status in ("stagnant", "critical")
                gt_alert = gt in ("stagnant", "critical")
                bio_false_pos.record(not (is_alert and not gt_alert))  # success = NOT false pos
                bio_false_neg.record(not (not is_alert and gt_alert))

                if gt == "converging":
                    bio_converge_disc.record(bio_status in ("converging", "healthy"))

                if gt_alert and is_alert and bio_first_detection is None:
                    bio_first_detection = i

            if bio_first_detection is not None:
                # Find first ground-truth stagnant step
                first_gt = next(
                    (i for i, s in enumerate(steps) if s.ground_truth_status in ("stagnant", "critical")),
                    None,
                )
                if first_gt is not None:
                    bio_latencies.append(bio_first_detection - first_gt)

            # --- Ablated variant (no monitor) ---
            for step in steps:
                gt = step.ground_truth_status or "healthy"
                # Always reports healthy
                correct = gt in ("healthy", "exploring", "converging")
                abl_accuracy.record(correct)
                gt_alert = gt in ("stagnant", "critical")
                abl_false_pos.record(True)  # Never alerts, so never false positive
                abl_false_neg.record(not gt_alert)  # Misses all stagnation

            # --- Naive variant ---
            counter = RepetitionCounter(
                embedding_provider=provider,
                window_size=config.window_size,
                similarity_threshold=config.naive_similarity_threshold,
                timeout_steps=config.naive_timeout_steps,
            )
            naive_first_detection: int | None = None
            for i, step in enumerate(steps):
                result = counter.measure(step.message)
                gt = step.ground_truth_status or "healthy"

                correct = _naive_matches_ground_truth(result.status, gt)
                naive_accuracy.record(correct)

                is_alert = result.status == "stagnant"
                gt_alert = gt in ("stagnant", "critical")
                naive_false_pos.record(not (is_alert and not gt_alert))
                naive_false_neg.record(not (not is_alert and gt_alert))

                if gt == "converging":
                    naive_converge_disc.record(result.status == "healthy")

                if gt_alert and is_alert and naive_first_detection is None:
                    naive_first_detection = i

            if naive_first_detection is not None:
                first_gt = next(
                    (i for i, s in enumerate(steps) if s.ground_truth_status in ("stagnant", "critical")),
                    None,
                )
                if first_gt is not None:
                    naive_latencies.append(naive_first_detection - first_gt)

        # Build result
        def _counter_to_trial(c: Counter, variant: Variant, latency: float | None = None) -> TrialResult:
            return TrialResult(
                variant=variant,
                metric_name=c.name,
                success=c.success,
                total=c.total,
                latency_steps=latency,
            )

        bio_avg_latency = sum(bio_latencies) / len(bio_latencies) if bio_latencies else None
        naive_avg_latency = sum(naive_latencies) / len(naive_latencies) if naive_latencies else None

        br = BenchmarkResult(
            name="epiplexity",
            scenario_name=scenario_name,
            seed=rng.randint(0, 2**32),
            config={
                "n_trials": config.n_trials,
                "alpha": config.alpha,
                "threshold": config.threshold,
            },
        )
        br.variants = {
            Variant.BIOLOGICAL: {
                c.name: _counter_to_trial(c, Variant.BIOLOGICAL)
                for c in [bio_accuracy, bio_false_pos, bio_false_neg, bio_converge_disc]
                if c.total > 0
            },
            Variant.ABLATED: {
                c.name: _counter_to_trial(c, Variant.ABLATED)
                for c in [abl_accuracy, abl_false_pos, abl_false_neg]
                if c.total > 0
            },
            Variant.NAIVE: {
                c.name: _counter_to_trial(c, Variant.NAIVE)
                for c in [naive_accuracy, naive_false_pos, naive_false_neg, naive_converge_disc]
                if c.total > 0
            },
        }
        # Inject latencies into accuracy results
        if bio_avg_latency is not None and "detection_accuracy" in br.variants[Variant.BIOLOGICAL]:
            br.variants[Variant.BIOLOGICAL]["detection_accuracy"].latency_steps = bio_avg_latency
        if naive_avg_latency is not None and "detection_accuracy" in br.variants[Variant.NAIVE]:
            br.variants[Variant.NAIVE]["detection_accuracy"].latency_steps = naive_avg_latency

        results.append(br)

    return results
