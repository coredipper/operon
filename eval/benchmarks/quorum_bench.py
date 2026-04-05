"""Benchmark 2: Quorum sensing (signal accumulation) vs. voting strategies.

Tests whether biologically-faithful autoinducer accumulation with
temporal decay outperforms standard coordination approaches for
multi-agent threat consensus.

The key hypothesis: signal accumulation + decay provides:
1. Noise averaging (continuous signals, not binary votes)
2. Temporal relevance (old evidence decays, recent evidence dominates)
3. Proportional weighting (more suspicious agents contribute more signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from operon_ai.coordination.quorum_sensing import QuorumSensingBio, SignalEnvironment

from eval.utils import Counter
from eval.benchmarks.types import BenchmarkResult, TrialResult, Variant
from eval.benchmarks.naive.simple_vote import (
    IndependentActors,
    MajorityVote,
)
from eval.benchmarks.pathways.quorum_sensing import (
    generate_gradual_infiltration,
    generate_noisy_environment,
    generate_static_compromise,
)


@dataclass
class QuorumBenchConfig:
    n_agents_range: list[int] = field(default_factory=lambda: [5, 10, 20])
    compromised_fractions: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4])
    n_trials: int = 50
    time_steps: int = 30
    noise_std: float = 0.15
    ahl_decay_half_life: float = 5.0
    expected_normal_suspicion: float = 0.15
    safety_margin: float = 2.0
    vote_threshold: float = 0.5
    majority_fraction: float = 0.5
    scenarios: list[str] = field(
        default_factory=lambda: ["static", "gradual_infiltration", "noisy"],
    )


def _run_scenario(
    scenario_name: str,
    n_agents: int,
    n_compromised: int,
    config: QuorumBenchConfig,
    rng: random.Random,
) -> BenchmarkResult:
    """Run one scenario configuration across all trials."""
    bio_tp = Counter("true_positive")
    bio_fp = Counter("false_positive")
    abl_tp = Counter("true_positive")
    abl_fp = Counter("false_positive")
    naive_tp = Counter("true_positive")
    naive_fp = Counter("false_positive")

    bio_latencies: list[int] = []
    naive_latencies: list[int] = []

    for trial in range(config.n_trials):
        trial_rng = random.Random(rng.randint(0, 2**32) + trial)

        # Generate scenario
        if scenario_name == "static":
            steps = generate_static_compromise(
                n_agents, n_compromised, config.time_steps, config.noise_std, trial_rng,
            )
        elif scenario_name == "gradual_infiltration":
            steps = generate_gradual_infiltration(
                n_agents, n_compromised, config.time_steps, config.noise_std, trial_rng,
            )
        elif scenario_name == "noisy":
            steps = generate_noisy_environment(
                n_agents, config.time_steps, config.noise_std, trial_rng,
            )
        else:
            continue

        # --- Biological variant: QuorumSensingBio (calibrated) ---
        qs = QuorumSensingBio(
            population_size=n_agents,
            environment=SignalEnvironment(decay_half_life=config.ahl_decay_half_life),
            expected_normal_suspicion=config.expected_normal_suspicion,
            safety_margin=config.safety_margin,
        )
        qs.calibrate()  # Auto-derive threshold from population + signal params
        bio_first_correct: int | None = None
        for i, step in enumerate(steps):
            t = step["time"]
            for aid, susp in step["suspicions"].items():
                qs.produce_signal(aid, susp, t)
            activated = qs.should_activate("AI-1", t)
            gt = step["ground_truth_compromised"]

            if gt:
                bio_tp.record(activated)
                if activated and bio_first_correct is None:
                    bio_first_correct = i
            else:
                bio_fp.record(activated)  # success = activated when shouldn't

        if bio_first_correct is not None:
            first_gt = next((i for i, s in enumerate(steps) if s["ground_truth_compromised"]), None)
            if first_gt is not None:
                bio_latencies.append(bio_first_correct - first_gt)

        # --- Ablated variant: independent actors (no coordination) ---
        indep = IndependentActors(threshold=config.vote_threshold)
        for step in steps:
            activated = indep.should_activate(step["suspicions"])
            gt = step["ground_truth_compromised"]
            if gt:
                abl_tp.record(activated)
            else:
                abl_fp.record(activated)

        # --- Naive variant: majority vote ---
        majority = MajorityVote(
            vote_threshold=config.vote_threshold,
            majority_fraction=config.majority_fraction,
        )
        naive_first_correct: int | None = None
        for i, step in enumerate(steps):
            activated = majority.should_activate(step["suspicions"])
            gt = step["ground_truth_compromised"]
            if gt:
                naive_tp.record(activated)
                if activated and naive_first_correct is None:
                    naive_first_correct = i
            else:
                naive_fp.record(activated)

        if naive_first_correct is not None:
            first_gt = next((i for i, s in enumerate(steps) if s["ground_truth_compromised"]), None)
            if first_gt is not None:
                naive_latencies.append(naive_first_correct - first_gt)

    def _counter_to_trial(c: Counter, variant: Variant) -> TrialResult:
        return TrialResult(variant=variant, metric_name=c.name, success=c.success, total=c.total)

    label = f"{scenario_name}_n{n_agents}_c{n_compromised}"
    br = BenchmarkResult(
        name="quorum_sensing",
        scenario_name=label,
        seed=rng.randint(0, 2**32),
        config={"n_agents": n_agents, "n_compromised": n_compromised, "scenario": scenario_name},
    )

    # For false_positive counter: "success" means it DID false-alarm, so invert for the rate
    # We want false_positive_rate = times_it_wrongly_alerted / total_non_compromise_steps
    br.variants = {
        Variant.BIOLOGICAL: {
            c.name: _counter_to_trial(c, Variant.BIOLOGICAL)
            for c in [bio_tp, bio_fp] if c.total > 0
        },
        Variant.ABLATED: {
            c.name: _counter_to_trial(c, Variant.ABLATED)
            for c in [abl_tp, abl_fp] if c.total > 0
        },
        Variant.NAIVE: {
            c.name: _counter_to_trial(c, Variant.NAIVE)
            for c in [naive_tp, naive_fp] if c.total > 0
        },
    }

    return br


def run_quorum_bench(
    config: QuorumBenchConfig,
    rng: random.Random,
) -> list[BenchmarkResult]:
    """Run the quorum sensing benchmark across all configurations."""
    results: list[BenchmarkResult] = []

    for scenario_name in config.scenarios:
        for n_agents in config.n_agents_range:
            for frac in config.compromised_fractions:
                n_compromised = int(n_agents * frac)
                # Skip noisy scenario with compromised agents (it's a clean-only test)
                if scenario_name == "noisy" and n_compromised > 0:
                    continue
                br = _run_scenario(scenario_name, n_agents, n_compromised, config, rng)
                results.append(br)

    return results
