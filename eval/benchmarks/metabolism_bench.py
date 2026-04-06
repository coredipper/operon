"""Benchmark 3: ATP_Store + mTOR scaling vs. simple budget counter.

Tests whether the full metabolic state machine (multiple currencies,
debt, regeneration, adaptive scaling) outperforms a flat budget counter
under realistic load patterns.

The key hypothesis: the biological system provides:
1. Graceful degradation (priority gating in STARVING state)
2. Anticipatory conservation (rate-sensitive mTOR scaling)
3. Recovery capability (regeneration, NADH conversion, debt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from operon_ai.state.metabolism import ATP_Store, MetabolicState
from operon_ai.state.mtor import MTORScaler

from eval.utils import Counter
from eval.benchmarks.types import BenchmarkResult, TrialResult, Variant
from eval.benchmarks.naive.simple_budget import SimpleBudget
from eval.benchmarks.pathways.ampk import (
    Operation,
    generate_bursty_load,
    generate_constant_load,
    generate_gradual_depletion,
    generate_mixed_priority,
    generate_sudden_spike,
)


@dataclass
class MetabolismBenchConfig:
    initial_budget: int = 1000
    gtp_budget: int = 100
    nadh_reserve: int = 200
    regeneration_per_step: int = 5
    max_debt: int = 100
    n_trials: int = 50
    min_workers: int = 1
    max_workers: int = 8
    scenarios: list[str] = field(
        default_factory=lambda: [
            "constant", "bursty", "gradual_depletion", "sudden_spike",
            "mixed_priority", "worker_scaling",
        ],
    )


def _generate_ops(
    scenario_name: str,
    config: MetabolismBenchConfig,
    rng: random.Random,
) -> list[Operation]:
    """Generate operation sequence for a scenario."""
    budget = config.initial_budget
    length = 200

    if scenario_name == "constant":
        return generate_constant_load(budget, cost_per_op=5, length=length, rng=rng)
    elif scenario_name == "bursty":
        return generate_bursty_load(budget, burst_size=50, burst_interval=20, length=length, rng=rng)
    elif scenario_name == "gradual_depletion":
        return generate_gradual_depletion(budget, length=length, rng=rng)
    elif scenario_name == "sudden_spike":
        return generate_sudden_spike(budget, spike_at=100, spike_cost=budget // 2, length=length, rng=rng)
    elif scenario_name == "mixed_priority":
        return generate_mixed_priority(budget, length=length, rng=rng)
    elif scenario_name == "worker_scaling":
        # Large pool of cheap ops — runner draws batches of recommended_workers()
        # per step over a fixed number of steps. Budget depletes from batch processing.
        return [Operation(cost=3, name=f"task_{i}", priority=rng.randint(0, 5))
                for i in range(2000)]
    else:
        return generate_constant_load(budget, cost_per_op=5, length=length, rng=rng)


def run_metabolism_bench(
    config: MetabolismBenchConfig,
    rng: random.Random,
) -> list[BenchmarkResult]:
    """Run the metabolism benchmark across all scenarios."""
    results: list[BenchmarkResult] = []

    for scenario_name in config.scenarios:
        bio_completed = Counter("operations_completed")
        bio_critical_served = Counter("critical_served_under_pressure")

        abl_completed = Counter("operations_completed")
        abl_critical_served = Counter("critical_served_under_pressure")

        naive_completed = Counter("operations_completed")
        naive_critical_served = Counter("critical_served_under_pressure")

        bio_transitions: list[int] = []
        bio_states_visited: list[set[str]] = []

        for trial in range(config.n_trials):
            trial_rng = random.Random(rng.randint(0, 2**32) + trial)
            ops = _generate_ops(scenario_name, config, trial_rng)

            # --- Biological variant: ATP_Store + MTORScaler ---
            store = ATP_Store(
                budget=config.initial_budget,
                gtp_budget=config.gtp_budget,
                nadh_reserve=config.nadh_reserve,
                max_debt=config.max_debt,
                silent=True,
            )
            scaler = MTORScaler(
                atp_store=store,
                min_workers=config.min_workers,
                max_workers=config.max_workers,
            )

            trial_states: set[str] = set()
            is_scaling = scenario_name == "worker_scaling"
            op_idx = 0

            if is_scaling:
                # Worker scaling: fixed 100 steps, each processes
                # recommended_workers() ops. Bio adapts batch size to
                # resource level; ablated uses fixed count.
                max_steps = 100
                for step in range(max_steps):
                    if step > 0 and step % 10 == 0:
                        store.regenerate(config.regeneration_per_step)
                    scaler.update()
                    trial_states.add(scaler.state.value)
                    batch_size = scaler.recommended_workers()
                    for _ in range(batch_size):
                        if op_idx >= len(ops):
                            break
                        op = ops[op_idx]
                        op_idx += 1
                        if not scaler.should_enable_feature(op.cost) and op.priority < 5:
                            bio_completed.record(False)
                            continue
                        success = store.consume(op.cost, op.name, priority=op.priority)
                        bio_completed.record(success)
                        if op.priority >= 5 and store.get_state() in (
                            MetabolicState.STARVING, MetabolicState.CONSERVING,
                        ):
                            bio_critical_served.record(success)
            else:
                for i, op in enumerate(ops):
                    if i > 0 and i % 10 == 0:
                        store.regenerate(config.regeneration_per_step)
                    scaler.update()
                    trial_states.add(scaler.state.value)
                    if not scaler.should_enable_feature(op.cost):
                        if op.priority < 5:
                            bio_completed.record(False)
                            continue
                    success = store.consume(op.cost, op.name, priority=op.priority)
                    bio_completed.record(success)
                    if op.priority >= 5 and store.get_state() in (
                        MetabolicState.STARVING, MetabolicState.CONSERVING,
                    ):
                        bio_critical_served.record(success)

            bio_transitions.append(scaler._transitions)
            bio_states_visited.append(trial_states)

            # --- Ablated variant: ATP_Store without mTOR (fixed workers) ---
            store_abl = ATP_Store(
                budget=config.initial_budget,
                gtp_budget=config.gtp_budget,
                nadh_reserve=config.nadh_reserve,
                max_debt=config.max_debt,
                silent=True,
            )
            op_idx = 0
            if is_scaling:
                # Fixed worker count = midpoint (no adaptive scaling)
                fixed_workers = (config.min_workers + config.max_workers) // 2
                max_steps = 100
                for step in range(max_steps):
                    if step > 0 and step % 10 == 0:
                        store_abl.regenerate(config.regeneration_per_step)
                    for _ in range(fixed_workers):
                        if op_idx >= len(ops):
                            break
                        op = ops[op_idx]
                        op_idx += 1
                        success = store_abl.consume(op.cost, op.name, priority=op.priority)
                        abl_completed.record(success)
                        if op.priority >= 5 and store_abl.get_state() in (
                            MetabolicState.STARVING, MetabolicState.CONSERVING,
                        ):
                            abl_critical_served.record(success)
            else:
                for i, op in enumerate(ops):
                    if i > 0 and i % 10 == 0:
                        store_abl.regenerate(config.regeneration_per_step)
                    success = store_abl.consume(op.cost, op.name, priority=op.priority)
                    abl_completed.record(success)
                    if op.priority >= 5 and store_abl.get_state() in (
                        MetabolicState.STARVING, MetabolicState.CONSERVING,
                    ):
                        abl_critical_served.record(success)

            # --- Naive variant: SimpleBudget ---
            budget = SimpleBudget(budget=config.initial_budget)
            for i, op in enumerate(ops):
                # No regeneration — simple counter
                success = budget.consume(op.cost, op.name, op.priority)
                naive_completed.record(success)
                # Simple budget has no priority gating, so critical_served
                # is the same as completed (no special treatment)
                if op.priority >= 5 and budget.get_balance() < config.initial_budget * 0.3:
                    naive_critical_served.record(success)

        def _counter_to_trial(c: Counter, variant: Variant) -> TrialResult:
            return TrialResult(variant=variant, metric_name=c.name, success=c.success, total=c.total)

        br = BenchmarkResult(
            name="metabolism",
            scenario_name=scenario_name,
            seed=rng.randint(0, 2**32),
            config={"budget": config.initial_budget, "scenario": scenario_name},
        )
        br.variants = {
            Variant.BIOLOGICAL: {
                c.name: _counter_to_trial(c, Variant.BIOLOGICAL)
                for c in [bio_completed, bio_critical_served] if c.total > 0
            },
            Variant.ABLATED: {
                c.name: _counter_to_trial(c, Variant.ABLATED)
                for c in [abl_completed, abl_critical_served] if c.total > 0
            },
            Variant.NAIVE: {
                c.name: _counter_to_trial(c, Variant.NAIVE)
                for c in [naive_completed, naive_critical_served] if c.total > 0
            },
        }

        # Add mTOR transition stats as metadata
        if bio_transitions:
            avg_transitions = sum(bio_transitions) / len(bio_transitions)
            br.variants[Variant.BIOLOGICAL].setdefault(
                "operations_completed", TrialResult(
                    variant=Variant.BIOLOGICAL,
                    metric_name="operations_completed",
                    success=bio_completed.success,
                    total=bio_completed.total,
                ),
            ).metadata["avg_mtor_transitions"] = avg_transitions

        results.append(br)

    return results
