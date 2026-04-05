"""Pathway-grounded scenario parameters for metabolism benchmark.

Derives test scenarios from KEGG hsa04152 (AMPK signaling pathway).

The biological insight: AMPK responds to the AMP:ATP RATIO and its
RATE OF CHANGE, not just the absolute ATP level.  This means the
biological system can anticipate depletion (rapid consumption rate)
and act preemptively, while a simple counter only reacts when empty.

Threshold values derived from AMPK pathway:
- AMP:ATP < 0.3: anabolic (growth factors active, mTOR dominant)
- 0.3-0.7: maintenance (balanced metabolism)
- 0.7-0.9: catabolic (AMPK activating, fatty acid oxidation)
- > 0.9: autophagy (ULK1-mediated, severe energy crisis)
- Hysteresis margin 0.05 (Hill coefficient kinetics create natural lag)
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class AMPKParams:
    """Parameters derived from KEGG hsa04152."""

    growth_ratio: float = 0.3
    conservation_ratio: float = 0.7
    autophagy_ratio: float = 0.9
    hysteresis_margin: float = 0.05
    rate_sensitivity: float = 0.1


@dataclass
class Operation:
    """A single operation to be consumed from the budget."""

    cost: int
    name: str
    priority: int = 0


def generate_constant_load(
    budget: int,
    cost_per_op: int,
    length: int,
    rng: random.Random,
) -> list[Operation]:
    """Steady consumption at constant rate.

    Tests basic state tracking — does the system correctly identify
    metabolic states as resources deplete linearly?
    """
    return [
        Operation(cost=cost_per_op, name=f"op_{i}", priority=rng.randint(0, 3))
        for i in range(length)
    ]


def generate_bursty_load(
    budget: int,
    burst_size: int,
    burst_interval: int,
    length: int,
    rng: random.Random,
) -> list[Operation]:
    """Periodic spikes in consumption.

    Tests whether hysteresis prevents oscillation between metabolic
    states.  The biological system should NOT flip-flop between
    NORMAL and CONSERVING on every burst.
    """
    ops = []
    for i in range(length):
        if i % burst_interval == 0:
            # Burst: high-cost operations
            cost = burst_size
            priority = rng.randint(3, 7)
        else:
            # Normal: low-cost operations
            cost = max(1, burst_size // 10)
            priority = rng.randint(0, 3)
        ops.append(Operation(cost=cost, name=f"op_{i}", priority=priority))
    return ops


def generate_gradual_depletion(
    budget: int,
    length: int,
    rng: random.Random,
) -> list[Operation]:
    """Slowly increasing consumption rate.

    Tests whether rate-sensing triggers CONSERVATION earlier than
    absolute-level checking.  The biological system should detect
    the accelerating depletion and conserve preemptively.
    """
    ops = []
    for i in range(length):
        # Cost increases linearly over time
        cost = max(1, int(1 + (i / length) * (budget / length) * 2))
        priority = rng.randint(0, 5)
        ops.append(Operation(cost=cost, name=f"op_{i}", priority=priority))
    return ops


def generate_sudden_spike(
    budget: int,
    spike_at: int,
    spike_cost: int,
    length: int,
    rng: random.Random,
) -> list[Operation]:
    """Normal operation then a single large cost.

    Tests recovery behavior after acute stress.  After the spike,
    the biological system should enter CONSERVATION, then recover
    to NORMAL as resources regenerate.
    """
    ops = []
    for i in range(length):
        if i == spike_at:
            cost = spike_cost
            priority = 8  # High priority — must be served
        else:
            cost = max(1, budget // (length * 2))
            priority = rng.randint(0, 3)
        ops.append(Operation(cost=cost, name=f"op_{i}", priority=priority))
    return ops


def generate_mixed_priority(
    budget: int,
    length: int,
    rng: random.Random,
) -> list[Operation]:
    """Mix of high and low priority operations under depletion.

    Tests graceful degradation: does the biological system correctly
    serve high-priority operations while rejecting low-priority ones
    when in STARVING state?
    """
    ops = []
    for i in range(length):
        cost = max(1, budget // length)
        # Alternate between critical and routine operations
        if rng.random() < 0.3:
            priority = rng.randint(5, 10)  # Critical
            name = f"critical_{i}"
        else:
            priority = rng.randint(0, 3)  # Routine
            name = f"routine_{i}"
        ops.append(Operation(cost=cost, name=name, priority=priority))
    return ops


SCENARIOS = {
    "constant": generate_constant_load,
    "bursty": generate_bursty_load,
    "gradual_depletion": generate_gradual_depletion,
    "sudden_spike": generate_sudden_spike,
    "mixed_priority": generate_mixed_priority,
}
