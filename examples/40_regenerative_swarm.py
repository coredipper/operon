#!/usr/bin/env python3
"""
Example 40: Regenerative Swarm (Metabolic Self-Healing)
======================================================

Demonstrates the biological Apoptosis + Regeneration pattern where stuck
agents are detected, cleanly terminated, and replaced with fresh workers
that inherit summarized learnings from their predecessors.

Key concepts:
- Entropy monitoring: Detect stuck agents via output repetition
- Clean apoptosis: Preserve useful state before termination
- Memory inheritance: Pass learnings to successor workers
- Adaptive regeneration: New workers avoid known failure modes

Biological parallel:
- Apoptosis: Programmed cell death that removes damaged cells cleanly
- Stem cell regeneration: Replace lost cells while preserving tissue function
- Cellular debris signaling: Dying cells inform neighbors about threats

The key insight: Agent death is not just cleanup - it's information transfer.
The dying agent's experience becomes a lesson for its successor.

Prerequisites:
- Example 37 for metabolic swarm budgeting concepts
- Example 04 for basic ATP budgeting

See Also:
- operon_ai/healing/regenerative_swarm.py for the core implementation
- Article Section 5.5: Homeostasis - Metabolic Healing

Usage:
    python examples/40_regenerative_swarm.py
    python examples/40_regenerative_swarm.py --test
"""

import sys
import random

from operon_ai.healing import (
    RegenerativeSwarm,
    SimpleWorker,
    WorkerMemory,
    ApoptosisReason,
    create_default_summarizer,
)


# =============================================================================
# Worker Implementations
# =============================================================================

def create_stubborn_worker(name: str, memory_hints: list[str]) -> SimpleWorker:
    """
    Create a worker that gets stuck in a loop.

    This simulates an agent that keeps repeating the same approach
    without making progress.
    """
    _ = memory_hints  # Ignores hints

    def work(task: str, memory: WorkerMemory) -> str:
        _ = task
        # Always outputs the same thing = stuck
        return "THINKING: Let me analyze this problem... still analyzing..."

    return SimpleWorker(id=name, work_function=work)


def create_learning_worker(name: str, memory_hints: list[str]) -> SimpleWorker:
    """
    Create a worker that learns from predecessors.

    If memory hints mention previous failures, this worker tries
    a different approach.
    """
    has_hint = any("different approach" in hint.lower() for hint in memory_hints)

    def work(task: str, memory: WorkerMemory) -> str:
        _ = task
        step = len(memory.output_history)

        if has_hint:
            # Learned from predecessor - try different approach
            if step == 0:
                return "STRATEGY: Based on previous failures, trying new approach..."
            elif step == 1:
                return "PROGRESS: New approach shows promise..."
            else:
                return "SUCCESS: Problem solved with alternative method!"
        else:
            # No hints - stuck repeating same output (triggers summarizer hint)
            return "THINKING: Analyzing... (no progress)"

    return SimpleWorker(id=name, work_function=work)


def create_eventually_successful_worker(name: str, memory_hints: list[str]) -> SimpleWorker:
    """
    Create a worker that eventually succeeds after some exploration.

    Simulates an agent that explores different approaches.
    """
    attempts = {"count": 0}
    generation = int(name.split("_")[1]) if "_" in name else 1
    hint_boost = len(memory_hints) > 0  # Hints help speed up success

    def work(task: str, memory: WorkerMemory) -> str:
        _ = task
        attempts["count"] += 1
        step = attempts["count"]

        # Earlier generations take longer, hints help
        success_threshold = max(2, 5 - generation - (2 if hint_boost else 0))

        if step < success_threshold:
            strategies = [
                "Analyzing problem structure...",
                "Testing hypothesis A...",
                "Hypothesis A failed, trying B...",
                "Making progress with approach B...",
            ]
            return strategies[step % len(strategies)]
        else:
            return "SUCCESS: Found the solution!"

    return SimpleWorker(id=name, work_function=work)


def create_random_outcome_worker(name: str, memory_hints: list[str]) -> SimpleWorker:
    """
    Create a worker with probabilistic outcomes.

    Has higher success probability if it has memory hints.
    """
    success_prob = 0.3 if not memory_hints else 0.7

    def work(task: str, memory: WorkerMemory) -> str:
        _ = task
        step = len(memory.output_history)

        if step > 3 and random.random() < success_prob:
            return "SUCCESS: Completed the task!"

        responses = [
            "Working on subtask 1...",
            "Processing intermediate results...",
            "Validating partial solution...",
            "Refining approach...",
        ]
        return responses[step % len(responses)]

    return SimpleWorker(id=name, work_function=work)


# =============================================================================
# Demo: Stuck Worker Detection and Regeneration
# =============================================================================

def demo_stuck_worker_regeneration():
    """
    Demo where a stuck worker is detected and regenerated.

    The first worker gets stuck in a loop. The swarm detects this via
    entropy collapse and regenerates with a new worker that inherits
    the failure summary.
    """
    print("=" * 60)
    print("Demo 1: Stuck Worker Detection and Regeneration")
    print("=" * 60)

    swarm = RegenerativeSwarm(
        worker_factory=create_learning_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,  # High similarity = stuck
        max_steps_per_worker=5,
        max_regenerations=2,
    )

    print("\nWorker 1 will get stuck (no hints). Swarm will detect and regenerate.")
    print("Worker 2 will receive hints and succeed.\n")

    result = swarm.supervise("Solve the complex puzzle")

    print(f"\nSwarm Result:")
    print(f"  Success: {result.success}")
    print(f"  Output: {result.output}")
    print(f"  Total Workers Spawned: {result.total_workers_spawned}")
    print(f"  Apoptosis Events: {len(result.apoptosis_events)}")
    print(f"  Regeneration Events: {len(result.regeneration_events)}")

    if result.apoptosis_events:
        print("\nApoptosis Events:")
        for event in result.apoptosis_events:
            print(f"  - Worker {event.worker_id}: {event.reason.value}")
            print(f"    Summary passed to successor: {event.memory_summary}")

    if result.regeneration_events:
        print("\nRegeneration Events:")
        for event in result.regeneration_events:
            print(f"  - {event.old_worker_id} → {event.new_worker_id}")
            print(f"    Hints injected: {event.injected_summary}")

    return result


# =============================================================================
# Demo: Maximum Regenerations Exhausted
# =============================================================================

def demo_max_regenerations():
    """
    Demo where all regenerations are exhausted.

    All workers get stuck, and the swarm gives up after max_regenerations.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Maximum Regenerations Exhausted")
    print("=" * 60)

    swarm = RegenerativeSwarm(
        worker_factory=create_stubborn_worker,  # Always gets stuck
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=4,
        max_regenerations=2,  # Allow 2 regenerations
    )

    print("\nAll workers will get stuck. Swarm will exhaust regenerations.\n")

    result = swarm.supervise("Impossible task")

    print(f"\nSwarm Result:")
    print(f"  Success: {result.success}")
    print(f"  Total Workers Spawned: {result.total_workers_spawned}")
    print(f"  Apoptosis Events: {len(result.apoptosis_events)}")

    print("\nApoptosis History (all workers failed):")
    for event in result.apoptosis_events:
        print(f"  - {event.worker_id}: {event.reason.value} - {event.details}")

    return result


# =============================================================================
# Demo: First Worker Succeeds
# =============================================================================

def demo_first_worker_success():
    """
    Demo where the first worker succeeds without needing regeneration.

    This is the ideal case - no apoptosis needed.
    """
    print("\n" + "=" * 60)
    print("Demo 3: First Worker Succeeds (No Regeneration Needed)")
    print("=" * 60)

    swarm = RegenerativeSwarm(
        worker_factory=create_eventually_successful_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=10,  # Enough steps to succeed
        max_regenerations=3,
    )

    print("\nFirst worker will succeed after a few steps.\n")

    result = swarm.supervise("Solvable task")

    print(f"\nSwarm Result:")
    print(f"  Success: {result.success}")
    print(f"  Output: {result.output}")
    print(f"  Total Workers Spawned: {result.total_workers_spawned}")
    print(f"  Apoptosis Events: {len(result.apoptosis_events)}")
    print(f"  Final Worker: {result.final_worker_id}")

    return result


# =============================================================================
# Demo: Progressive Learning Across Generations
# =============================================================================

def demo_progressive_learning():
    """
    Demo showing how each generation learns from previous failures.

    Workers improve their success rate as they inherit more hints.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Progressive Learning Across Generations")
    print("=" * 60)

    # Track hints received by each worker
    generation_hints: list[list[str]] = []

    def tracking_factory(name: str, memory_hints: list[str]) -> SimpleWorker:
        generation_hints.append(memory_hints.copy())
        generation = len(generation_hints)

        # Each generation is slightly more likely to succeed
        success_step = max(2, 6 - generation)

        def work(task: str, memory: WorkerMemory) -> str:
            _ = task
            step = len(memory.output_history)

            if step >= success_step:
                return f"SUCCESS: Generation {generation} solved it!"

            if step < 3:
                return f"THINKING: Generation {generation} step {step}..."
            else:
                return f"THINKING: Generation {generation} step {step}..."  # Will trigger entropy

        return SimpleWorker(id=name, work_function=work)

    swarm = RegenerativeSwarm(
        worker_factory=tracking_factory,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.8,
        max_steps_per_worker=4,
        max_regenerations=5,
    )

    print("\nEach generation inherits more hints and improves.\n")

    result = swarm.supervise("Multi-generation task")

    print(f"\nSwarm Result:")
    print(f"  Success: {result.success}")
    print(f"  Total Workers Spawned: {result.total_workers_spawned}")

    print("\nHints received by each generation:")
    for i, hints in enumerate(generation_hints):
        print(f"  Generation {i + 1}: {len(hints)} hints")
        for hint in hints[:2]:
            print(f"    - {hint[:60]}...")

    return result


# =============================================================================
# Smoke Test
# =============================================================================

def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke test...")

    # Test 1: Successful regeneration
    swarm = RegenerativeSwarm(
        worker_factory=create_learning_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=2,
        silent=True,
    )
    result = swarm.supervise("test task")
    assert result.success, "Expected success after regeneration"
    assert len(result.apoptosis_events) >= 1, "Expected at least one apoptosis"
    assert result.total_workers_spawned >= 2, "Expected multiple workers"

    # Test 2: Exhausted regenerations
    swarm2 = RegenerativeSwarm(
        worker_factory=create_stubborn_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=3,
        max_regenerations=1,
        silent=True,
    )
    result2 = swarm2.supervise("impossible")
    assert not result2.success, "Expected failure"
    assert len(result2.apoptosis_events) >= 1, "Expected apoptosis events"

    # Test 3: First worker success
    swarm3 = RegenerativeSwarm(
        worker_factory=create_eventually_successful_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=10,
        max_regenerations=3,
        silent=True,
    )
    result3 = swarm3.supervise("easy task")
    assert result3.success, "Expected first-worker success"
    assert result3.total_workers_spawned == 1, "Expected single worker"
    assert len(result3.apoptosis_events) == 0, "Expected no apoptosis"

    print("Smoke test passed!")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 40: Regenerative Swarm")
    print("Metabolic Self-Healing through Apoptosis and Regeneration")
    print("=" * 60)

    demo_stuck_worker_regeneration()
    demo_max_regenerations()
    demo_first_worker_success()
    demo_progressive_learning()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Regenerative Swarm demonstrates metabolic healing:

1. Entropy Monitoring: Detect stuck agents via output repetition
2. Clean Apoptosis: Terminate gracefully, preserving useful state
3. Memory Inheritance: Pass learnings to successor workers
4. Adaptive Success: Later generations benefit from earlier failures

Key biological parallel: When cells die via apoptosis, they don't
just disappear - they release signals that inform neighboring cells
about threats and help the tissue adapt.

Key software insight: Agent death is information transfer, not just
cleanup. "Worker_1 died trying strategy X" helps Worker_2 avoid the
same mistake and try something different.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
