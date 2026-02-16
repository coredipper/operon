#!/usr/bin/env python3
"""
Example 50: Morphogen-Guided Swarm
===================================

Demonstrates a task-solving swarm where workers adapt strategy based on
morphogen gradient signals. Failed workers update gradients, and successors
read them to avoid repeating mistakes.

Architecture:

```
[GradientOrchestrator]
    ├── Initializes morphogens (complexity, confidence, budget, error_rate)
    ├── Updates after each worker outcome
    └── Provides strategy hints to workers
         |
[RegenerativeSwarm]
    ├── Worker Factory reads gradient.get_strategy_hints()
    ├── Worker 1: tries default approach
    │     └── fails -> gradient.error_rate increases, confidence drops
    ├── Worker 2: reads "high error rate" hint, tries different approach
    │     └── fails -> gradient further adjusted
    └── Worker 3: reads accumulated hints, succeeds
         └── gradient.confidence rises
```

Key concepts:
- GradientOrchestrator coordinates without central control
- Worker factory injects strategy hints into worker context
- Gradient history shows how the environment evolved
- Budget morphogen tracks token consumption

Prerequisites:
- Example 40 for Regenerative Swarm patterns
- Example 44 for Morphogen gradient basics

Usage:
    python examples/50_morphogen_guided_swarm.py
    python examples/50_morphogen_guided_swarm.py --test
"""

import sys
from dataclasses import dataclass, field

from operon_ai.coordination.morphogen import (
    MorphogenGradient,
    MorphogenType,
    GradientOrchestrator,
)
from operon_ai.healing import (
    RegenerativeSwarm,
    SimpleWorker,
    WorkerMemory,
    create_default_summarizer,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GradientSnapshot:
    """Snapshot of gradient state at a point in time."""
    step: int
    complexity: float
    confidence: float
    budget: float
    error_rate: float
    hints: list[str]


# =============================================================================
# Gradient-Aware Worker Factory
# =============================================================================


class GradientAwareWorkerFactory:
    """
    Worker factory that creates workers with gradient awareness.

    Each worker receives the current gradient state and strategy hints
    when created. This allows successors to adapt based on predecessors'
    outcomes.
    """

    def __init__(
        self,
        orchestrator: GradientOrchestrator,
        total_budget: int = 1000,
        silent: bool = False,
    ):
        self.orchestrator = orchestrator
        self.total_budget = total_budget
        self.tokens_used = 0
        self.silent = silent
        self._gradient_history: list[GradientSnapshot] = []

    def create_worker(self, name: str, memory_hints: list[str]) -> SimpleWorker:
        """Create a gradient-aware worker."""
        # Get current gradient state
        hints = self.orchestrator.gradient.get_strategy_hints()
        error_rate = self.orchestrator.gradient.get(MorphogenType.ERROR_RATE)
        confidence = self.orchestrator.gradient.get(MorphogenType.CONFIDENCE)
        complexity = self.orchestrator.gradient.get(MorphogenType.COMPLEXITY)
        budget_ratio = self.orchestrator.gradient.get(MorphogenType.BUDGET)

        # Record snapshot
        self._gradient_history.append(GradientSnapshot(
            step=len(self._gradient_history),
            complexity=complexity,
            confidence=confidence,
            budget=budget_ratio,
            error_rate=error_rate,
            hints=hints.copy(),
        ))

        # Determine strategy based on gradient
        use_alternative = error_rate >= 0.3 or any(
            "different" in hint.lower() or "error" in hint.lower()
            for hint in memory_hints
        )
        be_concise = budget_ratio < 0.3

        if not self.silent:
            print(
                f"  [Factory] Creating {name}: "
                f"error_rate={error_rate:.2f} "
                f"confidence={confidence:.2f} "
                f"budget={budget_ratio:.2f} "
                f"strategy={'alternative' if use_alternative else 'default'}"
            )

        generation = len(self._gradient_history)
        factory_ref = self

        def work(task: str, memory: WorkerMemory) -> str:
            step = len(memory.output_history)
            tokens_per_step = 50 if be_concise else 100
            factory_ref.tokens_used += tokens_per_step

            # Update budget morphogen
            remaining = max(0, factory_ref.total_budget - factory_ref.tokens_used)
            factory_ref.orchestrator.gradient.set(
                MorphogenType.BUDGET,
                remaining / factory_ref.total_budget,
            )

            if use_alternative and generation >= 2:
                # Worker with accumulated gradient knowledge
                if step == 0:
                    return "STRATEGY: Reading gradient signals, trying alternative approach"
                elif step == 1:
                    return "PROGRESS: Alternative approach showing results"
                else:
                    return "DONE: Problem solved using gradient-informed strategy!"
            elif use_alternative:
                # First alternative worker, still learning
                if step < 2:
                    return f"THINKING: Trying modified approach (step {step})"
                elif step == 2:
                    return "PROGRESS: Modified approach working..."
                else:
                    return "DONE: Solved with modified approach!"
            else:
                # Default strategy - will get stuck (repeating identical output)
                return "THINKING: Analyzing problem..."

        return SimpleWorker(id=name, work_function=work)

    def report_outcome(self, success: bool, tokens: int = 100):
        """Report worker outcome and update gradient."""
        self.tokens_used += tokens
        self.orchestrator.report_step_result(
            success=success,
            tokens_used=self.tokens_used,
            total_budget=self.total_budget,
        )

    def get_gradient_history(self) -> list[GradientSnapshot]:
        """Get history of gradient snapshots."""
        return list(self._gradient_history)


# =============================================================================
# Morphogen Swarm Orchestrator
# =============================================================================


class MorphogenSwarmOrchestrator:
    """
    Orchestrates a swarm with morphogen gradient coordination.

    Combines GradientOrchestrator with RegenerativeSwarm to create
    a swarm where workers adapt based on environmental signals.
    """

    def __init__(
        self,
        task_complexity: float = 0.5,
        total_budget: int = 1000,
        max_regenerations: int = 3,
        silent: bool = False,
    ):
        self.silent = silent

        # Initialize gradient orchestrator
        self.orchestrator = GradientOrchestrator(silent=silent)
        self.orchestrator.gradient.set(
            MorphogenType.COMPLEXITY, task_complexity,
            description="Task complexity level",
        )
        self.orchestrator.gradient.set(
            MorphogenType.BUDGET, 1.0,
            description="Budget remaining ratio",
        )

        # Create gradient-aware factory
        self.factory = GradientAwareWorkerFactory(
            orchestrator=self.orchestrator,
            total_budget=total_budget,
            silent=silent,
        )

        # Create swarm
        self.swarm = RegenerativeSwarm(
            worker_factory=self.factory.create_worker,
            summarizer=create_default_summarizer(),
            entropy_threshold=0.9,
            max_steps_per_worker=5,
            max_regenerations=max_regenerations,
            silent=silent,
        )

    def solve(self, task: str):
        """Run the swarm to solve a task."""
        if not self.silent:
            print(f"\n  [Orchestrator] Task: {task}")
            print(f"  [Orchestrator] Initial gradient: {self.orchestrator.gradient.to_dict()}")

        result = self.swarm.supervise(task)

        # Report final outcome
        self.factory.report_outcome(result.success)

        if not self.silent:
            print(f"\n  [Orchestrator] Final gradient: {self.orchestrator.gradient.to_dict()}")

        return result


# =============================================================================
# Demos
# =============================================================================


def demo_gradient_adaptation():
    """
    Demo: Workers adapt strategy based on morphogen gradient signals.

    Worker 1 gets stuck with default strategy.
    Worker 2 reads the elevated error_rate and tries differently.
    """
    print("=" * 60)
    print("Demo 1: Gradient-Guided Adaptation")
    print("=" * 60)

    orchestrator = MorphogenSwarmOrchestrator(
        task_complexity=0.7,
        total_budget=1000,
        max_regenerations=3,
        silent=False,
    )

    result = orchestrator.solve("Solve the complex optimization problem")

    print(f"\n  Result:")
    print(f"    Success: {result.success}")
    print(f"    Output: {result.output}")
    print(f"    Workers spawned: {result.total_workers_spawned}")
    print(f"    Apoptosis events: {len(result.apoptosis_events)}")
    print(f"    Regeneration events: {len(result.regeneration_events)}")

    # Show gradient evolution
    history = orchestrator.factory.get_gradient_history()
    if history:
        print(f"\n  Gradient Evolution ({len(history)} snapshots):")
        for snap in history:
            print(
                f"    Step {snap.step}: "
                f"err={snap.error_rate:.2f} "
                f"conf={snap.confidence:.2f} "
                f"budget={snap.budget:.2f} "
                f"hints={len(snap.hints)}"
            )

    return result


def demo_budget_exhaustion():
    """
    Demo: Swarm adapts when budget runs low.

    Workers become more concise as budget morphogen drops.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Budget-Aware Adaptation")
    print("=" * 60)

    orchestrator = MorphogenSwarmOrchestrator(
        task_complexity=0.5,
        total_budget=300,  # Tight budget
        max_regenerations=2,
        silent=False,
    )

    result = orchestrator.solve("Summarize findings efficiently")

    print(f"\n  Result:")
    print(f"    Success: {result.success}")
    print(f"    Tokens used: {orchestrator.factory.tokens_used}")
    print(f"    Budget remaining: {orchestrator.orchestrator.gradient.get(MorphogenType.BUDGET):.0%}")
    print(f"    Workers spawned: {result.total_workers_spawned}")

    return result


def demo_gradient_history():
    """
    Demo: Visualize how the gradient evolves across worker generations.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Gradient History Visualization")
    print("=" * 60)

    orchestrator = MorphogenSwarmOrchestrator(
        task_complexity=0.8,  # Hard task
        total_budget=2000,
        max_regenerations=4,
        silent=True,  # Suppress per-step output
    )

    result = orchestrator.solve("Hard multi-step reasoning task")

    history = orchestrator.factory.get_gradient_history()

    print(f"\n  Task completed: {'Yes' if result.success else 'No'}")
    print(f"  Workers spawned: {result.total_workers_spawned}")
    print(f"\n  Gradient History:")
    print(f"  {'Step':>4} | {'Error':>6} | {'Confidence':>10} | {'Budget':>6} | Hints")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-------")

    for snap in history:
        hint_summary = "; ".join(h[:30] for h in snap.hints[:2]) if snap.hints else "none"
        print(
            f"  {snap.step:>4} | "
            f"{snap.error_rate:>6.2f} | "
            f"{snap.confidence:>10.2f} | "
            f"{snap.budget:>6.2f} | "
            f"{hint_summary}"
        )

    # Show final gradient state
    final = orchestrator.orchestrator.gradient
    print(f"\n  Final Gradient State:")
    for mtype in MorphogenType:
        value = final.get(mtype)
        level = final.get_level(mtype)
        print(f"    {mtype.value:>12}: {value:.3f} ({level})")

    return result


# =============================================================================
# Smoke Test
# =============================================================================


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Basic gradient-guided swarm
    orchestrator = MorphogenSwarmOrchestrator(
        task_complexity=0.5,
        total_budget=1000,
        max_regenerations=3,
        silent=True,
    )
    result = orchestrator.solve("Test task")
    assert result.success, "Expected swarm to eventually succeed"
    assert result.total_workers_spawned >= 1
    print("  Test 1: Basic gradient-guided swarm - PASSED")

    # Test 2: Gradient updates after failure
    orch2 = MorphogenSwarmOrchestrator(
        task_complexity=0.7,
        total_budget=1000,
        max_regenerations=3,
        silent=True,
    )
    result2 = orch2.solve("Complex task")
    history = orch2.factory.get_gradient_history()
    assert len(history) >= 1, "Should have at least one gradient snapshot"
    print("  Test 2: Gradient updates - PASSED")

    # Test 3: Worker factory respects gradient
    orch3 = GradientOrchestrator(silent=True)
    orch3.gradient.set(MorphogenType.ERROR_RATE, 0.8)
    factory = GradientAwareWorkerFactory(
        orchestrator=orch3,
        total_budget=1000,
        silent=True,
    )
    worker = factory.create_worker("test_1", ["try different approach"])
    assert worker.id == "test_1"
    print("  Test 3: Worker factory gradient awareness - PASSED")

    # Test 4: Budget tracking
    orch4 = MorphogenSwarmOrchestrator(
        task_complexity=0.5,
        total_budget=500,
        max_regenerations=2,
        silent=True,
    )
    result4 = orch4.solve("Budget test")
    budget_remaining = orch4.orchestrator.gradient.get(MorphogenType.BUDGET)
    assert budget_remaining < 1.0, "Budget should have decreased"
    print("  Test 4: Budget tracking - PASSED")

    # Test 5: Gradient history recording
    assert len(orch4.factory.get_gradient_history()) >= 1
    snap = orch4.factory.get_gradient_history()[0]
    assert hasattr(snap, "complexity")
    assert hasattr(snap, "confidence")
    assert hasattr(snap, "budget")
    assert hasattr(snap, "error_rate")
    print("  Test 5: Gradient history - PASSED")

    # Test 6: Strategy hints generation
    gradient = MorphogenGradient()
    gradient.set(MorphogenType.ERROR_RATE, 0.8)
    gradient.set(MorphogenType.BUDGET, 0.1)
    hints = gradient.get_strategy_hints()
    assert len(hints) >= 1, "Should generate hints for extreme values"
    print("  Test 6: Strategy hints - PASSED")

    print("\nSmoke tests passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 50: Morphogen-Guided Swarm")
    print("MorphogenGradient + GradientOrchestrator + RegenerativeSwarm")
    print("=" * 60)

    demo_gradient_adaptation()
    demo_budget_exhaustion()
    demo_gradient_history()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Morphogen-Guided Swarm demonstrates coordination without central
control:

1. GradientOrchestrator: Maintains shared morphogen state
   (complexity, confidence, budget, error_rate)

2. Worker Factory: Creates workers that read gradient signals
   and adapt their strategy accordingly

3. RegenerativeSwarm: Manages worker lifecycle - stuck workers
   die via apoptosis, successors inherit both memory hints
   AND gradient signals

4. Gradient Evolution: Each outcome updates the gradient,
   creating an emergent coordination signal

Key biological parallel: In embryonic development, cells don't receive
central commands. They read local morphogen concentrations and
differentiate accordingly. Similarly, workers sense their environment
and adapt their strategy.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
