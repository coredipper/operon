#!/usr/bin/env python3
"""
Example 51: Epiplexity Healing Cascade
=======================================

Demonstrates an agent that detects when it's stuck (via epiplexity monitoring)
and escalates through increasingly aggressive healing interventions.

Architecture (Cascade of escalating interventions):

```
Agent Output
    |
[EpiplexityMonitor: measure novelty per message]
    |
    ├── HEALTHY/EXPLORING -> continue normally
    |
    ├── STAGNANT -> Stage 1: AutophagyDaemon prunes stale context
    |
    ├── CRITICAL -> Stage 2: RegenerativeSwarm kills worker,
    |                         spawns fresh one with summary
    |
    └── Still stuck -> Stage 3: Abort with diagnostic report
```

Key concepts:
- EpiplexityMonitor detects epistemic stagnation (low Bayesian surprise)
- Cascade of interventions: mild (autophagy) -> moderate (regeneration) -> abort
- Each stage is attempted before escalating to the next
- Diagnostic report captures the full stagnation history

Prerequisites:
- Example 42 for EpiplexityMonitor basics
- Example 39 for Autophagy
- Example 40 for Regenerative Swarm

Usage:
    python examples/51_epiplexity_healing_cascade.py
    python examples/51_epiplexity_healing_cascade.py --test
"""

import sys
from dataclasses import dataclass, field
from enum import Enum

from operon_ai import (
    HistoneStore,
    Lysosome,
    Waste,
    WasteType,
)
from operon_ai.health import (
    EpiplexityMonitor,
    MockEmbeddingProvider,
    HealthStatus,
)
from operon_ai.healing import (
    AutophagyDaemon,
    RegenerativeSwarm,
    SimpleWorker,
    WorkerMemory,
    create_default_summarizer,
    create_simple_summarizer,
)


# =============================================================================
# Data Structures
# =============================================================================


class InterventionLevel(str, Enum):
    """Intervention levels in the healing cascade."""
    NONE = "none"
    AUTOPHAGY = "autophagy"        # Stage 1: prune context
    REGENERATION = "regeneration"  # Stage 2: kill + respawn worker
    ABORT = "abort"                # Stage 3: give up with diagnostics


@dataclass
class StagnationDiagnostic:
    """Diagnostic report for stagnation events."""
    total_messages: int
    stagnant_count: int
    critical_count: int
    interventions_applied: list[str]
    final_status: HealthStatus
    epiplexity_history: list[float]
    context_pruned: bool
    worker_regenerated: bool


@dataclass
class CascadeResult:
    """Result of the healing cascade."""
    success: bool
    output: str
    intervention_level: InterventionLevel
    diagnostic: StagnationDiagnostic


# =============================================================================
# Stagnation Detector
# =============================================================================


class StagnationDetector:
    """
    Monitors agent outputs for epistemic stagnation using EpiplexityMonitor.

    When stagnation is detected, classifies the severity and recommends
    an intervention level.
    """

    def __init__(
        self,
        window_size: int = 5,
        threshold: float = 0.2,
        critical_duration: int = 3,
        silent: bool = False,
    ):
        self.silent = silent
        self.monitor = EpiplexityMonitor(
            embedding_provider=MockEmbeddingProvider(dim=64),
            alpha=0.5,
            window_size=window_size,
            threshold=threshold,
            critical_duration=critical_duration,
        )
        self._measurements: list[dict] = []

    def observe(self, message: str) -> HealthStatus:
        """
        Observe a new agent output and return health status.
        """
        result = self.monitor.measure(message)

        self._measurements.append({
            "message_preview": message[:50],
            "epiplexity": result.epiplexity,
            "novelty": result.embedding_novelty,
            "status": result.status,
        })

        if not self.silent:
            print(
                f"  [Epiplexity] novelty={result.embedding_novelty:.3f} "
                f"epiplexity={result.epiplexity:.3f} "
                f"integral={result.epiplexic_integral:.3f} "
                f"status={result.status.value}"
            )

        return result.status

    def get_epiplexity_history(self) -> list[float]:
        """Get history of epiplexity scores."""
        return [m["epiplexity"] for m in self._measurements]

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return self.monitor.stats()


# =============================================================================
# Healing Cascade
# =============================================================================


class HealingCascade:
    """
    Escalating intervention cascade for stagnating agents.

    Stage 1 (Autophagy): Prune context to remove noise
    Stage 2 (Regeneration): Kill worker, spawn fresh one with summary
    Stage 3 (Abort): Give up with diagnostic report
    """

    def __init__(
        self,
        max_messages: int = 20,
        silent: bool = False,
    ):
        self.max_messages = max_messages
        self.silent = silent

        # Stagnation detector
        self.detector = StagnationDetector(
            window_size=5,
            threshold=0.2,
            critical_duration=3,
            silent=silent,
        )

        # Stage 1: Autophagy
        self.histone_store = HistoneStore()
        self.lysosome = Lysosome(silent=True)
        self.autophagy = AutophagyDaemon(
            histone_store=self.histone_store,
            lysosome=self.lysosome,
            summarizer=create_simple_summarizer(),
            toxicity_threshold=0.5,
            silent=silent,
        )

        # Tracking
        self._interventions: list[str] = []
        self._context_pruned = False
        self._worker_regenerated = False

    def run(
        self,
        messages: list[str],
        context: str = "",
    ) -> CascadeResult:
        """
        Process a sequence of agent messages through the healing cascade.

        Args:
            messages: Sequence of agent output messages to process
            context: Current agent context (for autophagy)

        Returns:
            CascadeResult with outcome and diagnostics
        """
        stagnant_count = 0
        critical_count = 0
        current_intervention = InterventionLevel.NONE
        output_lines: list[str] = []

        for i, message in enumerate(messages[:self.max_messages]):
            status = self.detector.observe(message)

            if status in (HealthStatus.HEALTHY, HealthStatus.EXPLORING, HealthStatus.CONVERGING):
                output_lines.append(message)
                stagnant_count = 0  # Reset on healthy signal
                continue

            if status == HealthStatus.STAGNANT:
                stagnant_count += 1

                if stagnant_count == 1 and current_intervention < InterventionLevel.AUTOPHAGY:
                    # Stage 1: Autophagy
                    if not self.silent:
                        print(f"\n  [Cascade] Stage 1: AUTOPHAGY triggered at message {i}")

                    context, prune_result = self.autophagy.check_and_prune(
                        context or "\n".join(messages[:i]),
                        max_tokens=4000,
                    )
                    self._context_pruned = prune_result is not None
                    current_intervention = InterventionLevel.AUTOPHAGY
                    self._interventions.append(f"autophagy at message {i}")

                    if self._context_pruned:
                        output_lines.append(f"[Context pruned: {prune_result.tokens_freed} tokens freed]")
                    continue

            if status == HealthStatus.CRITICAL:
                critical_count += 1

                if current_intervention < InterventionLevel.REGENERATION:
                    # Stage 2: Regeneration
                    if not self.silent:
                        print(f"\n  [Cascade] Stage 2: REGENERATION triggered at message {i}")

                    regen_output = self._run_regeneration(messages[:i])
                    self._worker_regenerated = True
                    current_intervention = InterventionLevel.REGENERATION
                    self._interventions.append(f"regeneration at message {i}")

                    if regen_output:
                        output_lines.append(f"[Regenerated worker output: {regen_output}]")
                        # Regeneration succeeded
                        return CascadeResult(
                            success=True,
                            output="\n".join(output_lines),
                            intervention_level=InterventionLevel.REGENERATION,
                            diagnostic=self._build_diagnostic(
                                len(messages), stagnant_count, critical_count
                            ),
                        )
                else:
                    # Stage 3: Abort
                    if not self.silent:
                        print(f"\n  [Cascade] Stage 3: ABORT at message {i}")

                    self._interventions.append(f"abort at message {i}")

                    return CascadeResult(
                        success=False,
                        output="\n".join(output_lines),
                        intervention_level=InterventionLevel.ABORT,
                        diagnostic=self._build_diagnostic(
                            len(messages), stagnant_count, critical_count
                        ),
                    )

        # Processed all messages without abort
        return CascadeResult(
            success=True,
            output="\n".join(output_lines),
            intervention_level=current_intervention,
            diagnostic=self._build_diagnostic(
                len(messages), stagnant_count, critical_count
            ),
        )

    def _run_regeneration(self, failed_messages: list[str]) -> str | None:
        """
        Run a regenerative swarm to recover from stagnation.

        Uses the failed messages as context for the new worker.
        """
        summary = "; ".join(failed_messages[-3:])[:200]

        def create_recovery_worker(name: str, hints: list[str]) -> SimpleWorker:
            has_context = bool(hints)

            def work(task: str, memory: WorkerMemory) -> str:
                step = len(memory.output_history)
                if has_context and step >= 1:
                    return "DONE: Recovered from stagnation with fresh approach!"
                return f"RECOVERY: Analyzing previous failures (step {step})"

            return SimpleWorker(id=name, work_function=work)

        swarm = RegenerativeSwarm(
            worker_factory=create_recovery_worker,
            summarizer=create_default_summarizer(),
            entropy_threshold=0.9,
            max_steps_per_worker=5,
            max_regenerations=1,
            silent=self.silent,
        )

        result = swarm.supervise(f"Recover from stagnation. Context: {summary}")

        if result.success:
            return result.output
        return None

    def _build_diagnostic(
        self,
        total_messages: int,
        stagnant_count: int,
        critical_count: int,
    ) -> StagnationDiagnostic:
        """Build diagnostic report."""
        stats = self.detector.get_stats()
        return StagnationDiagnostic(
            total_messages=total_messages,
            stagnant_count=stagnant_count,
            critical_count=critical_count,
            interventions_applied=list(self._interventions),
            final_status=HealthStatus(
                "critical" if critical_count > 0
                else "stagnant" if stagnant_count > 0
                else "healthy"
            ),
            epiplexity_history=self.detector.get_epiplexity_history(),
            context_pruned=self._context_pruned,
            worker_regenerated=self._worker_regenerated,
        )


# =============================================================================
# Demo: Healthy Agent
# =============================================================================


def demo_healthy_agent():
    """Demo: Agent with diverse outputs stays healthy."""
    print("=" * 60)
    print("Demo 1: Healthy Agent (no intervention needed)")
    print("=" * 60)

    cascade = HealingCascade(silent=False)

    # Diverse messages = healthy
    messages = [
        "First, let me analyze the requirements.",
        "The key constraint is memory efficiency.",
        "I'll use a hash map for O(1) lookups.",
        "Testing edge cases: empty input, large input.",
        "Implementation complete. Here are the results.",
        "Performance benchmarks show 2x improvement.",
    ]

    result = cascade.run(messages)

    print(f"\n  Success: {result.success}")
    print(f"  Intervention: {result.intervention_level.value}")
    print(f"  Diagnostics: stagnant={result.diagnostic.stagnant_count} "
          f"critical={result.diagnostic.critical_count}")
    return result


def demo_stagnant_agent():
    """Demo: Agent that starts repeating gets autophagy intervention."""
    print("\n" + "=" * 60)
    print("Demo 2: Stagnant Agent (autophagy intervention)")
    print("=" * 60)

    cascade = HealingCascade(silent=False)

    # Starts diverse, then repeats
    messages = [
        "Let me think about this problem.",
        "I need to consider the constraints.",
        "Hmm, let me think about this problem.",
        "I need to consider the constraints.",
        "Hmm, let me think about this problem.",
        "I need to consider the constraints.",
        "Let me try a completely different approach.",  # Recovery
        "Using dynamic programming instead.",
    ]

    result = cascade.run(messages)

    print(f"\n  Success: {result.success}")
    print(f"  Intervention: {result.intervention_level.value}")
    print(f"  Context pruned: {result.diagnostic.context_pruned}")
    print(f"  Interventions: {result.diagnostic.interventions_applied}")
    return result


def demo_critical_agent():
    """Demo: Agent deeply stuck, escalates to regeneration then abort."""
    print("\n" + "=" * 60)
    print("Demo 3: Critical Agent (regeneration + abort)")
    print("=" * 60)

    cascade = HealingCascade(silent=False)

    # Deeply stuck - same output over and over
    messages = [
        "Processing request.",
        "Processing request.",
        "Processing request.",
        "Processing request.",
        "Processing request.",
        "Processing request.",
        "Processing request.",
        "Processing request.",
        "Processing request.",
        "Processing request.",
    ]

    result = cascade.run(messages)

    print(f"\n  Success: {result.success}")
    print(f"  Intervention: {result.intervention_level.value}")
    print(f"  Worker regenerated: {result.diagnostic.worker_regenerated}")
    print(f"  Interventions: {result.diagnostic.interventions_applied}")

    # Show epiplexity history
    history = result.diagnostic.epiplexity_history
    if history:
        print(f"\n  Epiplexity history (lower = more stuck):")
        for i, ep in enumerate(history):
            bar = "#" * int(ep * 30)
            print(f"    msg {i:2d}: {ep:.3f} |{bar}")

    return result


# =============================================================================
# Smoke Test
# =============================================================================


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Healthy agent needs no intervention
    cascade1 = HealingCascade(silent=True)
    messages1 = [
        "First step of analysis.",
        "Second step: examining data.",
        "Third step: formulating hypothesis.",
        "Fourth step: testing hypothesis.",
        "Fifth step: drawing conclusions.",
    ]
    result1 = cascade1.run(messages1)
    assert result1.success, "Healthy agent should succeed"
    assert result1.intervention_level == InterventionLevel.NONE, (
        f"Expected no intervention, got {result1.intervention_level}"
    )
    print("  Test 1: Healthy agent - PASSED")

    # Test 2: Stagnation detector detects repetition
    detector = StagnationDetector(
        window_size=3,
        threshold=0.3,
        critical_duration=2,
        silent=True,
    )
    # Prime with initial message
    detector.observe("Initial thought about the problem")
    # Feed repeated messages
    statuses = []
    for _ in range(8):
        status = detector.observe("Repeating the same thought")
        statuses.append(status)
    # Should detect stagnation at some point
    non_healthy = [s for s in statuses if s not in (HealthStatus.HEALTHY, HealthStatus.EXPLORING)]
    assert len(non_healthy) > 0, "Should detect stagnation in repeated messages"
    print("  Test 2: Stagnation detection - PASSED")

    # Test 3: Cascade diagnostic report
    cascade3 = HealingCascade(silent=True)
    messages3 = ["Same message"] * 10
    result3 = cascade3.run(messages3)
    assert result3.diagnostic.total_messages == 10
    assert len(result3.diagnostic.epiplexity_history) == 10
    assert len(result3.diagnostic.interventions_applied) > 0, "Should have interventions"
    print("  Test 3: Diagnostic report - PASSED")

    # Test 4: Epiplexity history recorded
    history = result3.diagnostic.epiplexity_history
    assert len(history) > 0, "Should have epiplexity history"
    assert all(0 <= e <= 1 for e in history), "Epiplexity should be in [0, 1]"
    print("  Test 4: Epiplexity history - PASSED")

    # Test 5: Diverse messages maintain high epiplexity
    cascade5 = HealingCascade(silent=True)
    diverse = [
        "Analyzing the problem from perspective A",
        "Now considering angle B completely different",
        "Testing hypothesis C with new data",
        "Surprising result from experiment D",
        "Synthesizing findings into conclusion E",
    ]
    result5 = cascade5.run(diverse)
    assert result5.success
    print("  Test 5: Diverse messages healthy - PASSED")

    print("\nSmoke tests passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 51: Epiplexity Healing Cascade")
    print("EpiplexityMonitor + RegenerativeSwarm + Autophagy + Cascade")
    print("=" * 60)

    demo_healthy_agent()
    demo_stagnant_agent()
    demo_critical_agent()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Epiplexity Healing Cascade detects and recovers from epistemic
stagnation through escalating interventions:

1. EpiplexityMonitor: Measures Bayesian surprise per message
   - High novelty = EXPLORING (making progress)
   - Low novelty + low perplexity = CONVERGING (task complete)
   - Low novelty + high perplexity = STAGNANT (stuck in loop)
   - Sustained low surprise = CRITICAL (deep stagnation)

2. Stage 1 (Autophagy): Prune stale context, hoping fresh
   context allows the agent to break out of the loop

3. Stage 2 (Regeneration): Kill the stuck worker and spawn
   a fresh one with a summary of what was tried

4. Stage 3 (Abort): Give up with a diagnostic report
   detailing the stagnation history

Key insight: Not all stagnation requires the same intervention.
Mild repetition might be fixed by clearing noise from context.
Deep stagnation needs a fresh start. Knowing which level to apply
prevents over-reacting to temporary hiccups.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
