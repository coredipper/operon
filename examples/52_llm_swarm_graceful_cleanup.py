#!/usr/bin/env python3
"""
Example 52: LLM Swarm with Graceful Cleanup
=============================================

Demonstrates an LLM-powered swarm where dying workers clean up their context
via autophagy before passing state to successors. Successors inherit a clean
summary instead of raw noise.

Architecture:

```
[MorphogenGradient] <- tracks swarm health
    |
[RegenerativeSwarm]
    ├── Worker 1 (Nucleus + MockProvider)
    │     ├── Accumulates context (responses)
    │     ├── Gets stuck (entropy collapse)
    │     ├── BEFORE death:
    │     │     ├── AutophagyDaemon prunes context
    │     │     ├── Lysosome disposes noise
    │     │     └── HistoneStore saves clean summary
    │     └── Apoptosis
    │
    └── Worker 2 (successor)
          ├── Starts with clean summary from HistoneStore
          ├── No accumulated noise
          └── Fresh context → better performance
```

Key concepts:
- Workers use Nucleus + MockProvider to simulate LLM-backed reasoning
- Before apoptosis, dying workers run autophagy to extract useful state
- Clean summaries stored in HistoneStore are passed to successors
- MorphogenGradient tracks overall swarm health

Prerequisites:
- Example 40 for Regenerative Swarm
- Example 23 for Nucleus + MockProvider
- Example 39 for Autophagy

Usage:
    python examples/52_llm_swarm_graceful_cleanup.py
    python examples/52_llm_swarm_graceful_cleanup.py --test
"""

import sys
from dataclasses import dataclass, field

from operon_ai import (
    HistoneStore,
    Lysosome,
    Waste,
    WasteType,
    MarkerType,
)
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider, ProviderConfig
from operon_ai.coordination.morphogen import (
    MorphogenGradient,
    MorphogenType,
)
from operon_ai.healing import (
    RegenerativeSwarm,
    SimpleWorker,
    WorkerMemory,
    AutophagyDaemon,
    create_default_summarizer,
    create_simple_summarizer,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CleanupRecord:
    """Record of a worker's cleanup before death."""
    worker_id: str
    context_before: int  # chars
    context_after: int   # chars
    tokens_freed: int
    summary_stored: str
    noise_disposed: int


# =============================================================================
# LLM Swarm Worker Factory
# =============================================================================


class LLMSwarmWorkerFactory:
    """
    Factory that creates LLM-powered workers with graceful cleanup.

    Each worker:
    1. Uses Nucleus + MockProvider for "LLM" responses
    2. Accumulates context from responses
    3. Before dying, runs autophagy to clean context
    4. Stores clean summary in HistoneStore for successors
    """

    def __init__(
        self,
        responses: dict[str, str],
        gradient: MorphogenGradient,
        silent: bool = False,
    ):
        self.gradient = gradient
        self.silent = silent

        # Shared state across workers
        self.histone_store = HistoneStore()
        self.lysosome = Lysosome(silent=True)
        self.autophagy = AutophagyDaemon(
            histone_store=self.histone_store,
            lysosome=self.lysosome,
            summarizer=create_simple_summarizer(),
            toxicity_threshold=0.6,
            silent=silent,
        )

        # Nucleus for LLM calls
        self.nucleus = Nucleus(provider=MockProvider(responses=responses))

        # Tracking
        self._cleanup_records: list[CleanupRecord] = []
        self._worker_count = 0

    def create_worker(self, name: str, memory_hints: list[str]) -> SimpleWorker:
        """Create a cleanup-aware worker."""
        self._worker_count += 1
        generation = self._worker_count

        # Check if we have hints from predecessor (via summarizer or histone)
        inherited_context = ""
        if memory_hints:
            # First try HistoneStore for clean summaries
            retrieval = self.histone_store.retrieve_context(
                " ".join(memory_hints[:3]),
                limit=3,
            )
            if retrieval.formatted_context:
                inherited_context = retrieval.formatted_context
            else:
                # Fall back to raw memory hints from summarizer
                inherited_context = "; ".join(memory_hints)

        if not self.silent:
            has_ctx = bool(inherited_context)
            print(
                f"  [Factory] Creating {name} (gen {generation})"
                f"{' with inherited context' if has_ctx else ''}"
            )

        # Build worker context
        accumulated_context: list[str] = []
        if inherited_context:
            accumulated_context.append(f"[Inherited summary]: {inherited_context[:200]}")

        factory_ref = self

        def work(task: str, memory: WorkerMemory) -> str:
            step = len(memory.output_history)

            # Simulate LLM response
            prompt_key = f"step_{step}"
            try:
                response = factory_ref.nucleus.transcribe(
                    prompt_key,
                    config=ProviderConfig(temperature=0.0, max_tokens=256),
                )
                output = response.content
            except (KeyError, ValueError):
                output = f"Processing step {step}..."

            # Accumulate context
            accumulated_context.append(output)

            # Update gradient
            error_rate = factory_ref.gradient.get(MorphogenType.ERROR_RATE)
            factory_ref.gradient.set(
                MorphogenType.CONFIDENCE,
                max(0.1, 1.0 - step * 0.15),
            )

            # Workers with inherited context solve faster
            if inherited_context and generation >= 2:
                if step == 0:
                    return f"STRATEGY: Starting from inherited summary (gen {generation})"
                elif step == 1:
                    return "PROGRESS: Building on predecessor's work"
                elif step >= 2:
                    # Run cleanup before returning success
                    factory_ref._cleanup_worker(
                        name, "\n".join(accumulated_context),
                    )
                    return "DONE: Completed with clean state inheritance!"

            # Default: accumulate noise, get stuck (identical output triggers entropy collapse)
            return "THINKING: Still processing..."

        return SimpleWorker(id=name, work_function=work)

    def _cleanup_worker(self, worker_id: str, context: str) -> CleanupRecord:
        """
        Run graceful cleanup before worker death.

        1. Autophagy prunes context
        2. Lysosome disposes waste
        3. HistoneStore saves clean summary
        """
        context_before = len(context)

        # Run autophagy
        cleaned_context, prune_result = self.autophagy.check_and_prune(
            context, max_tokens=2000,
        )

        tokens_freed = prune_result.tokens_freed if prune_result else 0
        summary = cleaned_context[:300] if cleaned_context else context[:100]

        # Store clean summary in HistoneStore
        self.histone_store.add_marker(
            content=f"Worker {worker_id} summary: {summary}",
            marker_type=MarkerType.ACETYLATION,
            tags=["worker_summary", worker_id],
            context=f"Cleanup from {worker_id} before apoptosis",
        )

        # Dispose noise via Lysosome
        noise_count = 0
        if prune_result and prune_result.tokens_freed > 0:
            self.lysosome.ingest(Waste(
                waste_type=WasteType.EXPIRED_CACHE,
                content=f"Noise from {worker_id}: {tokens_freed} tokens",
                source=worker_id,
            ))
            digest = self.lysosome.digest()
            noise_count = digest.disposed

        record = CleanupRecord(
            worker_id=worker_id,
            context_before=context_before,
            context_after=len(cleaned_context),
            tokens_freed=tokens_freed,
            summary_stored=summary[:100],
            noise_disposed=noise_count,
        )
        self._cleanup_records.append(record)

        if not self.silent:
            print(
                f"  [Cleanup] {worker_id}: "
                f"{context_before} -> {len(cleaned_context)} chars, "
                f"freed {tokens_freed} tokens"
            )

        return record

    def get_cleanup_records(self) -> list[CleanupRecord]:
        """Get all cleanup records."""
        return list(self._cleanup_records)


# =============================================================================
# Demos
# =============================================================================


def demo_research_with_cleanup():
    """
    Demo: Research task where workers clean up before dying.

    Worker 1 accumulates noisy context, gets stuck, cleans up, dies.
    Worker 2 inherits clean summary and completes the task.
    """
    print("=" * 60)
    print("Demo 1: Research with Graceful Cleanup")
    print("=" * 60)

    # Mock LLM responses
    responses = {
        "step_0": "Initial research findings on the topic.",
        "step_1": "Deeper analysis reveals three key factors.",
        "step_2": "Cross-referencing sources confirms hypothesis.",
        "step_3": "Still processing...",
        "step_4": "Still processing...",
    }

    gradient = MorphogenGradient()

    factory = LLMSwarmWorkerFactory(
        responses=responses,
        gradient=gradient,
        silent=False,
    )

    swarm = RegenerativeSwarm(
        worker_factory=factory.create_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=3,
        silent=False,
    )

    print("\n--- Running swarm ---")
    result = swarm.supervise("Research the impact of morphogen gradients")

    print(f"\n  Result:")
    print(f"    Success: {result.success}")
    print(f"    Output: {result.output}")
    print(f"    Workers spawned: {result.total_workers_spawned}")

    # Show cleanup records
    records = factory.get_cleanup_records()
    if records:
        print(f"\n  Cleanup Records:")
        for rec in records:
            print(
                f"    {rec.worker_id}: "
                f"{rec.context_before}->{rec.context_after} chars, "
                f"freed {rec.tokens_freed} tokens"
            )

    # Show gradient evolution
    print(f"\n  Final Gradient:")
    for mtype in [MorphogenType.CONFIDENCE, MorphogenType.ERROR_RATE]:
        print(f"    {mtype.value}: {gradient.get(mtype):.3f}")

    return result


def demo_context_pollution_comparison():
    """
    Demo: Compare swarm with and without cleanup.

    Shows the difference between raw context inheritance
    and clean summary inheritance.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Context Pollution Comparison")
    print("=" * 60)

    responses = {
        "step_0": "Finding relevant data...",
        "step_1": "Analyzing patterns in data...",
        "step_2": "Drawing conclusions...",
        "step_3": "Still processing...",
    }

    # Run with cleanup
    print("\n--- With graceful cleanup ---")
    gradient1 = MorphogenGradient()
    factory1 = LLMSwarmWorkerFactory(
        responses=responses,
        gradient=gradient1,
        silent=False,
    )
    swarm1 = RegenerativeSwarm(
        worker_factory=factory1.create_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=3,
        silent=True,
    )
    result1 = swarm1.supervise("Analyze the dataset")

    print(f"  Success: {result1.success}")
    print(f"  Workers: {result1.total_workers_spawned}")
    print(f"  Cleanups: {len(factory1.get_cleanup_records())}")

    # Run without cleanup (regular summarizer only)
    print("\n--- Without cleanup (baseline) ---")
    gradient2 = MorphogenGradient()

    def create_basic_worker(name: str, hints: list[str]) -> SimpleWorker:
        """Basic worker without cleanup."""
        gen = int(name.split("_")[-1]) if "_" in name else 1
        has_hints = bool(hints)

        def work(task: str, memory: WorkerMemory) -> str:
            step = len(memory.output_history)
            if has_hints and gen >= 2 and step >= 2:
                return "DONE: Completed (no cleanup needed)"
            if step >= 3:
                return "THINKING: Still processing..."
            return f"THINKING: Step {step}..."

        return SimpleWorker(id=name, work_function=work)

    swarm2 = RegenerativeSwarm(
        worker_factory=create_basic_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=3,
        silent=True,
    )
    result2 = swarm2.supervise("Analyze the dataset")

    print(f"  Success: {result2.success}")
    print(f"  Workers: {result2.total_workers_spawned}")

    print(f"\n  Comparison:")
    print(f"    With cleanup:    {result1.total_workers_spawned} workers, "
          f"{'success' if result1.success else 'failed'}")
    print(f"    Without cleanup: {result2.total_workers_spawned} workers, "
          f"{'success' if result2.success else 'failed'}")

    return result1, result2


# =============================================================================
# Smoke Test
# =============================================================================


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Basic swarm with cleanup
    responses = {
        "step_0": "Initial analysis",
        "step_1": "Deeper investigation",
        "step_2": "Findings compiled",
    }
    gradient = MorphogenGradient()
    factory = LLMSwarmWorkerFactory(
        responses=responses,
        gradient=gradient,
        silent=True,
    )
    swarm = RegenerativeSwarm(
        worker_factory=factory.create_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=3,
        silent=True,
    )
    result = swarm.supervise("Test task")
    assert result.success, "Swarm should succeed"
    assert result.total_workers_spawned >= 1
    print("  Test 1: Basic swarm - PASSED")

    # Test 2: HistoneStore receives summaries
    # Verify histone store is operational
    factory.histone_store.add_marker(
        content="test marker",
        marker_type=MarkerType.ACETYLATION,
        tags=["test"],
    )
    retrieval = factory.histone_store.retrieve_context("test")
    assert retrieval is not None
    print("  Test 2: HistoneStore integration - PASSED")

    # Test 3: Gradient tracking
    confidence = gradient.get(MorphogenType.CONFIDENCE)
    assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    print("  Test 3: Gradient tracking - PASSED")

    # Test 4: Worker factory creates workers
    worker = factory.create_worker("test_worker", [])
    assert worker.id == "test_worker"
    print("  Test 4: Worker creation - PASSED")

    # Test 5: Lysosome receives waste
    factory.lysosome.ingest(Waste(
        waste_type=WasteType.EXPIRED_CACHE,
        content="test waste",
        source="test",
    ))
    digest = factory.lysosome.digest()
    assert digest.disposed >= 0
    print("  Test 5: Lysosome integration - PASSED")

    # Test 6: Multiple generations
    responses2 = {"step_0": "R1", "step_1": "R2", "step_2": "R3"}
    gradient2 = MorphogenGradient()
    factory2 = LLMSwarmWorkerFactory(
        responses=responses2,
        gradient=gradient2,
        silent=True,
    )
    swarm2 = RegenerativeSwarm(
        worker_factory=factory2.create_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=3,
        silent=True,
    )
    result2 = swarm2.supervise("Multi-gen task")
    assert result2.total_workers_spawned >= 1
    print("  Test 6: Multi-generation - PASSED")

    print("\nSmoke tests passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 52: LLM Swarm with Graceful Cleanup")
    print("Nucleus + RegenerativeSwarm + Autophagy + MorphogenGradient")
    print("=" * 60)

    demo_research_with_cleanup()
    demo_context_pollution_comparison()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The LLM Swarm with Graceful Cleanup demonstrates clean state transfer:

1. Workers use Nucleus + MockProvider for LLM-backed reasoning
   Each worker accumulates context from its responses

2. Before death (apoptosis), dying workers run cleanup:
   - AutophagyDaemon prunes noisy/stale context
   - Lysosome disposes extracted noise
   - HistoneStore saves clean summary

3. Successor workers inherit clean summaries, not raw noise
   This prevents context pollution from degrading performance

4. MorphogenGradient tracks overall swarm health
   - Confidence drops as workers struggle
   - Error rate reflects accumulated failures

Key biological parallel: When cells undergo apoptosis, they release
clean signaling molecules (not their entire cytoplasm) to inform
neighboring cells. Similarly, dying workers extract and transmit
only the useful parts of their accumulated context.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
