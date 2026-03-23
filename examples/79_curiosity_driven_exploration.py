"""
Example 79 — Curiosity-Driven Exploration
==========================================

Demonstrates curiosity signals in the WatcherComponent. When epiplexity
status is EXPLORING (high novelty), the watcher emits a curiosity signal
that can trigger ESCALATE on fast models.

Usage:
    python examples/79_curiosity_driven_exploration.py
"""

from operon_ai import (
    MockProvider,
    Nucleus,
    SkillStage,
    WatcherComponent,
    WatcherConfig,
    skill_organism,
)

# ---------------------------------------------------------------------------
# 1. Setup with curiosity threshold
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={
    "classify the input": "EXECUTE: billing",
    "analyze in depth": "EXECUTE: Deep analysis complete.",
}))
deep = Nucleus(provider=MockProvider(responses={
    "classify the input": "EXECUTE: billing (deep analysis)",
    "analyze in depth": "EXECUTE: Comprehensive deep analysis.",
}))

watcher = WatcherComponent(
    config=WatcherConfig(curiosity_escalation_threshold=0.5),
    # Note: without an EpiplexityMonitor, curiosity signals won't fire.
    # In production, attach a real EpiplexityMonitor here.
)

organism = skill_organism(
    stages=[
        SkillStage(name="intake", role="Normalizer", handler=lambda task: {"parsed": task}),
        SkillStage(name="classifier", role="Router", instructions="Classify the input.", mode="fast"),
        SkillStage(name="analyst", role="Analyst", instructions="Analyze in depth.", mode="deep"),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[watcher],
)

# ---------------------------------------------------------------------------
# 2. Run — no monitor attached, so no curiosity signals
# ---------------------------------------------------------------------------

result = organism.run("Customer reports unexpected charge on account #1234.")

print("=== Run Without Monitor ===")
print(f"  Final output: {result.final_output}")
print(f"  Curiosity signals: {len([s for s in watcher.signals if s.source == 'curiosity'])}")
print(f"  Total signals: {watcher.summary()['total_signals']}")
print(f"  Interventions: {watcher.summary()['total_interventions']}")
print()

# ---------------------------------------------------------------------------
# 3. Demonstrate curiosity signal mechanism
# ---------------------------------------------------------------------------

# Manually demonstrate what happens when curiosity fires
from operon_ai.patterns.watcher import WatcherSignal, SignalCategory

print("=== Curiosity Signal Demonstration ===")
print("  When EpiplexityMonitor detects EXPLORING status:")
print("  → Curiosity signal emitted (category=EPISTEMIC, source=curiosity)")
print("  → If value > threshold AND model=fast → ESCALATE intervention")
print("  → Deep model investigates the novel input more thoroughly")
print()
print(f"  Current threshold: {watcher.config.curiosity_escalation_threshold}")
print(f"  Signal taxonomy: epistemic (curiosity is a sub-source of epistemic)")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert result.final_output is not None
assert len(result.stage_results) == 3
assert watcher.config.curiosity_escalation_threshold == 0.5
print("\n--- all assertions passed ---")
