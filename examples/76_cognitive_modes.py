"""
Example 76 — Cognitive Modes
=============================

Demonstrates CognitiveMode annotations on SkillStage: System A (observational)
vs System B (action-oriented). The watcher detects mode mismatches and reports
the A/B balance across the run.

Usage:
    python examples/76_cognitive_modes.py
"""

from operon_ai import (
    CognitiveMode,
    MockProvider,
    Nucleus,
    SkillStage,
    WatcherComponent,
    skill_organism,
)
from operon_ai.patterns.types import resolve_cognitive_mode

# ---------------------------------------------------------------------------
# 1. Build stages with explicit cognitive mode annotations
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={
    "classify the input": "EXECUTE: billing",
    "plan the resolution": "EXECUTE: escalate to supervisor",
}))
deep = Nucleus(provider=MockProvider(responses={
    "classify the input": "EXECUTE: billing (deep)",
    "plan the resolution": "EXECUTE: detailed resolution plan",
}))

stages = [
    SkillStage(
        name="intake",
        role="Normalizer",
        handler=lambda task: {"parsed": task},
        cognitive_mode=CognitiveMode.OBSERVATIONAL,
    ),
    SkillStage(
        name="classifier",
        role="Router",
        instructions="Classify the input.",
        mode="fast",
        cognitive_mode=CognitiveMode.OBSERVATIONAL,
    ),
    SkillStage(
        name="planner",
        role="Planner",
        instructions="Plan the resolution.",
        mode="deep",
        cognitive_mode=CognitiveMode.ACTION_ORIENTED,
    ),
]

# ---------------------------------------------------------------------------
# 2. Show resolved modes
# ---------------------------------------------------------------------------

print("=== Cognitive Mode Annotations ===")
for s in stages:
    resolved = resolve_cognitive_mode(s)
    print(f"  {s.name}: mode={s.mode}, cognitive_mode={resolved.value}")
print()

# ---------------------------------------------------------------------------
# 3. Run with watcher
# ---------------------------------------------------------------------------

watcher = WatcherComponent()

organism = skill_organism(
    stages=stages,
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[watcher],
)

result = organism.run("Customer reports duplicate charge.")

print("=== Run Result ===")
print(f"  Final output: {result.final_output}")
print(f"  Stages: {len(result.stage_results)}")
print()

# ---------------------------------------------------------------------------
# 4. Mode balance from watcher
# ---------------------------------------------------------------------------

balance = watcher.mode_balance()
print("=== Mode Balance ===")
print(f"  Observational (System A): {balance['observational']}")
print(f"  Action-Oriented (System B): {balance['action_oriented']}")
print(f"  Balance ratio: {balance['balance_ratio']:.2f}")
print(f"  Mismatches: {balance['mismatches']}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert balance["observational"] == 2
assert balance["action_oriented"] == 1
assert balance["mismatches"] == 0
assert result.final_output is not None
print("\n--- all assertions passed ---")
