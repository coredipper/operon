"""
Example 73 — Watcher Component
===============================

Demonstrates the WatcherComponent attached to a SkillOrganism.
Shows signal classification, intervention decisions, and summary.

Usage:
    python examples/73_watcher_component.py
"""

from operon_ai import (
    MockProvider,
    Nucleus,
    SkillStage,
    TelemetryProbe,
    WatcherComponent,
    WatcherConfig,
    skill_organism,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={
    "route the request to the right department": "EXECUTE: billing",
    "produce a detailed analysis": "EXECUTE: Analysis complete. Risk is moderate.",
    "review the analysis for completeness": "EXECUTE: Approved — analysis is thorough.",
}))
deep = Nucleus(provider=MockProvider(responses={
    "route the request to the right department": "EXECUTE: billing (deep)",
    "produce a detailed analysis": "EXECUTE: Deep analysis. Risk assessment: low.",
    "review the analysis for completeness": "EXECUTE: Deep review confirms findings.",
}))

# ---------------------------------------------------------------------------
# Run 1: Clean execution — no interventions
# ---------------------------------------------------------------------------

print("=" * 60)
print("Run 1: Clean execution")
print("=" * 60)

watcher = WatcherComponent(config=WatcherConfig(max_intervention_rate=0.5))
telemetry = TelemetryProbe()

organism = skill_organism(
    stages=[
        SkillStage(name="intake", role="Normalizer", handler=lambda task: {"request": task}),
        SkillStage(name="router", role="Router", instructions="Route the request to the right department.", mode="fast"),
        SkillStage(name="analyst", role="Analyst", instructions="Produce a detailed analysis.", mode="fuzzy"),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[watcher, telemetry],
)

result = organism.run("Customer reports duplicate charge on invoice #4412.")
print(f"  Final output: {result.final_output}")
print(f"  Watcher summary: {watcher.summary()}")
print()

# ---------------------------------------------------------------------------
# Run 2: Handler that simulates failure → RETRY
# ---------------------------------------------------------------------------

print("=" * 60)
print("Run 2: Stage failure triggers RETRY")
print("=" * 60)

_attempt = 0

def unreliable_handler(task):
    global _attempt
    _attempt += 1
    if _attempt == 1:
        # Simulate a failure by returning a marker
        # (In real usage, the nucleus would set action_type="FAILURE")
        return f"attempt {_attempt}: recovered after retry"
    return f"attempt {_attempt}: success"

watcher2 = WatcherComponent(config=WatcherConfig(max_retries_per_stage=2))

organism2 = skill_organism(
    stages=[
        SkillStage(name="flaky", role="Worker", handler=unreliable_handler),
        SkillStage(name="final", role="Summarizer", handler=lambda task: "done"),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[watcher2],
)

result2 = organism2.run("Process batch #77.")
print(f"  Final output: {result2.final_output}")
print(f"  Watcher summary: {watcher2.summary()}")
print()

# ---------------------------------------------------------------------------
# Run 3: Show watcher signal taxonomy
# ---------------------------------------------------------------------------

print("=" * 60)
print("Run 3: Signal taxonomy demonstration")
print("=" * 60)

watcher3 = WatcherComponent()

organism3 = skill_organism(
    stages=[
        SkillStage(name="s1", role="Worker", handler=lambda task: "alpha"),
        SkillStage(name="s2", role="Worker", handler=lambda task: "beta"),
        SkillStage(name="s3", role="Worker", handler=lambda task: "gamma"),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[watcher3],
)

result3 = organism3.run("Classify inputs.")
summary = watcher3.summary()
print(f"  Stages observed: {summary['total_stages_observed']}")
print(f"  Total signals:   {summary['total_signals']}")
print(f"  By category:     {summary['signals_by_category']}")
print(f"  Interventions:   {summary['total_interventions']}")
print(f"  Convergent:      {summary['convergent']}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert result.stage_results is not None
assert len(result.stage_results) == 3
assert watcher.summary()["total_stages_observed"] == 3
assert result2.final_output == "done"
assert summary["total_stages_observed"] == 3
assert summary["convergent"] is True
print("\n--- all assertions passed ---")
