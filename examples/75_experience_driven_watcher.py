"""
Example 75 — Experience-Driven Watcher
=======================================

Demonstrates the watcher's experience pool: past intervention outcomes
inform future intervention decisions. Shows record_experience(),
retrieve_similar_experiences(), and recommend_intervention().

Usage:
    python examples/75_experience_driven_watcher.py
"""

from operon_ai import (
    InterventionKind,
    MockProvider,
    Nucleus,
    SkillStage,
    TaskFingerprint,
    WatcherComponent,
    WatcherConfig,
    skill_organism,
)
from operon_ai.patterns.watcher import ExperienceRecord

# ---------------------------------------------------------------------------
# 1. Create a watcher with pre-populated experience
# ---------------------------------------------------------------------------

watcher = WatcherComponent(config=WatcherConfig(max_retries_per_stage=2))

fp = TaskFingerprint(
    task_shape="sequential", tool_count=3, subtask_count=3,
    required_roles=("router", "planner"),
)

# Record past experiences: ESCALATE was successful for epistemic stagnation on "router"
watcher.record_experience(
    fingerprint=fp,
    stage_name="router",
    signal_category="epistemic",
    signal_detail={"status": "stagnant"},
    intervention_kind="escalate",
    intervention_reason="stagnant on fast model",
    outcome_success=True,
)
watcher.record_experience(
    fingerprint=fp,
    stage_name="router",
    signal_category="epistemic",
    signal_detail={"status": "stagnant"},
    intervention_kind="escalate",
    intervention_reason="stagnant on fast model",
    outcome_success=True,
)
# One retry that failed
watcher.record_experience(
    fingerprint=fp,
    stage_name="router",
    signal_category="epistemic",
    signal_detail={"status": "stagnant"},
    intervention_kind="retry",
    intervention_reason="attempted retry",
    outcome_success=False,
)

print("=== Experience Pool ===")
print(f"  Records: {len(watcher.experience_pool)}")

# ---------------------------------------------------------------------------
# 2. Query the experience pool
# ---------------------------------------------------------------------------

print("\n=== Similar Experiences for 'router' + 'epistemic' ===")
similar = watcher.retrieve_similar_experiences(
    stage_name="router",
    signal_category="epistemic",
    fingerprint=fp,
)
for exp in similar:
    print(f"  {exp.intervention_kind} → success={exp.outcome_success}")

# ---------------------------------------------------------------------------
# 3. Get a recommendation
# ---------------------------------------------------------------------------

recommended = watcher.recommend_intervention(
    stage_name="router",
    signal_category="epistemic",
    fingerprint=fp,
)
print(f"\n=== Recommendation ===")
print(f"  {recommended.value if recommended else 'None'}")

# ---------------------------------------------------------------------------
# 4. Run an organism with the experience-enriched watcher
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={
    "route the request": "EXECUTE: billing",
}))
deep = Nucleus(provider=MockProvider(responses={
    "route the request": "EXECUTE: billing (deep)",
    "plan the solution": "EXECUTE: solution planned",
}))

watcher.set_fingerprint(fp)

organism = skill_organism(
    stages=[
        SkillStage(name="intake", role="Normalizer", handler=lambda task: {"req": task}),
        SkillStage(name="router", role="Router", instructions="Route the request.", mode="fast"),
        SkillStage(name="planner", role="Planner", instructions="Plan the solution.", mode="deep"),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[watcher],
)

result = organism.run("Handle invoice dispute #4412.")

print(f"\n=== Organism Run ===")
print(f"  Stages: {len(result.stage_results)}")
print(f"  Final output: {result.final_output}")
print(f"  Watcher signals: {watcher.summary()['total_signals']}")
print(f"  Watcher interventions: {watcher.summary()['total_interventions']}")
print(f"  Experience pool size: {len(watcher.experience_pool)}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert recommended == InterventionKind.ESCALATE
assert len(similar) == 3
assert result.final_output is not None
print("\n--- all assertions passed ---")
