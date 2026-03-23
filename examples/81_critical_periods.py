"""
Example 81 — Critical Periods and Teacher-Learner Scaffolding
==============================================================

Demonstrates critical periods that close as organisms mature, and
teacher-learner scaffolding where a mature organism guides a younger one.

Usage:
    python examples/81_critical_periods.py
"""

from operon_ai import (
    DevelopmentController,
    DevelopmentConfig,
    DevelopmentalStage,
    CriticalPeriod,
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    SocialLearning,
    TaskFingerprint,
    Telomere,
)

# ---------------------------------------------------------------------------
# 1. Create a mature teacher organism
# ---------------------------------------------------------------------------

lib_teacher = PatternLibrary()
lib_teacher.register_template(PatternTemplate(
    template_id="basic",
    name="Basic Pipeline",
    topology="skill_organism",
    stage_specs=({"name": "s1", "role": "Worker"},),
    intervention_policy={},
    fingerprint=TaskFingerprint("sequential", 1, 1, ("worker",)),
    tags=("beginner",),
))
lib_teacher.register_template(PatternTemplate(
    template_id="advanced",
    name="Advanced Multi-Stage",
    topology="skill_organism",
    stage_specs=(
        {"name": "research", "role": "Researcher", "min_stage": "adolescent"},
        {"name": "strategy", "role": "Strategist"},
    ),
    intervention_policy={},
    fingerprint=TaskFingerprint("sequential", 3, 3, ("researcher", "strategist")),
    tags=("advanced",),
))
# Record successes for both
for tid in ["basic", "advanced"]:
    for _ in range(3):
        lib_teacher.record_run(PatternRunRecord(
            record_id=lib_teacher.make_id(), template_id=tid,
            fingerprint=TaskFingerprint("sequential", 2, 2, ("worker",)),
            success=True, latency_ms=100.0, tokens_used=500,
        ))

teacher_telomere = Telomere(max_operations=100)
teacher_telomere.start()
teacher_dev = DevelopmentController(telomere=teacher_telomere)
for _ in range(80):  # Tick to MATURE
    teacher_dev.tick()

sl_teacher = SocialLearning(organism_id="teacher", library=lib_teacher)

print(f"Teacher stage: {teacher_dev.stage.value}")
print(f"Teacher templates: {lib_teacher.summary()['template_count']}")
print()

# ---------------------------------------------------------------------------
# 2. Create an embryonic learner organism
# ---------------------------------------------------------------------------

lib_learner = PatternLibrary()
learner_telomere = Telomere(max_operations=100)
learner_telomere.start()
learner_dev = DevelopmentController(
    telomere=learner_telomere,
    critical_periods=(
        CriticalPeriod("rapid_adoption", DevelopmentalStage.EMBRYONIC, DevelopmentalStage.JUVENILE,
                        "accept templates freely from teachers"),
    ),
)

sl_learner = SocialLearning(organism_id="learner", library=lib_learner)

print(f"Learner stage: {learner_dev.stage.value}")
print(f"Critical period 'rapid_adoption' open: {learner_dev.is_critical_period_open('rapid_adoption')}")
print()

# ---------------------------------------------------------------------------
# 3. Teacher scaffolds learner
# ---------------------------------------------------------------------------

result = sl_teacher.scaffold_learner(
    sl_learner,
    teacher_stage=teacher_dev.stage.value,
    learner_stage=learner_dev.stage.value,
)

print("=== Scaffolding (EMBRYONIC learner) ===")
print(f"  Adopted: {result.adoption.adopted_template_ids}")
print(f"  Filtered out (too advanced): {result.templates_filtered_out}")
print(f"  Plasticity bonus: {result.plasticity_bonus:.2f}")
print()

# ---------------------------------------------------------------------------
# 4. Tick learner to ADOLESCENT, scaffold again
# ---------------------------------------------------------------------------

for _ in range(40):
    learner_dev.tick()

print(f"Learner stage after ticking: {learner_dev.stage.value}")
print(f"Critical period 'rapid_adoption' open: {learner_dev.is_critical_period_open('rapid_adoption')}")

result2 = sl_teacher.scaffold_learner(
    sl_learner,
    teacher_stage=teacher_dev.stage.value,
    learner_stage=learner_dev.stage.value,
)

print(f"\n=== Scaffolding (ADOLESCENT learner) ===")
print(f"  Adopted: {result2.adoption.adopted_template_ids}")
print(f"  Filtered out: {result2.templates_filtered_out}")
print()

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert teacher_dev.stage == DevelopmentalStage.MATURE
assert learner_dev.stage in (DevelopmentalStage.ADOLESCENT, DevelopmentalStage.MATURE)
assert "basic" in result.adoption.adopted_template_ids
assert learner_dev.is_critical_period_open("rapid_adoption") is False
print("--- all assertions passed ---")
