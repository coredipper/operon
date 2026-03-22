"""
Example 72 — Pattern Repository
================================

Demonstrates the PatternLibrary: register reusable collaboration templates,
fingerprint a new task, retrieve ranked matches, and show how success rates
improve template selection over time.

Usage:
    python examples/72_pattern_repository.py
"""

from operon_ai import PatternLibrary, PatternRunRecord, PatternTemplate, TaskFingerprint

# ---------------------------------------------------------------------------
# 1. Build a library with three templates
# ---------------------------------------------------------------------------

lib = PatternLibrary()

# A simple sequential pipeline template
sequential_fp = TaskFingerprint(
    task_shape="sequential",
    tool_count=2,
    subtask_count=3,
    required_roles=("writer", "reviewer"),
)
lib.register_template(PatternTemplate(
    template_id=lib.make_id(),
    name="Sequential Review Pipeline",
    topology="reviewer_gate",
    stage_specs=(
        {"name": "draft", "role": "Writer", "mode": "fuzzy"},
        {"name": "review", "role": "Reviewer", "mode": "fixed"},
    ),
    intervention_policy={"max_retries": 2},
    fingerprint=sequential_fp,
    tags=("content", "review"),
))

# A parallel specialist swarm template
parallel_fp = TaskFingerprint(
    task_shape="parallel",
    tool_count=5,
    subtask_count=4,
    required_roles=("researcher", "analyst", "writer"),
)
lib.register_template(PatternTemplate(
    template_id=lib.make_id(),
    name="Research Swarm",
    topology="specialist_swarm",
    stage_specs=(
        {"name": "research", "role": "Researcher"},
        {"name": "analysis", "role": "Analyst"},
        {"name": "synthesis", "role": "Writer"},
    ),
    intervention_policy={"escalate_on_stagnation": True},
    fingerprint=parallel_fp,
    tags=("research", "analysis"),
))

# A multi-stage skill organism template
mixed_fp = TaskFingerprint(
    task_shape="sequential",
    tool_count=4,
    subtask_count=4,
    required_roles=("router", "researcher", "strategist", "critic"),
)
organism_tmpl = PatternTemplate(
    template_id=lib.make_id(),
    name="Enterprise Review Organism",
    topology="skill_organism",
    stage_specs=(
        {"name": "intake", "role": "Router", "mode": "fast"},
        {"name": "research", "role": "Researcher", "mode": "fuzzy"},
        {"name": "strategy", "role": "Strategist", "mode": "deep"},
        {"name": "critique", "role": "Critic", "mode": "fuzzy"},
    ),
    intervention_policy={"max_retries": 1, "escalate_on_stagnation": True},
    fingerprint=mixed_fp,
    tags=("enterprise", "review"),
)
lib.register_template(organism_tmpl)

print("=== Library Summary ===")
print(lib.summary())
print()

# ---------------------------------------------------------------------------
# 2. Fingerprint a new task and retrieve ranked templates
# ---------------------------------------------------------------------------

new_task = TaskFingerprint(
    task_shape="sequential",
    tool_count=3,
    subtask_count=4,
    required_roles=("researcher", "strategist", "critic"),
    tags=("enterprise",),
)

print("=== Ranked Templates for New Task ===")
ranked = lib.top_templates_for(new_task)
for template, score in ranked:
    print(f"  {score:.4f}  {template.name} ({template.topology})")
print()

# ---------------------------------------------------------------------------
# 3. Record runs and show success rate influence
# ---------------------------------------------------------------------------

# Record some outcomes
for tmpl, score in ranked:
    if tmpl.name == "Enterprise Review Organism":
        tid = tmpl.template_id
        # 3 successful runs
        for _ in range(3):
            lib.record_run(PatternRunRecord(
                record_id=lib.make_id(),
                template_id=tid,
                fingerprint=new_task,
                success=True,
                latency_ms=1200.0,
                tokens_used=3500,
            ))
    elif tmpl.name == "Sequential Review Pipeline":
        tid = tmpl.template_id
        # 1 failed run
        lib.record_run(PatternRunRecord(
            record_id=lib.make_id(),
            template_id=tid,
            fingerprint=new_task,
            success=False,
            latency_ms=800.0,
            tokens_used=1500,
        ))

print("=== After Recording Runs ===")
for tmpl, score in ranked:
    sr = lib.success_rate(tmpl.template_id)
    sr_str = f"{sr:.0%}" if sr is not None else "n/a"
    print(f"  {tmpl.name}: success_rate={sr_str}")
print()

print("=== Re-ranked After Run Records ===")
re_ranked = lib.top_templates_for(new_task)
for template, score in re_ranked:
    print(f"  {score:.4f}  {template.name}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert len(ranked) == 3
assert re_ranked[0][0].name == "Enterprise Review Organism"
assert re_ranked[0][1] >= re_ranked[1][1]
print("\n--- all assertions passed ---")
