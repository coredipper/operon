"""
Example 74 — Adaptive Assembly
================================

Demonstrates the adaptive assembly loop: task fingerprinting, template
selection from PatternLibrary, automatic organism construction, execution,
and outcome recording that feeds back into future selections.

Usage:
    python examples/74_adaptive_assembly.py
"""

from operon_ai import (
    MockProvider,
    Nucleus,
    PatternLibrary,
    PatternTemplate,
    TaskFingerprint,
    adaptive_skill_organism,
)

# ---------------------------------------------------------------------------
# 1. Build library with templates
# ---------------------------------------------------------------------------

lib = PatternLibrary()

lib.register_template(PatternTemplate(
    template_id="review",
    name="Sequential Review Pipeline",
    topology="skill_organism",
    stage_specs=(
        {"name": "draft", "role": "Writer", "mode": "fuzzy"},
        {"name": "review", "role": "Reviewer", "mode": "fast"},
    ),
    intervention_policy={"max_retries": 2},
    fingerprint=TaskFingerprint(
        task_shape="sequential", tool_count=2, subtask_count=2,
        required_roles=("writer", "reviewer"),
    ),
    tags=("content", "review"),
))

lib.register_template(PatternTemplate(
    template_id="enterprise",
    name="Enterprise Analysis Organism",
    topology="skill_organism",
    stage_specs=(
        {"name": "intake", "role": "Normalizer"},
        {"name": "research", "role": "Researcher", "mode": "fuzzy"},
        {"name": "strategy", "role": "Strategist", "mode": "deep"},
    ),
    intervention_policy={"max_retries": 1, "escalate_on_stagnation": True},
    fingerprint=TaskFingerprint(
        task_shape="sequential", tool_count=4, subtask_count=3,
        required_roles=("researcher", "strategist"),
    ),
    tags=("enterprise", "analysis"),
))

lib.register_template(PatternTemplate(
    template_id="swarm",
    name="Research Swarm",
    topology="specialist_swarm",
    stage_specs=(
        {"name": "analyst", "role": "Analyst"},
        {"name": "writer", "role": "Writer"},
    ),
    intervention_policy={},
    fingerprint=TaskFingerprint(
        task_shape="parallel", tool_count=5, subtask_count=4,
        required_roles=("analyst", "writer"),
    ),
    tags=("research",),
))

# ---------------------------------------------------------------------------
# 2. Create adaptive organism for a sequential task
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

fp = TaskFingerprint(
    task_shape="sequential", tool_count=3, subtask_count=3,
    required_roles=("researcher", "strategist"),
    tags=("enterprise",),
)

adaptive = adaptive_skill_organism(
    "Prepare a Q4 earnings analysis.",
    fingerprint=fp,
    library=lib,
    fast_nucleus=fast,
    deep_nucleus=deep,
    handlers={
        "intake": lambda task: {"parsed": task},
        "research": lambda task, state, outputs: "Revenue up 12%. Margins stable.",
        "strategy": lambda task, state, outputs: "Recommend: hold position, monitor margins.",
    },
)

print("=== Template Selected ===")
print(f"  {adaptive.template.name} (score={adaptive.template_score:.4f})")
print(f"  Topology: {adaptive.template.topology}")
print()

# ---------------------------------------------------------------------------
# 3. Run and record
# ---------------------------------------------------------------------------

result = adaptive.run("Prepare a Q4 earnings analysis.")

print("=== Run 1 Result ===")
print(f"  Success: {result.record.success}")
print(f"  Final output: {result.run_result.final_output}")
print(f"  Stages: {len(result.run_result.stage_results)}")
print(f"  Watcher: {result.watcher_summary}")
print(f"  Library records: {lib.summary()['record_count']}")
print()

# ---------------------------------------------------------------------------
# 4. Second run — library feedback
# ---------------------------------------------------------------------------

result2 = adaptive.run("Review the Q3 compliance report.")

print("=== Run 2 Result ===")
print(f"  Success: {result2.record.success}")
print(f"  Library records: {lib.summary()['record_count']}")
sr = lib.success_rate("enterprise")
print(f"  Enterprise success rate: {sr:.0%}" if sr else "  No records yet")
print()

# ---------------------------------------------------------------------------
# 5. Show ranking shift
# ---------------------------------------------------------------------------

print("=== Re-ranked Templates ===")
for tmpl, score in lib.top_templates_for(fp):
    sr = lib.success_rate(tmpl.template_id)
    sr_str = f"{sr:.0%}" if sr is not None else "n/a"
    print(f"  {score:.4f}  {tmpl.name} (success={sr_str})")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert adaptive.template.template_id == "enterprise"
assert result.record.success is True
assert lib.summary()["record_count"] == 2
print("\n--- all assertions passed ---")
