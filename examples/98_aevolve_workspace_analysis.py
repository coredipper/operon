"""
Example 98 -- A-Evolve Workspace Analysis
============================================

Demonstrates parsing an A-Evolve workspace manifest, analyzing its
single-agent topology, and mapping skills to cognitive stages.
No a-evolve installation required.

Usage:
    python examples/98_aevolve_workspace_analysis.py
"""

from operon_ai.convergence import (
    aevolve_skills_to_stages,
    aevolve_to_template,
    analyze_external_topology,
    parse_aevolve_workspace,
    seed_library_from_aevolve,
)
from operon_ai.patterns.repository import PatternLibrary

# ---------------------------------------------------------------------------
# 1. Define an A-Evolve workspace manifest
# ---------------------------------------------------------------------------

workspace_manifest = {
    "name": "swe-agent",
    "entrypoints": {"solve": "solve.py"},
    "skills": ["bash_exec", "file_edit", "git_ops"],
    "memory": {
        "episodic": "memory/episodic.jsonl",
        "semantic": "memory/semantic.jsonl",
    },
    "evolution": {
        "algorithm": "adaptive_evolve",
        "gate": "holdout",
    },
}

# ---------------------------------------------------------------------------
# 2. Parse and analyze topology
# ---------------------------------------------------------------------------

topology = parse_aevolve_workspace(workspace_manifest)
print("=== A-Evolve Topology ===")
print(f"  Source: {topology.source}")
print(f"  Pattern: {topology.pattern_name}")
print(f"  Agents: {len(topology.agents)} (single workspace)")
print(f"  Edges: {len(topology.edges)} (none -- single agent)")
print(f"  Skills: {topology.metadata.get('skills', [])}")
print(f"  Evolution: {topology.metadata.get('evolution', {})}")
print()

result = analyze_external_topology(topology)
print("=== Epistemic Analysis ===")
print(f"  Risk score: {result.risk_score:.3f}")
print(f"  Recommended pattern: {result.topology_advice.recommended_pattern}")
if result.warnings:
    for w in result.warnings:
        print(f"  WARNING: {w}")
else:
    print("  No warnings")
print()

# ---------------------------------------------------------------------------
# 3. Map skills to cognitive stages
# ---------------------------------------------------------------------------

skill_dicts = [
    {"name": "bash_exec", "description": "Execute bash commands", "category": "execution"},
    {"name": "file_edit", "description": "Edit source files", "category": "editing"},
    {"name": "git_ops", "description": "Git operations", "category": "execution"},
    {"name": "code_review", "description": "Review code changes", "category": "review"},
]
stages = aevolve_skills_to_stages(skill_dicts)
print("=== Skills-to-Stages Mapping ===")
for stage in stages:
    print(f"  {stage.name}: category={stage.role}, mode={stage.mode}, "
          f"cognitive={stage.cognitive_mode.value}")
print()

# ---------------------------------------------------------------------------
# 4. Convert to template
# ---------------------------------------------------------------------------

template = aevolve_to_template(workspace_manifest)
print("=== Pattern Template ===")
print(f"  ID: {template.template_id}")
print(f"  Name: {template.name}")
print(f"  Topology: {template.topology}")
print(f"  Stages: {len(template.stage_specs)}")
print(f"  Tags: {template.tags}")
print()

# ---------------------------------------------------------------------------
# 5. Seed a library from multiple workspaces
# ---------------------------------------------------------------------------

library = PatternLibrary()
workspaces = [
    workspace_manifest,
    {**workspace_manifest, "name": "code-agent", "skills": ["bash_exec", "file_edit"]},
]
count = seed_library_from_aevolve(library, workspaces)
print("=== Seeded Library ===")
print(f"  Templates registered: {count}")
print(f"  Library summary: {library.summary()}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert topology.source == "aevolve"
assert len(topology.agents) == 1
assert topology.edges == ()
assert result.risk_score >= 0.0
# execution skills should be action-oriented (fuzzy)
exec_stages = [s for s in stages if s.name == "bash_exec"]
assert exec_stages and exec_stages[0].mode == "fuzzy"
# review skills should be observational (fixed)
review_stages = [s for s in stages if s.name == "code_review"]
assert review_stages and review_stages[0].mode == "fixed"
assert template.topology == "single_worker"
assert count == 2
print("\n--- all assertions passed ---")
