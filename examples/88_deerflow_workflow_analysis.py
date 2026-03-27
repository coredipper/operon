"""
Example 88 — DeerFlow Workflow Analysis
==========================================

Demonstrates analyzing a DeerFlow 2.0 session config with Operon's epistemic
theorems. No DeerFlow or LangChain installation required.

Usage:
    python examples/88_deerflow_workflow_analysis.py
"""

from operon_ai.convergence import (
    analyze_external_topology,
    deerflow_skills_to_stages,
    deerflow_to_template,
    parse_deerflow_session,
)

# ---------------------------------------------------------------------------
# 1. Define a DeerFlow-style session config
# ---------------------------------------------------------------------------

session_config = {
    "assistant_id": "lead_agent",
    "skills": ["web_research", "report_generation", "code_execution"],
    "sub_agents": [
        {"name": "researcher", "role": "researcher", "skills": ["web_search", "summarize"]},
        {"name": "coder", "role": "developer", "skills": ["python", "testing"]},
        {"name": "reviewer", "role": "reviewer", "skills": ["code_review", "security"]},
    ],
    "recursion_limit": 100,
    "sandbox": "docker",
    "config": {"thinking_enabled": True},
}

# ---------------------------------------------------------------------------
# 2. Parse and analyze
# ---------------------------------------------------------------------------

topology = parse_deerflow_session(session_config)
print("=== DeerFlow Topology ===")
print(f"  Source: {topology.source}")
print(f"  Agents: {len(topology.agents)} (lead + {len(topology.agents) - 1} sub-agents)")
print(f"  Sandbox: {topology.metadata.get('sandbox', 'unknown')}")
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
# 3. Map skills to stages
# ---------------------------------------------------------------------------

skill_dicts = [
    {"name": "web_research", "description": "Search and summarize", "category": "research"},
    {"name": "code_gen", "description": "Generate Python code", "category": "code"},
    {"name": "quality_check", "description": "Review for bugs", "category": "verification"},
]
stages = deerflow_skills_to_stages(skill_dicts)
print("=== Skill Stages ===")
for stage in stages:
    print(f"  {stage.name}: mode={stage.mode}")
print()

# ---------------------------------------------------------------------------
# 4. Convert to template
# ---------------------------------------------------------------------------

template = deerflow_to_template(session_config)
print("=== Pattern Template ===")
print(f"  ID: {template.template_id}")
print(f"  Topology: {template.topology}")
print(f"  Stages: {len(template.stage_specs)}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert topology.source == "deerflow"
assert topology.metadata.get("sandbox") == "docker"
assert len(topology.agents) == 4  # lead + 3 sub-agents
assert result.risk_score >= 0.0
# Research skill should be observational (fixed mode)
research_stages = [s for s in stages if s.name == "web_research"]
assert research_stages and research_stages[0].mode == "fixed"
# Code skill should be action-oriented (fuzzy mode)
code_stages = [s for s in stages if s.name == "code_gen"]
assert code_stages and code_stages[0].mode == "fuzzy"
assert template.topology in ("specialist_swarm", "skill_organism", "single_worker")
print("\n--- all assertions passed ---")
