"""
Example 87 — AnimaWorks Role Mapping
=======================================

Demonstrates mapping AnimaWorks organizational configs (supervisor hierarchies,
role templates) to Operon's typed stage system. No AnimaWorks installation
required.

Usage:
    python examples/87_animaworks_role_mapping.py
"""

from operon_ai.convergence import (
    animaworks_roles_to_stages,
    animaworks_to_template,
    parse_animaworks_org,
)

# ---------------------------------------------------------------------------
# 1. Define an AnimaWorks-style organization
# ---------------------------------------------------------------------------

org_config = {
    "name": "engineering_team",
    "supervisor": {"name": "tech_lead", "role": "manager"},
    "agents": [
        {"name": "backend_dev", "role": "engineer", "skills": ["python", "api_design"]},
        {"name": "frontend_dev", "role": "engineer", "skills": ["react", "css"]},
        {"name": "qa_analyst", "role": "reviewer", "skills": ["testing", "security"]},
        {"name": "tech_writer", "role": "writer", "skills": ["documentation"]},
    ],
    "communication": "hierarchical",
}

# ---------------------------------------------------------------------------
# 2. Parse into ExternalTopology
# ---------------------------------------------------------------------------

topology = parse_animaworks_org(org_config)
print("=== External Topology ===")
print(f"  Source: {topology.source}")
print(f"  Pattern: {topology.pattern_name}")
print(f"  Agents: {len(topology.agents)}")
print(f"  Edges: {len(topology.edges)}")
for src, dst in topology.edges:
    print(f"    {src} → {dst}")
print()

# ---------------------------------------------------------------------------
# 3. Map roles to SkillStages with cognitive modes
# ---------------------------------------------------------------------------

roles = [a for a in org_config["agents"]] + [org_config["supervisor"]]
stages = animaworks_roles_to_stages(roles)
print("=== Skill Stages ===")
for stage in stages:
    print(f"  {stage.name}: mode={stage.mode}, role={stage.role}")
print()

# ---------------------------------------------------------------------------
# 4. Convert to PatternTemplate
# ---------------------------------------------------------------------------

template = animaworks_to_template(org_config)
print("=== Pattern Template ===")
print(f"  ID: {template.template_id}")
print(f"  Topology: {template.topology}")
print(f"  Stages: {len(template.stage_specs)}")
print(f"  Tags: {template.tags}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert topology.source == "animaworks"
assert len(topology.agents) == 5  # 4 agents + supervisor
assert len(topology.edges) == 4  # supervisor -> each agent
assert all(s.mode in ("fixed", "fuzzy") for s in stages)
# Reviewer should be observational (fixed mode)
reviewer_stages = [s for s in stages if "qa" in s.name]
assert reviewer_stages and reviewer_stages[0].mode == "fixed"
assert template.topology in ("specialist_swarm", "skill_organism")
print("\n--- all assertions passed ---")
