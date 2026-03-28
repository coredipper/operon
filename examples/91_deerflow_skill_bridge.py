"""
Example 91 — DeerFlow Skill Bridge
=====================================

Demonstrates bidirectional conversion between DeerFlow Markdown skills
and Operon PatternTemplates.

Usage:
    python examples/91_deerflow_skill_bridge.py
"""

from operon_ai.convergence import (
    skill_to_template,
    template_to_skill,
    parse_skill_frontmatter,
    extract_workflow_steps,
)

# ---------------------------------------------------------------------------
# 1. Parse a DeerFlow Markdown skill
# ---------------------------------------------------------------------------

skill_md = """---
name: competitive_analysis
description: Research competitors and produce a comparison report
version: 2.0
author: research_team
category: research
---

# Competitive Analysis

1. Identify top 5 competitors from industry databases
2. Gather pricing, features, and market positioning data
3. Analyze strengths and weaknesses of each competitor
4. Produce a structured comparison matrix
5. Write executive summary with recommendations
"""

print("=== Parse Skill ===")
frontmatter = parse_skill_frontmatter(skill_md)
steps = extract_workflow_steps(skill_md)
print(f"  Name: {frontmatter.get('name')}")
print(f"  Category: {frontmatter.get('category')}")
print(f"  Steps: {len(steps)}")
for i, step in enumerate(steps, 1):
    print(f"    {i}. {step}")
print()

# ---------------------------------------------------------------------------
# 2. Convert to PatternTemplate
# ---------------------------------------------------------------------------

template = skill_to_template(skill_md)
print("=== PatternTemplate ===")
print(f"  Name: {template.name}")
print(f"  Topology: {template.topology}")
print(f"  Stages: {len(template.stage_specs)}")
print(f"  Fingerprint: shape={template.fingerprint.task_shape}, "
      f"subtasks={template.fingerprint.subtask_count}")
print()

# ---------------------------------------------------------------------------
# 3. Export back to Markdown
# ---------------------------------------------------------------------------

exported = template_to_skill(template)
print("=== Exported Markdown ===")
print(exported[:300])
print("...")
print()

# ---------------------------------------------------------------------------
# 4. Roundtrip verification
# ---------------------------------------------------------------------------

roundtrip = skill_to_template(exported)
print("=== Roundtrip ===")
print(f"  Original stages: {len(template.stage_specs)}")
print(f"  Roundtrip stages: {len(roundtrip.stage_specs)}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert frontmatter["name"] == "competitive_analysis"
assert len(steps) == 5
assert template.topology == "skill_organism"
assert len(template.stage_specs) == 5
assert template.fingerprint.task_shape == "mixed"  # 5 steps → mixed
assert "---" in exported  # has frontmatter
assert len(roundtrip.stage_specs) == len(template.stage_specs)
print("\n--- all assertions passed ---")
