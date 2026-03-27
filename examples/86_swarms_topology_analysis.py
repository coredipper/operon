"""
Example 86 — Swarms Topology Analysis
========================================

Demonstrates how Operon's epistemic theorems can analyze Swarms workflow
patterns for structural risks before deployment. No Swarms installation
required — the adapter operates on plain dict configurations.

Usage:
    python examples/86_swarms_topology_analysis.py
"""

from operon_ai.convergence import (
    analyze_external_topology,
    parse_swarm_topology,
    swarm_to_template,
)

# ---------------------------------------------------------------------------
# 1. Define Swarms-style workflow patterns as plain dicts
# ---------------------------------------------------------------------------

# A simple sequential pipeline (low risk)
sequential = parse_swarm_topology(
    pattern_name="SequentialWorkflow",
    agent_specs=[
        {"name": "planner", "role": "Planner"},
        {"name": "executor", "role": "Executor"},
    ],
    edges=[("planner", "executor")],
)

# A deep chain (higher risk — error amplification)
deep_chain = parse_swarm_topology(
    pattern_name="SequentialWorkflow",
    agent_specs=[
        {"name": f"stage_{i}", "role": f"Worker {i}"}
        for i in range(8)
    ],
    edges=[(f"stage_{i}", f"stage_{i+1}") for i in range(7)],
)

# A hierarchical swarm (moderate risk)
hierarchical = parse_swarm_topology(
    pattern_name="HierarchicalSwarm",
    agent_specs=[
        {"name": "manager", "role": "Manager"},
        {"name": "researcher", "role": "Researcher"},
        {"name": "coder", "role": "Developer"},
        {"name": "reviewer", "role": "Reviewer"},
    ],
    edges=[
        ("manager", "researcher"),
        ("manager", "coder"),
        ("manager", "reviewer"),
    ],
)

# ---------------------------------------------------------------------------
# 2. Analyze each topology with Operon's epistemic theorems
# ---------------------------------------------------------------------------

for name, topology in [("Sequential (2)", sequential), ("Deep Chain (8)", deep_chain), ("Hierarchical", hierarchical)]:
    result = analyze_external_topology(topology)
    print(f"=== {name} ===")
    print(f"  Pattern: {topology.pattern_name}")
    print(f"  Risk score: {result.risk_score:.3f}")
    print(f"  Recommended: {result.topology_advice.recommended_pattern}")
    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING: {w}")
    else:
        print("  No warnings")
    print()

# ---------------------------------------------------------------------------
# 3. Convert to PatternTemplate for Operon's library
# ---------------------------------------------------------------------------

template = swarm_to_template(hierarchical)
print(f"=== Template ===")
print(f"  ID: {template.template_id}")
print(f"  Topology: {template.topology}")
print(f"  Stages: {len(template.stage_specs)}")
print(f"  Fingerprint: shape={template.fingerprint.task_shape}, "
      f"subtasks={template.fingerprint.subtask_count}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert sequential.source == "swarms"
assert len(deep_chain.agents) == 8
result_deep = analyze_external_topology(deep_chain)
assert result_deep.risk_score > 0.0
assert result_deep.warnings  # deep chain should trigger warnings
result_simple = analyze_external_topology(sequential)
assert result_simple.risk_score <= result_deep.risk_score
assert template.topology in ("specialist_swarm", "skill_organism", "reviewer_gate", "single_worker")
print("\n--- all assertions passed ---")
