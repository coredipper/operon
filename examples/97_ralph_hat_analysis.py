"""
Example 97 -- Ralph Hat Analysis
==================================

Demonstrates parsing a Ralph hat-based config, analyzing its event-driven
topology with Operon's epistemic theorems, and mapping hats to cognitive
stages. No ralph-orchestrator installation required.

Usage:
    python examples/97_ralph_hat_analysis.py
"""

from operon_ai.convergence import (
    analyze_external_topology,
    parse_ralph_config,
    ralph_hats_to_stages,
    ralph_to_template,
)

# ---------------------------------------------------------------------------
# 1. Define a Ralph-style config
# ---------------------------------------------------------------------------

ralph_config = {
    "backend": "claude",
    "hats": [
        {"name": "coder", "pattern": "code-assist"},
        {"name": "debugger", "pattern": "debug"},
        {"name": "reviewer", "pattern": "review"},
    ],
    "events": [
        {"from": "coder", "event": "code.failure", "to": "debugger"},
        {"from": "debugger", "event": "fix.complete", "to": "reviewer"},
    ],
    "backpressure": ["tests", "lint", "typecheck"],
    "iteration_limit": 10,
}

# ---------------------------------------------------------------------------
# 2. Parse and analyze topology
# ---------------------------------------------------------------------------

topology = parse_ralph_config(ralph_config)
print("=== Ralph Topology ===")
print(f"  Source: {topology.source}")
print(f"  Pattern: {topology.pattern_name}")
print(f"  Hats: {len(topology.agents)}")
print(f"  Events: {len(topology.edges)}")
print(f"  Backend: {topology.metadata.get('backend', 'unknown')}")
print(f"  Backpressure: {topology.metadata.get('backpressure', [])}")
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
# 3. Map hats to cognitive stages
# ---------------------------------------------------------------------------

stages = ralph_hats_to_stages(ralph_config["hats"])
print("=== Hat-to-Stage Mapping ===")
for stage in stages:
    print(f"  {stage.name}: pattern={stage.role}, mode={stage.mode}, "
          f"cognitive={stage.cognitive_mode.value}")
print()

# ---------------------------------------------------------------------------
# 4. Convert to template
# ---------------------------------------------------------------------------

template = ralph_to_template(ralph_config)
print("=== Pattern Template ===")
print(f"  ID: {template.template_id}")
print(f"  Name: {template.name}")
print(f"  Topology: {template.topology}")
print(f"  Stages: {len(template.stage_specs)}")
print(f"  Tags: {template.tags}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert topology.source == "ralph"
assert topology.metadata.get("backend") == "claude"
assert len(topology.agents) == 3
assert len(topology.edges) == 2
assert result.risk_score >= 0.0
# code-assist should be action-oriented (fuzzy)
coder_stages = [s for s in stages if s.name == "coder"]
assert coder_stages and coder_stages[0].mode == "fuzzy"
# review should be observational (fixed)
reviewer_stages = [s for s in stages if s.name == "reviewer"]
assert reviewer_stages and reviewer_stages[0].mode == "fixed"
assert template.topology in ("skill_organism", "specialist_swarm", "single_worker")
print("\n--- all assertions passed ---")
