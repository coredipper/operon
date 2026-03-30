"""
Example 100 -- DeerFlow Deployment Compiler
=============================================

Demonstrates compiling a 3-stage SkillOrganism into a DeerFlow session
config dict.  The output is a plain dict (no DeerFlow installation needed).

Usage:
    python examples/100_deerflow_deployment.py
"""

import json

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.deerflow_compiler import organism_to_deerflow
from operon_ai.convergence.types import RuntimeConfig
from operon_ai.patterns.types import CognitiveMode

# ---------------------------------------------------------------------------
# 1. Build a 3-stage organism
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

organism = skill_organism(
    stages=[
        SkillStage(
            name="coordinator",
            role="Lead",
            instructions="Coordinate the research team. Assign sub-tasks to agents.",
            mode="fuzzy",
        ),
        SkillStage(
            name="researcher",
            role="Researcher",
            instructions="Search the web for relevant papers. Summarize key findings.",
            mode="fixed",
            cognitive_mode=CognitiveMode.OBSERVATIONAL,
        ),
        SkillStage(
            name="writer",
            role="Writer",
            instructions="Write the final report based on research findings.",
            mode="fuzzy",
        ),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
)

# ---------------------------------------------------------------------------
# 2. Compile to DeerFlow config (default RuntimeConfig)
# ---------------------------------------------------------------------------

deerflow_config = organism_to_deerflow(organism)

print("=== DeerFlow Session Config (default) ===")
print(json.dumps(deerflow_config, indent=2, default=str))
print()

# ---------------------------------------------------------------------------
# 3. Compile with custom RuntimeConfig
# ---------------------------------------------------------------------------

custom_cfg = RuntimeConfig(
    provider="anthropic",
    timeout=120.0,
    sandbox="docker",
)

deerflow_custom = organism_to_deerflow(organism, config=custom_cfg)

print("=== DeerFlow Session Config (custom) ===")
print(json.dumps(deerflow_custom, indent=2, default=str))
print()

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

# Verify structure.
assert deerflow_config["assistant_id"] == "coordinator"
assert isinstance(deerflow_config["skills"], list)
assert len(deerflow_config["skills"]) == 2  # 2 sentences in coordinator instructions
assert len(deerflow_config["sub_agents"]) == 2

# Verify sub-agents.
names = [sa["name"] for sa in deerflow_config["sub_agents"]]
assert names == ["researcher", "writer"]

researcher = deerflow_config["sub_agents"][0]
assert researcher["role"] == "Researcher"
assert len(researcher["skills"]) == 2  # 2 sentences

# Verify thinking_enabled (lead is fuzzy -> action-oriented -> True).
assert deerflow_config["config"]["thinking_enabled"] is True

# Verify custom config propagation.
assert deerflow_custom["sandbox"] == "docker"
assert deerflow_custom["recursion_limit"] == 60  # 120 / 2

print("--- all assertions passed ---")
