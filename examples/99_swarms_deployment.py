"""
Example 99 -- Swarms Deployment Compiler
==========================================

Demonstrates compiling a 3-stage SkillOrganism into a Swarms workflow
config dict.  The output is a plain dict (no Swarms installation needed).

Usage:
    python examples/99_swarms_deployment.py
"""

import json

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.swarms_compiler import organism_to_swarms
from operon_ai.convergence.types import RuntimeConfig

# ---------------------------------------------------------------------------
# 1. Build a 3-stage organism
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

organism = skill_organism(
    stages=[
        SkillStage(
            name="intake",
            role="Normalizer",
            handler=lambda task: {"parsed": task},
        ),
        SkillStage(
            name="router",
            role="Classifier",
            instructions="Classify the incoming request into the correct department.",
            mode="fixed",
        ),
        SkillStage(
            name="planner",
            role="Planner",
            instructions="Create a detailed action plan based on the classification.",
            mode="fuzzy",
        ),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
)

# ---------------------------------------------------------------------------
# 2. Compile to Swarms config (default RuntimeConfig)
# ---------------------------------------------------------------------------

swarms_config = organism_to_swarms(organism)

print("=== Swarms Workflow Config (default) ===")
print(json.dumps(swarms_config, indent=2, default=str))
print()

# ---------------------------------------------------------------------------
# 3. Compile with custom RuntimeConfig
# ---------------------------------------------------------------------------

custom_cfg = RuntimeConfig(
    provider="openai",
    timeout=60.0,
    max_retries=2,
    sandbox="docker",
)

swarms_custom = organism_to_swarms(organism, config=custom_cfg)

print("=== Swarms Workflow Config (custom) ===")
print(json.dumps(swarms_custom, indent=2, default=str))
print()

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

# Verify structure.
assert swarms_config["workflow_type"] == "SequentialWorkflow"
assert len(swarms_config["agents"]) == 3
assert len(swarms_config["edges"]) == 2
assert swarms_config["edges"][0] == ("intake", "router")
assert swarms_config["edges"][1] == ("router", "planner")

# Verify agent fields.
router = [a for a in swarms_config["agents"] if a["name"] == "router"][0]
assert router["system_prompt"] == "Classify the incoming request into the correct department."
assert router["role"] == "Classifier"
assert "timeout" in router

# Verify custom config propagation.
assert swarms_custom["config"]["provider"] == "openai"
assert swarms_custom["config"]["sandbox"] == "docker"
assert swarms_custom["config"]["max_loops"] == 3  # max_retries(2) + 1
for agent in swarms_custom["agents"]:
    assert agent["timeout"] == 60.0

# Verify model selection: openai provider should get gpt models.
planner = [a for a in swarms_custom["agents"] if a["name"] == "planner"][0]
assert "gpt" in planner["model"]

print("--- all assertions passed ---")
