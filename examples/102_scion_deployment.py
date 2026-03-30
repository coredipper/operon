"""
Example 102 — Scion Deployment
=================================

Compiles an Operon organism into a Scion grove config with containerized agents.

Usage:
    python examples/102_scion_deployment.py
"""

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.scion_compiler import organism_to_scion

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

organism = skill_organism(
    stages=[
        SkillStage(name="research", role="Researcher", instructions="Search for papers", mode="fixed",
                   handler=lambda t: {"papers": 3}),
        SkillStage(name="analyze", role="Analyst", instructions="Analyze findings", mode="fuzzy",
                   handler=lambda t: {"analysis": "summary"}),
        SkillStage(name="report", role="Writer", instructions="Write the report", mode="fuzzy",
                   handler=lambda t: {"report": "draft"}),
    ],
    fast_nucleus=fast, deep_nucleus=deep,
)

config = organism_to_scion(organism, grove_name="research-project", runtime="docker")

print("=== Scion Grove Config ===")
print(f"  Grove: {config['grove']['name']} ({config['grove']['runtime']})")
print(f"  Agents: {len(config['agents'])} ({len(config['agents']) - 1} stages + watcher)")
for agent in config["agents"]:
    isolation = agent.get("isolation", {})
    print(f"    {agent['name']}: worktree={isolation.get('git_worktree', '?')}, creds={isolation.get('credentials', '?')}")
print(f"  Messaging: {len(config['messaging'])} channels")
for msg in config["messaging"]:
    print(f"    {msg['from']} → {msg['to']}")
print(f"  Watcher: enabled={config['watcher']['enabled']}, telemetry={config['watcher']['telemetry']}")

# --test
assert config["grove"]["name"] == "research-project"
assert len(config["agents"]) == 4  # 3 stages + watcher
assert config["watcher"]["enabled"] is True
assert config["watcher"]["telemetry"] == "otel"
assert len(config["messaging"]) == 2
print("\n--- all assertions passed ---")
