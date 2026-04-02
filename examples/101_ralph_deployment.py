"""
Example 101 — Ralph Deployment
=================================

Compiles an Operon organism into a Ralph orchestrator config.

Usage:
    python examples/101_ralph_deployment.py
"""

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.ralph_compiler import organism_to_ralph

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

organism = skill_organism(
    stages=[
        SkillStage(name="plan", role="Planner", instructions="Break task into steps", mode="fuzzy",
                   handler=lambda t: {"plan": "step1, step2"}),
        SkillStage(name="execute", role="Executor", instructions="Implement the plan", mode="fuzzy",
                   handler=lambda t: {"code": "print('hello')"}),
        SkillStage(name="review", role="Reviewer", instructions="Review for bugs", mode="fixed",
                   handler=lambda t: {"approved": True}),
    ],
    fast_nucleus=fast, deep_nucleus=deep,
)

config = organism_to_ralph(organism, backend="claude")

print("=== Ralph Config ===")
print(f"  Backend: {config['backend']}")
print(f"  Hats: {len(config['hats'])}")
for hat in config["hats"]:
    print(f"    {hat['name']}: pattern={hat['pattern']}")
print(f"  Events: {len(config['events'])}")
for evt in config["events"]:
    print(f"    {evt['from']} --{evt['event']}--> {evt['to']}")
print(f"  Backpressure: {config['backpressure']}")
print(f"  Iteration limit: {config['iteration_limit']}")

# --test
assert len(config["hats"]) == 3
assert config["hats"][2]["pattern"] == "review"
assert len(config["events"]) == 2
assert config["backend"] == "claude"
print("\n--- all assertions passed ---")
