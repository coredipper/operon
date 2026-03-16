"""
Example 69: Skill Organism Runtime
=================================

Demonstrates a provider-bound organism built from multiple stages:

1. A deterministic intake stage
2. A fast/cheap routing stage for fixed classification work
3. A deeper planning stage for fuzzy reasoning
4. A telemetry component attached without changing the skill itself

This is the practical layer behind the feedback that skills often start as
scripts with instructions between stages. Operon turns that into a runtime
where model choice, telemetry, and stage composition stay reconfigurable.
"""

from __future__ import annotations

import sys

from operon_ai import MockProvider, Nucleus, SkillStage, TelemetryProbe, skill_organism


def build_demo_organism():
    fast = Nucleus(provider=MockProvider(responses={
        "return a deterministic routing label": "EXECUTE: billing",
    }))
    deep = Nucleus(provider=MockProvider(responses={
        "billing": (
            "EXECUTE: Ask the billing team for invoice context, then route the "
            "customer to a human reviewer if the dispute touches refunds."
        ),
    }))
    telemetry = TelemetryProbe()

    organism = skill_organism(
        stages=[
            SkillStage(
                name="intake",
                role="Normalizer",
                handler=lambda task, state, outputs, stage: {
                    "request": task,
                    "channel": "support",
                    "priority": "normal",
                },
            ),
            SkillStage(
                name="router",
                role="Classifier",
                instructions="Return a deterministic routing label.",
                mode="fixed",
            ),
            SkillStage(
                name="planner",
                role="Planner",
                instructions=(
                    "Use the task, routing label, and telemetry to propose a "
                    "next-step plan for the support team."
                ),
                mode="fuzzy",
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[telemetry],
    )
    return organism, telemetry


def main() -> None:
    organism, telemetry = build_demo_organism()
    result = organism.run("Customer says an invoice refund was never applied.")

    print("=" * 72)
    print("Skill Organism Runtime")
    print("=" * 72)
    for stage in result.stage_results:
        print(f"\n[{stage.stage_name}]")
        print(f"  role:        {stage.role}")
        print(f"  model alias: {stage.model_alias}")
        print(f"  provider:    {stage.provider}")
        print(f"  action:      {stage.action_type}")
        print(f"  output:      {stage.output}")

    print("\nTelemetry summary:")
    summary = telemetry.summary()
    print(f"  events:       {summary['events']}")
    print(f"  total tokens: {summary['total_tokens']}")
    print(f"  stages:       {summary['stages']}")

    print("\nShared telemetry trail:")
    for event in result.shared_state.get("_operon_telemetry", []):
        print(f"  - {event['kind']}: {event.get('stage_name')}")


if __name__ == "__main__":
    try:
        main()
        if "--test" in sys.argv:
            organism, telemetry = build_demo_organism()
            result = organism.run("Customer says an invoice refund was never applied.")
            assert result.stage_results[1].model_alias == "fast"
            assert result.stage_results[2].model_alias == "deep"
            assert "billing" in str(result.stage_results[1].output).lower()
            assert "billing team" in str(result.final_output).lower()
            assert telemetry.summary()["events"] == 8
            print("\n[OK] Skill organism example completed.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
