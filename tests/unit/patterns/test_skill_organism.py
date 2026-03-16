"""Tests for the skill-organism runtime."""

from operon_ai import SkillStage, TelemetryProbe, skill_organism
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider


def test_skill_organism_routes_fixed_and_fuzzy_stages():
    fast = Nucleus(provider=MockProvider(responses={
        "return a deterministic category": "EXECUTE: billing",
    }))
    deep = Nucleus(provider=MockProvider(responses={
        "handle the fuzzy policy call": "EXECUTE: escalate to a senior reviewer",
    }))

    organism = skill_organism(
        stages=[
            SkillStage(
                name="classify",
                role="Classifier",
                instructions="Return a deterministic category",
                mode="fixed",
            ),
            SkillStage(
                name="decide",
                role="Planner",
                instructions="Handle the fuzzy policy call",
                mode="fuzzy",
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
    )

    result = organism.run("Route this reimbursement dispute")

    assert [stage.model_alias for stage in result.stage_results] == ["fast", "deep"]
    assert result.stage_results[0].output == "billing"
    assert result.stage_results[1].output == "escalate to a senior reviewer"
    assert result.final_output == "escalate to a senior reviewer"


def test_skill_organism_links_stage_outputs_into_later_prompts():
    deep = Nucleus(provider=MockProvider(responses={
        "billing": "EXECUTE: use the billing escalation workflow",
    }))

    organism = skill_organism(
        stages=[
            SkillStage(
                name="router",
                role="Classifier",
                handler=lambda task: {"category": "billing", "priority": "normal"},
            ),
            SkillStage(
                name="planner",
                role="Planner",
                instructions="Use the prior routing result to choose a workflow",
                mode="fuzzy",
            ),
        ],
        deep_nucleus=deep,
        fast_nucleus=deep,
    )

    result = organism.run("Resolve this invoice mismatch")

    assert result.stage_results[0].output["category"] == "billing"
    assert result.stage_results[1].output == "use the billing escalation workflow"


def test_skill_organism_telemetry_component_updates_shared_state():
    probe = TelemetryProbe()
    fast = Nucleus(provider=MockProvider(responses={
        "concise answer": "EXECUTE: done",
    }))

    organism = skill_organism(
        stages=[
            SkillStage(
                name="answer",
                role="Responder",
                instructions="Concise answer",
                mode="fixed",
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=fast,
        components=[probe],
    )

    result = organism.run("Answer the question")

    assert "_operon_telemetry" in result.shared_state
    assert [event["kind"] for event in result.shared_state["_operon_telemetry"]] == [
        "run_start",
        "stage_start",
        "stage_result",
        "run_complete",
    ]
    assert probe.summary()["total_tokens"] >= 0


def test_skill_organism_halts_on_blocking_stage_by_default():
    fast = Nucleus(provider=MockProvider(responses={
        "check for safety": "BLOCK: unsafe request",
        "this should not execute": "EXECUTE: should not happen",
    }))

    organism = skill_organism(
        stages=[
            SkillStage(
                name="review",
                role="RiskAssessor",
                instructions="Check for safety",
                mode="fixed",
            ),
            SkillStage(
                name="execute",
                role="Executor",
                instructions="This should not execute",
                mode="fuzzy",
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=fast,
    )

    result = organism.run("Drop the table")

    assert len(result.stage_results) == 1
    assert result.stage_results[0].action_type == "BLOCK"
    assert result.final_output == "unsafe request"
