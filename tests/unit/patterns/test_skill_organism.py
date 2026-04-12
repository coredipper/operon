"""Tests for the skill-organism runtime."""

import pytest
from datetime import datetime, timedelta

from operon_ai import ATP_Store, BiTemporalMemory, SkillStage, SubstrateView, TelemetryProbe, skill_organism
from operon_ai.memory.bitemporal import BiTemporalQuery
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider


def test_skill_organism_rejects_duplicate_stage_names():
    with pytest.raises(ValueError, match="Duplicate stage name 'echo'"):
        skill_organism(
            stages=[
                SkillStage(name="echo", role="Echo", handler=lambda t: t),
                SkillStage(name="echo", role="Echo2", handler=lambda t: t),
            ],
        )


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


def test_telemetry_captures_organism_config():
    """TelemetryProbe run_start event includes organism config metadata."""
    probe = TelemetryProbe()
    fast = Nucleus(provider=MockProvider(responses={
        "do step 1": "EXECUTE: done1",
        "do step 2": "EXECUTE: done2",
    }))

    organism = skill_organism(
        stages=[
            SkillStage(name="reader", role="Reader", instructions="do step 1", mode="fixed"),
            SkillStage(name="writer", role="Writer", instructions="do step 2", mode="fuzzy"),
        ],
        fast_nucleus=fast,
        deep_nucleus=fast,
        components=[probe],
        budget=ATP_Store(budget=1000),
    )

    result = organism.run("test")

    # Find the run_start event
    run_start = next(
        e for e in result.shared_state["_operon_telemetry"]
        if e["kind"] == "run_start"
    )
    assert "organism_config" in run_start, "run_start should contain organism_config"
    config = run_start["organism_config"]
    assert config["stage_count"] == 2
    assert config["stage_names"] == ["reader", "writer"]
    assert config["mode_assignments"] == {"reader": "fixed", "writer": "fuzzy"}
    assert "priority_gating" in config["certificate_theorems"]

    # _organism_config must not leak into returned shared_state
    assert "_organism_config" not in result.shared_state, (
        "_organism_config should be removed after on_run_start"
    )


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


# ---------------------------------------------------------------------------
# Phase 2: Bi-Temporal Substrate Integration
# ---------------------------------------------------------------------------

def _t(day: int) -> datetime:
    """Helper: deterministic datetime for day N of March 2026."""
    return datetime(2026, 3, day, 12, 0, 0)


# -- Backward compatibility ------------------------------------------------

def test_skill_organism_works_unchanged_without_substrate():
    organism = skill_organism(
        stages=[
            SkillStage(name="echo", role="Echo", handler=lambda task: task),
        ],
    )
    result = organism.run("hello")
    assert result.final_output == "hello"
    assert result.shared_state.get("echo") == "hello"


# -- Read path -------------------------------------------------------------

def test_substrate_view_injected_into_handler_stage():
    mem = BiTemporalMemory()
    mem.record_fact(
        subject="acct:1", predicate="tier", value="gold",
        valid_from=_t(1), recorded_from=_t(1), source="crm",
    )

    captured = {}

    def reader(task, state, outputs, stage, view):
        captured["view"] = view
        return "read_ok"

    organism = skill_organism(
        stages=[
            SkillStage(
                name="reader", role="Reader", handler=reader,
                read_query="acct:1",
            ),
        ],
        substrate=mem,
    )
    result = organism.run("check account")

    assert result.final_output == "read_ok"
    view = captured["view"]
    assert isinstance(view, SubstrateView)
    assert len(view.facts) == 1
    assert view.facts[0].value == "gold"


def test_substrate_view_with_callable_read_query():
    mem = BiTemporalMemory()
    mem.record_fact(
        subject="acct:1", predicate="tier", value="silver",
        valid_from=_t(1), recorded_from=_t(1), source="crm",
    )

    captured = {}

    def custom_query(task, state, outputs, substrate):
        return BiTemporalQuery(subject="acct:1")

    def reader(task, state, outputs, stage, view):
        captured["view"] = view
        return "ok"

    organism = skill_organism(
        stages=[
            SkillStage(
                name="reader", role="Reader", handler=reader,
                read_query=custom_query,
            ),
        ],
        substrate=mem,
    )
    organism.run("check")

    assert len(captured["view"].facts) == 1
    assert captured["view"].facts[0].value == "silver"


def test_substrate_view_empty_when_no_matching_facts():
    mem = BiTemporalMemory()
    captured = {}

    def reader(task, state, outputs, stage, view):
        captured["view"] = view
        return "empty"

    organism = skill_organism(
        stages=[
            SkillStage(
                name="reader", role="Reader", handler=reader,
                read_query="nonexistent",
            ),
        ],
        substrate=mem,
    )
    organism.run("check")

    assert captured["view"].facts == ()


def test_substrate_view_none_when_read_query_not_set():
    mem = BiTemporalMemory()
    captured = {}

    def reader(task, state, outputs, stage, view):
        captured["view"] = view
        return "no_query"

    organism = skill_organism(
        stages=[
            SkillStage(name="reader", role="Reader", handler=reader),
        ],
        substrate=mem,
    )
    organism.run("check")

    assert captured["view"] is None


# -- Write path: emit_output_fact ------------------------------------------

def test_emit_output_fact_records_stage_output():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="research", role="Researcher",
                handler=lambda task: {"revenue": 1_000_000},
                emit_output_fact=True,
            ),
        ],
        substrate=mem,
    )
    organism.run("analyze acct:7")

    facts = mem.retrieve_valid_at(at=datetime.now())
    assert len(facts) == 1
    assert facts[0].predicate == "research"
    assert facts[0].value == {"revenue": 1_000_000}
    assert facts[0].source == "research"


def test_emit_output_fact_uses_fact_tags():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="audit", role="Auditor",
                handler=lambda task: "clean",
                emit_output_fact=True,
                fact_tags=("compliance", "auto"),
            ),
        ],
        substrate=mem,
    )
    organism.run("check")

    facts = mem.retrieve_valid_at(at=datetime.now())
    assert facts[0].tags == ("compliance", "auto")


def test_emit_output_fact_false_records_nothing():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="silent", role="Worker",
                handler=lambda task: "done",
            ),
        ],
        substrate=mem,
    )
    organism.run("work")

    assert mem.retrieve_valid_at(at=datetime.now()) == []


# -- Write path: fact_extractor --------------------------------------------

def test_fact_extractor_asserts_new_fact():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="research", role="Researcher",
                handler=lambda task: "high revenue",
                fact_extractor=lambda task: {
                    "op": "assert",
                    "subject": "acct:7",
                    "predicate": "revenue_class",
                    "value": "high",
                },
            ),
        ],
        substrate=mem,
    )
    organism.run("analyze")

    facts = mem.retrieve_valid_at(at=datetime.now())
    assert len(facts) == 1
    assert facts[0].subject == "acct:7"
    assert facts[0].predicate == "revenue_class"
    assert facts[0].value == "high"


def test_fact_extractor_corrects_existing_fact():
    mem = BiTemporalMemory()
    original = mem.record_fact(
        subject="acct:7", predicate="risk", value="low",
        valid_from=_t(1), recorded_from=_t(1), source="initial",
    )

    organism = skill_organism(
        stages=[
            SkillStage(
                name="adversary", role="Adversary",
                handler=lambda task: "corrected",
                fact_extractor=lambda task: {
                    "op": "correct",
                    "old_fact_id": original.fact_id,
                    "value": "high",
                },
            ),
        ],
        substrate=mem,
    )
    organism.run("challenge")

    # Original fact should be closed
    history = mem.history("acct:7", "risk")
    assert len(history) == 2
    assert history[0].recorded_to is not None  # closed
    assert history[1].value == "high"
    assert history[1].supersedes == original.fact_id


def test_fact_extractor_invalidates_existing_fact():
    mem = BiTemporalMemory()
    fact = mem.record_fact(
        subject="acct:7", predicate="status", value="active",
        valid_from=_t(1), recorded_from=_t(1), source="crm",
    )

    organism = skill_organism(
        stages=[
            SkillStage(
                name="cleanup", role="Cleaner",
                handler=lambda task: "invalidated",
                fact_extractor=lambda task: {
                    "op": "invalidate",
                    "fact_id": fact.fact_id,
                },
            ),
        ],
        substrate=mem,
    )
    organism.run("clean up")

    # Active facts should be empty now
    assert mem.retrieve_valid_at(at=datetime.now()) == []
    # But history preserves it
    assert len(mem.history("acct:7")) == 1
    assert mem.history("acct:7")[0].recorded_to is not None


def test_fact_extractor_returns_tuple_shorthand():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="tagger", role="Tagger",
                handler=lambda task: "tagged",
                fact_extractor=lambda task: ("client:42", "status", "approved"),
            ),
        ],
        substrate=mem,
    )
    organism.run("tag it")

    facts = mem.retrieve_valid_at(at=datetime.now())
    assert len(facts) == 1
    assert facts[0].subject == "client:42"
    assert facts[0].predicate == "status"
    assert facts[0].value == "approved"
    assert facts[0].source == "tagger"


def test_fact_extractor_returns_none_records_nothing():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="maybe", role="Conditional",
                handler=lambda task: "nothing to record",
                fact_extractor=lambda task: None,
            ),
        ],
        substrate=mem,
    )
    organism.run("check")

    assert mem.retrieve_valid_at(at=datetime.now()) == []


def test_fact_extractor_returns_multiple_events():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="multi", role="Multi",
                handler=lambda task: "done",
                fact_extractor=lambda task: [
                    ("acct:1", "status", "active"),
                    ("acct:1", "tier", "gold"),
                ],
            ),
        ],
        substrate=mem,
    )
    organism.run("record")

    facts = mem.retrieve_valid_at(at=datetime.now())
    assert len(facts) == 2
    predicates = {f.predicate for f in facts}
    assert predicates == {"status", "tier"}


def test_fact_extractor_default_source_from_stage():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="sourcer", role="Sourcer",
                handler=lambda task: "done",
                fact_extractor=lambda task: {
                    "subject": "x", "predicate": "y", "value": "z",
                },
            ),
        ],
        substrate=mem,
    )
    organism.run("go")

    facts = mem.retrieve_valid_at(at=datetime.now())
    assert facts[0].source == "sourcer"


# -- Integration: historical reconstruction --------------------------------

def test_later_stage_correction_preserves_history():
    mem = BiTemporalMemory()

    def adversary_extractor(task):
        # Look up the fact the research stage recorded
        facts = mem.retrieve_valid_at(at=datetime.now(), subject="acct:1", predicate="risk")
        assert len(facts) == 1
        return {
            "op": "correct",
            "old_fact_id": facts[0].fact_id,
            "value": "high",
        }

    organism = skill_organism(
        stages=[
            SkillStage(
                name="research", role="Researcher",
                handler=lambda task: "researched",
                fact_extractor=lambda task: ("acct:1", "risk", "low"),
            ),
            SkillStage(
                name="adversary", role="Adversary",
                handler=lambda task: "challenged",
                fact_extractor=adversary_extractor,
            ),
        ],
        substrate=mem,
    )
    organism.run("evaluate acct:1")

    # Current state: risk=high
    current = mem.retrieve_valid_at(at=datetime.now(), subject="acct:1", predicate="risk")
    assert len(current) == 1
    assert current[0].value == "high"

    # History preserves both versions
    history = mem.history("acct:1", "risk")
    assert len(history) == 2
    assert history[0].value == "low"
    assert history[0].recorded_to is not None  # closed
    assert history[1].value == "high"
    assert history[1].supersedes == history[0].fact_id


def test_substrate_and_shared_state_remain_separate():
    mem = BiTemporalMemory()
    organism = skill_organism(
        stages=[
            SkillStage(
                name="writer", role="Writer",
                handler=lambda task: "output_value",
                emit_output_fact=True,
            ),
        ],
        substrate=mem,
    )
    result = organism.run("task")

    # Substrate has the fact
    assert len(mem.retrieve_valid_at(at=datetime.now())) == 1
    # shared_state has the stage output but NOT the fact
    assert result.shared_state["writer"] == "output_value"
    assert "_bitemporal" not in result.shared_state


def test_halt_on_block_does_not_write_facts_for_skipped_stages():
    mem = BiTemporalMemory()
    fast = Nucleus(provider=MockProvider(responses={
        "check safety": "BLOCK: unsafe",
    }))

    organism = skill_organism(
        stages=[
            SkillStage(
                name="guard", role="Guard",
                instructions="Check safety",
                mode="fixed",
            ),
            SkillStage(
                name="recorder", role="Recorder",
                handler=lambda task: "should not run",
                emit_output_fact=True,
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=fast,
        substrate=mem,
    )
    result = organism.run("dangerous task")

    assert len(result.stage_results) == 1
    assert result.stage_results[0].action_type == "BLOCK"
    # Recorder stage was skipped, so no facts
    assert mem.retrieve_valid_at(at=datetime.now()) == []
