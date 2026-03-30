"""Tests for the Swarms compiler (organism -> Swarms workflow config dict)."""

from __future__ import annotations

import pytest

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.swarms_compiler import (
    managed_to_swarms,
    organism_to_swarms,
)
from operon_ai.convergence.types import RuntimeConfig
from operon_ai.patterns.managed import ManagedOrganism
from operon_ai.patterns.types import CognitiveMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_organism(stages: list[SkillStage] | None = None):
    """Build a minimal SkillOrganism backed by MockProvider."""
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))
    if stages is None:
        stages = [
            SkillStage(
                name="intake",
                role="Normalizer",
                handler=lambda task: {"parsed": task},
            ),
            SkillStage(
                name="router",
                role="Classifier",
                instructions="Classify the task into a category.",
                mode="fixed",
            ),
            SkillStage(
                name="planner",
                role="Planner",
                instructions="Create a step-by-step plan for the task.",
                mode="fuzzy",
            ),
        ]
    return skill_organism(
        stages=stages,
        fast_nucleus=fast,
        deep_nucleus=deep,
    )


# ---------------------------------------------------------------------------
# organism_to_swarms
# ---------------------------------------------------------------------------


class TestOrganismToSwarms:
    """Tests for organism_to_swarms."""

    def test_produces_valid_dict_with_agents_and_edges(self) -> None:
        organism = _make_organism()
        result = organism_to_swarms(organism)
        assert isinstance(result, dict)
        assert "agents" in result
        assert "edges" in result
        assert "workflow_type" in result
        assert "config" in result

    def test_sequential_organism_produces_sequential_workflow(self) -> None:
        organism = _make_organism()
        result = organism_to_swarms(organism)
        assert result["workflow_type"] == "SequentialWorkflow"

    def test_agents_count_matches_stages(self) -> None:
        organism = _make_organism()
        result = organism_to_swarms(organism)
        assert len(result["agents"]) == 3

    def test_stage_instructions_mapped_to_system_prompt(self) -> None:
        organism = _make_organism()
        result = organism_to_swarms(organism)
        router = [a for a in result["agents"] if a["name"] == "router"][0]
        assert router["system_prompt"] == "Classify the task into a category."

    def test_edges_form_linear_chain(self) -> None:
        organism = _make_organism()
        result = organism_to_swarms(organism)
        assert result["edges"] == [
            ("intake", "router"),
            ("router", "planner"),
        ]

    def test_runtime_config_timeout_propagated(self) -> None:
        organism = _make_organism()
        cfg = RuntimeConfig(timeout=60.0)
        result = organism_to_swarms(organism, config=cfg)
        for agent in result["agents"]:
            assert agent["timeout"] == 60.0

    def test_runtime_config_provider_in_output(self) -> None:
        organism = _make_organism()
        cfg = RuntimeConfig(provider="anthropic")
        result = organism_to_swarms(organism, config=cfg)
        assert result["config"]["provider"] == "anthropic"
        # Anthropic provider should use claude model names.
        router = [a for a in result["agents"] if a["name"] == "router"][0]
        assert "claude" in router["model"]

    def test_empty_stages_raises_value_error(self) -> None:
        fast = Nucleus(provider=MockProvider(responses={}))
        deep = Nucleus(provider=MockProvider(responses={}))
        organism = skill_organism(
            stages=[
                SkillStage(
                    name="dummy",
                    role="Dummy",
                    handler=lambda task: task,
                ),
            ],
            fast_nucleus=fast,
            deep_nucleus=deep,
        )
        # Monkey-patch stages to empty tuple to test the guard.
        object.__setattr__(organism, "stages", ())
        with pytest.raises(ValueError, match="no stages"):
            organism_to_swarms(organism)

    def test_single_stage_no_edges(self) -> None:
        organism = _make_organism(stages=[
            SkillStage(
                name="solo",
                role="Worker",
                handler=lambda task: task,
            ),
        ])
        result = organism_to_swarms(organism)
        assert len(result["agents"]) == 1
        assert result["edges"] == []

    def test_mock_provider_model_prefix(self) -> None:
        organism = _make_organism()
        cfg = RuntimeConfig(provider="mock")
        result = organism_to_swarms(organism, config=cfg)
        router = [a for a in result["agents"] if a["name"] == "router"][0]
        assert router["model"].startswith("mock-")

    def test_max_loops_from_retries(self) -> None:
        organism = _make_organism()
        cfg = RuntimeConfig(max_retries=3)
        result = organism_to_swarms(organism, config=cfg)
        assert result["config"]["max_loops"] == 4


# ---------------------------------------------------------------------------
# managed_to_swarms
# ---------------------------------------------------------------------------


class TestManagedToSwarms:
    """Tests for managed_to_swarms."""

    def test_extracts_inner_organism(self) -> None:
        inner = _make_organism()
        managed = ManagedOrganism()
        object.__setattr__(managed, "_organism", inner)
        result = managed_to_swarms(managed)
        assert isinstance(result, dict)
        assert len(result["agents"]) == 3

    def test_no_inner_organism_raises(self) -> None:
        managed = ManagedOrganism()
        with pytest.raises(ValueError, match="no inner SkillOrganism"):
            managed_to_swarms(managed)
