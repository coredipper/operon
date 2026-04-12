"""Tests for the DeerFlow compiler (organism -> DeerFlow session config dict)."""

from __future__ import annotations

import pytest

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.deerflow_compiler import (
    _instructions_to_skills,
    managed_to_deerflow,
    organism_to_deerflow,
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
                name="coordinator",
                role="Lead",
                instructions="Coordinate the research team. Assign tasks to sub-agents.",
                mode="fuzzy",
            ),
            SkillStage(
                name="researcher",
                role="Researcher",
                instructions="Search the web for relevant papers. Summarize findings.",
                mode="fixed",
                cognitive_mode=CognitiveMode.OBSERVATIONAL,
            ),
            SkillStage(
                name="writer",
                role="Writer",
                instructions="Write the final report based on research findings.",
                mode="fuzzy",
            ),
        ]
    return skill_organism(
        stages=stages,
        fast_nucleus=fast,
        deep_nucleus=deep,
    )


# ---------------------------------------------------------------------------
# _instructions_to_skills
# ---------------------------------------------------------------------------


class TestInstructionsToSkills:
    """Tests for the _instructions_to_skills helper."""

    def test_multi_sentence_split(self) -> None:
        skills = _instructions_to_skills("Do X. Then do Y. Finally Z.")
        assert len(skills) == 3

    def test_single_sentence(self) -> None:
        skills = _instructions_to_skills("Do the thing")
        assert skills == ["Do the thing"]

    def test_empty_string(self) -> None:
        skills = _instructions_to_skills("")
        assert skills == []

    def test_whitespace_only(self) -> None:
        skills = _instructions_to_skills("   ")
        assert skills == []


# ---------------------------------------------------------------------------
# organism_to_deerflow
# ---------------------------------------------------------------------------


class TestOrganismToDeerflow:
    """Tests for organism_to_deerflow."""

    def test_produces_valid_dict(self) -> None:
        organism = _make_organism()
        result = organism_to_deerflow(organism)
        assert isinstance(result, dict)
        assert "assistant_id" in result
        assert "skills" in result
        assert "sub_agents" in result
        assert "recursion_limit" in result
        assert "sandbox" in result
        assert "config" in result

    def test_first_stage_becomes_assistant_id(self) -> None:
        organism = _make_organism()
        result = organism_to_deerflow(organism)
        assert result["assistant_id"] == "coordinator"

    def test_remaining_stages_become_sub_agents(self) -> None:
        organism = _make_organism()
        result = organism_to_deerflow(organism)
        assert len(result["sub_agents"]) == 2
        names = [sa["name"] for sa in result["sub_agents"]]
        assert names == ["researcher", "writer"]

    def test_skills_extracted_from_instructions(self) -> None:
        organism = _make_organism()
        result = organism_to_deerflow(organism)
        # Lead has 2 sentences.
        assert len(result["skills"]) == 2
        # Researcher sub-agent has 2 sentences.
        researcher = result["sub_agents"][0]
        assert len(researcher["skills"]) == 2

    def test_sandbox_from_runtime_config(self) -> None:
        organism = _make_organism()
        cfg = RuntimeConfig(sandbox="docker")
        result = organism_to_deerflow(organism, config=cfg)
        assert result["sandbox"] == "docker"

    def test_recursion_limit_from_timeout(self) -> None:
        organism = _make_organism()
        cfg = RuntimeConfig(timeout=60.0)
        result = organism_to_deerflow(organism, config=cfg)
        assert result["recursion_limit"] == 30  # 60 / 2

    def test_thinking_enabled_from_cognitive_mode(self) -> None:
        """Lead stage with fuzzy mode should have thinking_enabled=True."""
        organism = _make_organism()
        result = organism_to_deerflow(organism)
        assert result["config"]["thinking_enabled"] is True

    def test_observational_lead_disables_thinking(self) -> None:
        """Lead stage with OBSERVATIONAL cognitive mode disables thinking."""
        organism = _make_organism(stages=[
            SkillStage(
                name="observer",
                role="Monitor",
                instructions="Watch for anomalies.",
                mode="fixed",
                cognitive_mode=CognitiveMode.OBSERVATIONAL,
            ),
            SkillStage(
                name="worker",
                role="Worker",
                instructions="Process the data.",
                mode="fuzzy",
            ),
        ])
        result = organism_to_deerflow(organism)
        assert result["config"]["thinking_enabled"] is False

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
        object.__setattr__(organism, "stages", ())
        with pytest.raises(ValueError, match="no stages"):
            organism_to_deerflow(organism)

    def test_single_stage_no_sub_agents(self) -> None:
        organism = _make_organism(stages=[
            SkillStage(
                name="solo",
                role="Worker",
                instructions="Do everything.",
                mode="fuzzy",
            ),
        ])
        result = organism_to_deerflow(organism)
        assert result["assistant_id"] == "solo"
        assert result["sub_agents"] == []


# ---------------------------------------------------------------------------
# managed_to_deerflow
# ---------------------------------------------------------------------------


class TestManagedToDeerflow:
    """Tests for managed_to_deerflow."""

    def test_extracts_inner_organism(self) -> None:
        inner = _make_organism()
        managed = ManagedOrganism()
        object.__setattr__(managed, "_organism", inner)
        result = managed_to_deerflow(managed)
        assert isinstance(result, dict)
        assert result["assistant_id"] == "coordinator"

    def test_no_inner_organism_raises(self) -> None:
        managed = ManagedOrganism()
        with pytest.raises(ValueError, match="no inner SkillOrganism"):
            managed_to_deerflow(managed)


# ---------------------------------------------------------------------------
# Round-trip: compile -> decompile
# ---------------------------------------------------------------------------


class TestDeerflowDecompile:
    """Tests for deerflow_to_topology() round-trip."""

    def test_roundtrip_agents_preserved(self) -> None:
        from operon_ai.convergence.deerflow_compiler import deerflow_to_topology

        org = _make_organism()
        compiled = organism_to_deerflow(org)
        topology = deerflow_to_topology(compiled)

        # Lead + sub_agents should produce matching agent names
        agent_names = {a["name"] for a in topology.agents}
        assert "coordinator" in agent_names
        assert "researcher" in agent_names
        assert "writer" in agent_names

    def test_roundtrip_edges_preserved(self) -> None:
        from operon_ai.convergence.deerflow_compiler import deerflow_to_topology

        org = _make_organism()
        compiled = organism_to_deerflow(org)
        topology = deerflow_to_topology(compiled)

        # DeerFlow uses hub-and-spoke: lead -> each sub_agent
        assert ("coordinator", "researcher") in topology.edges
        assert ("coordinator", "writer") in topology.edges

    def test_roundtrip_certificates_preserved(self) -> None:
        from operon_ai.convergence.deerflow_compiler import deerflow_to_topology

        org = _make_organism()
        compiled = organism_to_deerflow(org)

        # Certificates should survive the round trip
        topology = deerflow_to_topology(compiled)
        certs = topology.metadata.get("certificates", [])

        # Source certificates match decompiled certificates
        assert certs == compiled.get("certificates", [])

    def test_roundtrip_capabilities_populated(self) -> None:
        from operon_ai.convergence.deerflow_compiler import deerflow_to_topology

        org = _make_organism()
        compiled = organism_to_deerflow(org)
        topology = deerflow_to_topology(compiled)

        # Skills from compiled dict should populate capabilities
        if compiled.get("skills"):
            cap_agents = {name for name, _ in topology.capabilities}
            assert compiled["assistant_id"] in cap_agents
