"""Tests for the DeerFlow 2.0 session-config adapter."""

from __future__ import annotations

from operon_ai.convergence.deerflow_adapter import (
    deerflow_skills_to_stages,
    deerflow_to_template,
    parse_deerflow_session,
)
from operon_ai.convergence.types import ExternalTopology
from operon_ai.patterns.repository import PatternTemplate
from operon_ai.patterns.types import CognitiveMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SESSION_CONFIG: dict = {
    "assistant_id": "lead_agent",
    "skills": ["web_research", "report_generation", "code_execution"],
    "sub_agents": [
        {"name": "researcher", "role": "researcher", "skills": ["web_search", "summarize"]},
        {"name": "coder", "role": "developer", "skills": ["python", "testing"]},
    ],
    "recursion_limit": 100,
    "sandbox": "docker",
    "config": {"thinking_enabled": True},
}

_SKILL_DICTS: list[dict] = [
    {"name": "web_research", "description": "Search and summarize web content", "category": "research"},
    {"name": "report_gen", "description": "Generate a report", "category": "generation"},
    {"name": "code_exec", "description": "Execute Python code", "category": "execution"},
    {"name": "verify", "description": "Verify outputs", "category": "verification"},
    {"name": "unknown_skill", "description": "Mystery skill", "category": "alien"},
]


# ---------------------------------------------------------------------------
# parse_deerflow_session
# ---------------------------------------------------------------------------


class TestParseDeerflowSession:
    """Tests for parse_deerflow_session."""

    def test_returns_external_topology(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        assert isinstance(result, ExternalTopology)

    def test_source_is_deerflow(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        assert result.source == "deerflow"

    def test_pattern_name(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        assert result.pattern_name == "HierarchicalDeerFlow"

    def test_agents_include_lead_and_sub_agents(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        names = [a["name"] for a in result.agents]
        assert names == ["lead_agent", "researcher", "coder"]

    def test_lead_agent_has_skills(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        lead = result.agents[0]
        assert lead["skills"] == ("web_research", "report_generation", "code_execution")

    def test_edges_are_hierarchical(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        assert result.edges == (("lead_agent", "researcher"), ("lead_agent", "coder"))

    def test_sandbox_in_metadata(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        assert result.metadata["sandbox"] == "docker"

    def test_recursion_limit_in_metadata(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        assert result.metadata["recursion_limit"] == 100

    def test_config_in_metadata(self) -> None:
        result = parse_deerflow_session(_SESSION_CONFIG)
        assert result.metadata["config"]["thinking_enabled"] is True

    def test_minimal_config(self) -> None:
        """Empty config still yields a valid topology."""
        result = parse_deerflow_session({})
        assert isinstance(result, ExternalTopology)
        assert result.source == "deerflow"
        assert len(result.agents) == 1  # lead only
        assert result.edges == ()


# ---------------------------------------------------------------------------
# deerflow_skills_to_stages
# ---------------------------------------------------------------------------


class TestDeerflowSkillsToStages:
    """Tests for deerflow_skills_to_stages."""

    def test_returns_tuple_of_skill_stages(self) -> None:
        stages = deerflow_skills_to_stages(_SKILL_DICTS)
        assert isinstance(stages, tuple)
        assert len(stages) == 5

    def test_research_category_is_observational(self) -> None:
        stages = deerflow_skills_to_stages(_SKILL_DICTS)
        research = stages[0]
        assert research.name == "web_research"
        assert research.mode == "fixed"
        assert research.cognitive_mode == CognitiveMode.OBSERVATIONAL

    def test_generation_category_is_action_oriented(self) -> None:
        stages = deerflow_skills_to_stages(_SKILL_DICTS)
        gen = stages[1]
        assert gen.name == "report_gen"
        assert gen.mode == "fuzzy"
        assert gen.cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_execution_category_is_action_oriented(self) -> None:
        stages = deerflow_skills_to_stages(_SKILL_DICTS)
        exe = stages[2]
        assert exe.mode == "fuzzy"
        assert exe.cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_verification_category_is_observational(self) -> None:
        stages = deerflow_skills_to_stages(_SKILL_DICTS)
        verify = stages[3]
        assert verify.mode == "fixed"
        assert verify.cognitive_mode == CognitiveMode.OBSERVATIONAL

    def test_unknown_category_defaults_to_action_oriented(self) -> None:
        stages = deerflow_skills_to_stages(_SKILL_DICTS)
        unknown = stages[4]
        assert unknown.mode == "fuzzy"
        assert unknown.cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_description_becomes_instructions(self) -> None:
        stages = deerflow_skills_to_stages(_SKILL_DICTS)
        assert stages[0].instructions == "Search and summarize web content"

    def test_empty_skills_list(self) -> None:
        stages = deerflow_skills_to_stages([])
        assert stages == ()


# ---------------------------------------------------------------------------
# deerflow_to_template
# ---------------------------------------------------------------------------


class TestDeerflowToTemplate:
    """Tests for deerflow_to_template."""

    def test_returns_pattern_template(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert isinstance(result, PatternTemplate)

    def test_template_name_includes_assistant_id(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert result.name == "deerflow_lead_agent"

    def test_topology_is_specialist_swarm_for_multiple_agents(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert result.topology == "specialist_swarm"

    def test_topology_is_single_worker_for_one_or_zero_agents(self) -> None:
        cfg = {**_SESSION_CONFIG, "sub_agents": []}
        result = deerflow_to_template(cfg)
        assert result.topology == "single_worker"

    def test_stage_specs_match_sub_agents(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert len(result.stage_specs) == 2
        assert result.stage_specs[0]["name"] == "researcher"
        assert result.stage_specs[1]["name"] == "coder"

    def test_fingerprint_roles(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert "developer" in result.fingerprint.required_roles
        assert "researcher" in result.fingerprint.required_roles

    def test_fingerprint_tool_count(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert result.fingerprint.tool_count == 3

    def test_fingerprint_subtask_count(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert result.fingerprint.subtask_count == 2

    def test_tags_include_deerflow_and_sandbox(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert "deerflow" in result.tags
        assert "docker" in result.tags

    def test_intervention_policy_captures_sandbox(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert result.intervention_policy["sandbox"] == "docker"
        assert result.intervention_policy["recursion_limit"] == 100

    def test_template_id_is_nonempty(self) -> None:
        result = deerflow_to_template(_SESSION_CONFIG)
        assert len(result.template_id) == 8
