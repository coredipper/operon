"""Tests for the Ralph convergence adapter."""

from __future__ import annotations

from operon_ai.convergence.ralph_adapter import (
    parse_ralph_config,
    ralph_hats_to_stages,
    ralph_to_template,
)
from operon_ai.convergence.types import ExternalTopology
from operon_ai.patterns.repository import PatternTemplate
from operon_ai.patterns.types import CognitiveMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RALPH_CONFIG: dict = {
    "backend": "claude",
    "hats": [
        {"name": "coder", "pattern": "code-assist"},
        {"name": "debugger", "pattern": "debug"},
        {"name": "reviewer", "pattern": "review"},
    ],
    "events": [
        {"from": "coder", "event": "code.failure", "to": "debugger"},
        {"from": "debugger", "event": "fix.complete", "to": "reviewer"},
    ],
    "backpressure": ["tests", "lint", "typecheck"],
    "iteration_limit": 10,
}


# ---------------------------------------------------------------------------
# parse_ralph_config
# ---------------------------------------------------------------------------


class TestParseRalphConfig:
    """Tests for parse_ralph_config."""

    def test_returns_external_topology(self) -> None:
        result = parse_ralph_config(_RALPH_CONFIG)
        assert isinstance(result, ExternalTopology)

    def test_source_is_ralph(self) -> None:
        result = parse_ralph_config(_RALPH_CONFIG)
        assert result.source == "ralph"

    def test_agents_match_hats(self) -> None:
        result = parse_ralph_config(_RALPH_CONFIG)
        names = [a["name"] for a in result.agents]
        assert names == ["coder", "debugger", "reviewer"]

    def test_edges_from_events(self) -> None:
        result = parse_ralph_config(_RALPH_CONFIG)
        assert result.edges == (("coder", "debugger"), ("debugger", "reviewer"))

    def test_metadata_contains_backend(self) -> None:
        result = parse_ralph_config(_RALPH_CONFIG)
        assert result.metadata["backend"] == "claude"

    def test_metadata_contains_backpressure(self) -> None:
        result = parse_ralph_config(_RALPH_CONFIG)
        assert result.metadata["backpressure"] == ["tests", "lint", "typecheck"]

    def test_metadata_contains_iteration_limit(self) -> None:
        result = parse_ralph_config(_RALPH_CONFIG)
        assert result.metadata["iteration_limit"] == 10

    def test_minimal_config(self) -> None:
        """Empty config yields a valid topology."""
        result = parse_ralph_config({})
        assert isinstance(result, ExternalTopology)
        assert result.source == "ralph"
        assert len(result.agents) == 0
        assert result.edges == ()


# ---------------------------------------------------------------------------
# ralph_hats_to_stages
# ---------------------------------------------------------------------------


class TestRalphHatsToStages:
    """Tests for ralph_hats_to_stages."""

    def test_returns_tuple_of_stages(self) -> None:
        hats = _RALPH_CONFIG["hats"]
        stages = ralph_hats_to_stages(hats)
        assert isinstance(stages, tuple)
        assert len(stages) == 3

    def test_code_assist_is_action_oriented(self) -> None:
        hats = [{"name": "coder", "pattern": "code-assist"}]
        stages = ralph_hats_to_stages(hats)
        assert stages[0].mode == "fuzzy"
        assert stages[0].cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_debug_is_action_oriented(self) -> None:
        hats = [{"name": "debugger", "pattern": "debug"}]
        stages = ralph_hats_to_stages(hats)
        assert stages[0].mode == "fuzzy"
        assert stages[0].cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_review_is_observational(self) -> None:
        hats = [{"name": "reviewer", "pattern": "review"}]
        stages = ralph_hats_to_stages(hats)
        assert stages[0].mode == "fixed"
        assert stages[0].cognitive_mode == CognitiveMode.OBSERVATIONAL

    def test_research_is_observational(self) -> None:
        hats = [{"name": "scholar", "pattern": "research"}]
        stages = ralph_hats_to_stages(hats)
        assert stages[0].mode == "fixed"
        assert stages[0].cognitive_mode == CognitiveMode.OBSERVATIONAL

    def test_unknown_pattern_defaults_to_fuzzy(self) -> None:
        hats = [{"name": "mystery", "pattern": "alien-hat"}]
        stages = ralph_hats_to_stages(hats)
        assert stages[0].mode == "fuzzy"
        assert stages[0].cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_pdd_to_code_assist_is_action(self) -> None:
        hats = [{"name": "pdd", "pattern": "pdd-to-code-assist"}]
        stages = ralph_hats_to_stages(hats)
        assert stages[0].mode == "fuzzy"
        assert stages[0].cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_empty_hats_list(self) -> None:
        stages = ralph_hats_to_stages([])
        assert stages == ()


# ---------------------------------------------------------------------------
# ralph_to_template
# ---------------------------------------------------------------------------


class TestRalphToTemplate:
    """Tests for ralph_to_template."""

    def test_returns_pattern_template(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert isinstance(result, PatternTemplate)

    def test_template_name_includes_backend(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert result.name == "ralph_claude"

    def test_topology_is_skill_organism_for_linear_chain(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        # 3 hats, 2 events (linear chain) -> skill_organism
        assert result.topology == "skill_organism"

    def test_topology_is_single_worker_for_one_hat(self) -> None:
        cfg = {**_RALPH_CONFIG, "hats": [{"name": "solo", "pattern": "debug"}], "events": []}
        result = ralph_to_template(cfg)
        assert result.topology == "single_worker"

    def test_stage_specs_match_hats(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert len(result.stage_specs) == 3
        assert result.stage_specs[0]["name"] == "coder"
        assert result.stage_specs[1]["name"] == "debugger"
        assert result.stage_specs[2]["name"] == "reviewer"

    def test_fingerprint_tool_count_matches_backpressure(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert result.fingerprint.tool_count == 3

    def test_fingerprint_subtask_count(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert result.fingerprint.subtask_count == 3

    def test_tags_include_ralph(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert "ralph" in result.tags

    def test_intervention_policy_captures_limits(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert result.intervention_policy["iteration_limit"] == 10
        assert result.intervention_policy["backpressure"] == ["tests", "lint", "typecheck"]

    def test_template_id_is_nonempty(self) -> None:
        result = ralph_to_template(_RALPH_CONFIG)
        assert len(result.template_id) == 8
