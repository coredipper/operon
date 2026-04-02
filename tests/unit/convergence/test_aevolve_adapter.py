"""Tests for the A-Evolve convergence adapter."""

from __future__ import annotations

from operon_ai.convergence.aevolve_adapter import (
    aevolve_skills_to_stages,
    aevolve_to_template,
    parse_aevolve_workspace,
)
from operon_ai.convergence.aevolve_skills import (
    import_aevolve_skills,
    seed_library_from_aevolve,
)
from operon_ai.convergence.types import ExternalTopology
from operon_ai.patterns.repository import PatternLibrary, PatternTemplate
from operon_ai.patterns.types import CognitiveMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WORKSPACE_MANIFEST: dict = {
    "name": "swe-agent",
    "entrypoints": {"solve": "solve.py"},
    "skills": ["bash_exec", "file_edit", "git_ops"],
    "memory": {
        "episodic": "memory/episodic.jsonl",
        "semantic": "memory/semantic.jsonl",
    },
    "evolution": {
        "algorithm": "adaptive_evolve",
        "gate": "holdout",
    },
}

_SKILL_DICTS: list[dict] = [
    {"name": "bash_exec", "description": "Execute bash commands", "category": "execution"},
    {"name": "file_edit", "description": "Edit source files", "category": "editing"},
    {"name": "git_ops", "description": "Git operations", "category": "execution"},
    {"name": "code_review", "description": "Review code changes", "category": "review"},
    {"name": "web_search", "description": "Search the web", "category": "research"},
    {"name": "unknown_tool", "description": "Mystery tool", "category": "alien"},
]


# ---------------------------------------------------------------------------
# parse_aevolve_workspace
# ---------------------------------------------------------------------------


class TestParseAevolveWorkspace:
    """Tests for parse_aevolve_workspace."""

    def test_returns_external_topology(self) -> None:
        result = parse_aevolve_workspace(_WORKSPACE_MANIFEST)
        assert isinstance(result, ExternalTopology)

    def test_source_is_aevolve(self) -> None:
        result = parse_aevolve_workspace(_WORKSPACE_MANIFEST)
        assert result.source == "aevolve"

    def test_single_agent(self) -> None:
        result = parse_aevolve_workspace(_WORKSPACE_MANIFEST)
        assert len(result.agents) == 1
        assert result.agents[0]["name"] == "swe-agent"

    def test_no_edges(self) -> None:
        result = parse_aevolve_workspace(_WORKSPACE_MANIFEST)
        assert result.edges == ()

    def test_skills_in_metadata(self) -> None:
        result = parse_aevolve_workspace(_WORKSPACE_MANIFEST)
        assert result.metadata["skills"] == ["bash_exec", "file_edit", "git_ops"]

    def test_agent_capabilities(self) -> None:
        result = parse_aevolve_workspace(_WORKSPACE_MANIFEST)
        assert result.agents[0]["capabilities"] == ["bash_exec", "file_edit", "git_ops"]

    def test_evolution_in_metadata(self) -> None:
        result = parse_aevolve_workspace(_WORKSPACE_MANIFEST)
        assert result.metadata["evolution"]["algorithm"] == "adaptive_evolve"

    def test_minimal_manifest(self) -> None:
        """Empty manifest yields a valid topology."""
        result = parse_aevolve_workspace({})
        assert isinstance(result, ExternalTopology)
        assert result.source == "aevolve"
        assert len(result.agents) == 1
        assert result.edges == ()


# ---------------------------------------------------------------------------
# aevolve_skills_to_stages
# ---------------------------------------------------------------------------


class TestAevolveSkillsToStages:
    """Tests for aevolve_skills_to_stages."""

    def test_returns_tuple_of_stages(self) -> None:
        stages = aevolve_skills_to_stages(_SKILL_DICTS)
        assert isinstance(stages, tuple)
        assert len(stages) == 6

    def test_execution_is_action_oriented(self) -> None:
        stages = aevolve_skills_to_stages(_SKILL_DICTS)
        bash = stages[0]
        assert bash.name == "bash_exec"
        assert bash.mode == "fuzzy"
        assert bash.cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_editing_is_action_oriented(self) -> None:
        stages = aevolve_skills_to_stages(_SKILL_DICTS)
        edit = stages[1]
        assert edit.mode == "fuzzy"
        assert edit.cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_review_is_observational(self) -> None:
        stages = aevolve_skills_to_stages(_SKILL_DICTS)
        review = stages[3]
        assert review.name == "code_review"
        assert review.mode == "fixed"
        assert review.cognitive_mode == CognitiveMode.OBSERVATIONAL

    def test_research_is_observational(self) -> None:
        stages = aevolve_skills_to_stages(_SKILL_DICTS)
        search = stages[4]
        assert search.name == "web_search"
        assert search.mode == "fixed"
        assert search.cognitive_mode == CognitiveMode.OBSERVATIONAL

    def test_unknown_category_defaults_to_action(self) -> None:
        stages = aevolve_skills_to_stages(_SKILL_DICTS)
        unknown = stages[5]
        assert unknown.mode == "fuzzy"
        assert unknown.cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_description_becomes_instructions(self) -> None:
        stages = aevolve_skills_to_stages(_SKILL_DICTS)
        assert stages[0].instructions == "Execute bash commands"

    def test_empty_skills_list(self) -> None:
        stages = aevolve_skills_to_stages([])
        assert stages == ()


# ---------------------------------------------------------------------------
# aevolve_to_template
# ---------------------------------------------------------------------------


class TestAevolveToTemplate:
    """Tests for aevolve_to_template."""

    def test_returns_pattern_template(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert isinstance(result, PatternTemplate)

    def test_template_name_includes_workspace(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert result.name == "aevolve_swe-agent"

    def test_topology_is_single_worker(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert result.topology == "single_worker"

    def test_single_stage_spec(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert len(result.stage_specs) == 1
        assert result.stage_specs[0]["name"] == "swe-agent"

    def test_fingerprint_tool_count(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert result.fingerprint.tool_count == 3

    def test_tags_include_aevolve(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert "aevolve" in result.tags

    def test_intervention_policy_captures_evolution(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert result.intervention_policy["evolution_algorithm"] == "adaptive_evolve"
        assert result.intervention_policy["evolution_gate"] == "holdout"

    def test_template_id_is_nonempty(self) -> None:
        result = aevolve_to_template(_WORKSPACE_MANIFEST)
        assert len(result.template_id) == 8


# ---------------------------------------------------------------------------
# seed_library_from_aevolve
# ---------------------------------------------------------------------------


class TestSeedLibraryFromAevolve:
    """Tests for seed_library_from_aevolve."""

    def test_registers_templates(self) -> None:
        library = PatternLibrary()
        count = seed_library_from_aevolve(library, [_WORKSPACE_MANIFEST])
        assert count == 1
        assert library.summary()["template_count"] == 1

    def test_multiple_workspaces(self) -> None:
        library = PatternLibrary()
        ws2 = {**_WORKSPACE_MANIFEST, "name": "code-agent"}
        count = seed_library_from_aevolve(library, [_WORKSPACE_MANIFEST, ws2])
        assert count == 2

    def test_empty_list(self) -> None:
        library = PatternLibrary()
        count = seed_library_from_aevolve(library, [])
        assert count == 0


# ---------------------------------------------------------------------------
# import_aevolve_skills
# ---------------------------------------------------------------------------


class TestImportAevolveSkills:
    """Tests for import_aevolve_skills."""

    def test_imports_valid_skill_md(self) -> None:
        skill_md = "---\nname: test_skill\ncategory: execution\n---\n\n# Test\n\n1. Run tests\n2. Report results\n"
        library = PatternLibrary()
        count = import_aevolve_skills([skill_md], library)
        assert count == 1

    def test_skips_invalid_skills(self) -> None:
        # No numbered steps -> ValueError from skill_to_template
        bad_md = "---\nname: bad\n---\n\nNo steps here.\n"
        library = PatternLibrary()
        count = import_aevolve_skills([bad_md], library)
        assert count == 0
