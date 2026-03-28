"""Tests for DeerFlow Markdown skill <-> PatternTemplate conversion."""

from __future__ import annotations

import pytest

from operon_ai.convergence.deerflow_skills import (
    extract_workflow_steps,
    parse_skill_frontmatter,
    skill_to_template,
    template_to_skill,
)
from operon_ai.patterns.repository import PatternTemplate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASIC_SKILL = """\
---
name: web_research
description: Search and summarize web content
version: 1.0
author: community
category: research
---

# Web Research

1. Search for relevant sources using Tavily
2. Extract key information from each source
3. Synthesize findings into a summary
4. Verify claims against multiple sources
"""

_NO_FRONTMATTER_SKILL = """\
# Quick Task

1. Gather data
2. Process results
"""

_SINGLE_STEP_SKILL = """\
---
name: simple_lookup
category: general
---

# Simple Lookup

1. Look up the answer
"""


# ---------------------------------------------------------------------------
# skill_to_template
# ---------------------------------------------------------------------------


class TestSkillToTemplateBasic:
    """Parse a well-formed DeerFlow skill and verify template fields."""

    def test_returns_pattern_template(self) -> None:
        result = skill_to_template(_BASIC_SKILL)
        assert isinstance(result, PatternTemplate)

    def test_name_from_frontmatter(self) -> None:
        result = skill_to_template(_BASIC_SKILL)
        assert result.name == "web_research"

    def test_topology_multi_step(self) -> None:
        result = skill_to_template(_BASIC_SKILL)
        assert result.topology == "skill_organism"

    def test_tags_include_deerflow_skill(self) -> None:
        result = skill_to_template(_BASIC_SKILL)
        assert "deerflow_skill" in result.tags

    def test_fingerprint_roles(self) -> None:
        result = skill_to_template(_BASIC_SKILL)
        assert "research" in result.fingerprint.required_roles


class TestSkillToTemplateNoFrontmatter:
    """Skill without YAML frontmatter delimiters."""

    def test_defaults_name_to_unnamed(self) -> None:
        result = skill_to_template(_NO_FRONTMATTER_SKILL)
        assert result.name == "unnamed_skill"

    def test_role_defaults_to_worker(self) -> None:
        result = skill_to_template(_NO_FRONTMATTER_SKILL)
        for spec in result.stage_specs:
            assert spec["role"] == "worker"

    def test_steps_still_extracted(self) -> None:
        result = skill_to_template(_NO_FRONTMATTER_SKILL)
        assert len(result.stage_specs) == 2


class TestSkillToTemplateStepCount:
    """Verify stage_specs match step count and shape heuristics."""

    def test_four_steps_give_mixed_shape(self) -> None:
        result = skill_to_template(_BASIC_SKILL)
        assert result.fingerprint.subtask_count == 4
        assert result.fingerprint.task_shape == "mixed"

    def test_two_steps_give_sequential_shape(self) -> None:
        result = skill_to_template(_NO_FRONTMATTER_SKILL)
        assert result.fingerprint.subtask_count == 2
        assert result.fingerprint.task_shape == "sequential"

    def test_single_step_gives_single_worker_topology(self) -> None:
        result = skill_to_template(_SINGLE_STEP_SKILL)
        assert result.topology == "single_worker"
        assert result.fingerprint.subtask_count == 1
        assert result.fingerprint.task_shape == "sequential"

    def test_stage_spec_count_matches_steps(self) -> None:
        result = skill_to_template(_BASIC_SKILL)
        assert len(result.stage_specs) == 4


# ---------------------------------------------------------------------------
# template_to_skill roundtrip
# ---------------------------------------------------------------------------


class TestTemplateToSkillRoundtrip:
    """Convert template to skill Markdown and parse back."""

    def test_roundtrip_preserves_name(self) -> None:
        original = skill_to_template(_BASIC_SKILL)
        md = template_to_skill(original)
        recovered = skill_to_template(md)
        assert recovered.name == original.name

    def test_roundtrip_preserves_step_count(self) -> None:
        original = skill_to_template(_BASIC_SKILL)
        md = template_to_skill(original)
        recovered = skill_to_template(md)
        assert len(recovered.stage_specs) == len(original.stage_specs)

    def test_output_contains_frontmatter(self) -> None:
        template = skill_to_template(_BASIC_SKILL)
        md = template_to_skill(template)
        assert md.startswith("---\n")
        assert "\n---\n" in md[3:]

    def test_output_contains_numbered_steps(self) -> None:
        template = skill_to_template(_BASIC_SKILL)
        md = template_to_skill(template)
        assert "1. " in md
        assert "4. " in md

    def test_roundtrip_preserves_roles(self) -> None:
        original = skill_to_template(_BASIC_SKILL)
        md = template_to_skill(original)
        recovered = skill_to_template(md)
        original_roles = tuple(s["role"] for s in original.stage_specs)
        recovered_roles = tuple(s["role"] for s in recovered.stage_specs)
        assert recovered_roles == original_roles

    def test_roundtrip_preserves_instructions(self) -> None:
        original = skill_to_template(_BASIC_SKILL)
        md = template_to_skill(original)
        recovered = skill_to_template(md)
        for orig_spec, rec_spec in zip(original.stage_specs, recovered.stage_specs):
            assert rec_spec["instructions"] == orig_spec["instructions"]

    def test_roundtrip_no_name_prefix_in_instructions(self) -> None:
        original = skill_to_template(_BASIC_SKILL)
        md = template_to_skill(original)
        recovered = skill_to_template(md)
        for spec in recovered.stage_specs:
            assert not spec["instructions"].startswith("step_")


# ---------------------------------------------------------------------------
# parse_skill_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    """Direct tests for the frontmatter parser."""

    def test_extracts_all_keys(self) -> None:
        fm = parse_skill_frontmatter(_BASIC_SKILL)
        assert fm["name"] == "web_research"
        assert fm["description"] == "Search and summarize web content"
        assert fm["version"] == "1.0"
        assert fm["author"] == "community"
        assert fm["category"] == "research"

    def test_empty_string_returns_empty_dict(self) -> None:
        assert parse_skill_frontmatter("") == {}

    def test_no_frontmatter_returns_empty_dict(self) -> None:
        assert parse_skill_frontmatter("# Just a heading\n\nSome text.") == {}


# ---------------------------------------------------------------------------
# extract_workflow_steps
# ---------------------------------------------------------------------------


class TestExtractSteps:
    """Direct tests for step extraction."""

    def test_extracts_numbered_steps(self) -> None:
        steps = extract_workflow_steps(_BASIC_SKILL)
        assert len(steps) == 4
        assert steps[0] == "Search for relevant sources using Tavily"
        assert steps[3] == "Verify claims against multiple sources"

    def test_empty_string_returns_empty_list(self) -> None:
        assert extract_workflow_steps("") == []

    def test_no_numbered_items(self) -> None:
        md = "# Title\n\nSome paragraph with no steps."
        assert extract_workflow_steps(md) == []


# ---------------------------------------------------------------------------
# Edge case: empty skill
# ---------------------------------------------------------------------------


class TestEmptySkill:
    """Empty or blank input should raise ValueError."""

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="no workflow steps"):
            skill_to_template("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="no workflow steps"):
            skill_to_template("   \n\n  ")
