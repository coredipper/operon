"""Tests for operon_ai.convergence.catalog -- PatternLibrary seeders."""

from __future__ import annotations

import pytest

from operon_ai.convergence.catalog import (
    get_builtin_swarms_patterns,
    seed_library_from_acg_survey,
    seed_library_from_deerflow,
    seed_library_from_swarms,
)
from operon_ai.patterns.repository import PatternLibrary, TaskFingerprint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def library() -> PatternLibrary:
    return PatternLibrary()


def _make_swarms_patterns(n: int = 3) -> list[dict]:
    """Build *n* minimal Swarms-style pattern dicts."""
    patterns = get_builtin_swarms_patterns()
    return patterns[:n]


def _make_deerflow_sessions(n: int = 2) -> list[dict]:
    """Build *n* minimal DeerFlow session config dicts."""
    sessions = []
    for i in range(n):
        sessions.append({
            "assistant_id": f"lead_{i}",
            "skills": ["web_search", "summarization"],
            "sub_agents": [
                {"name": f"researcher_{i}", "role": "researcher", "skills": ["search"]},
                {"name": f"writer_{i}", "role": "writer", "skills": ["writing"]},
            ],
            "recursion_limit": 50,
            "sandbox": "local",
            "config": {},
        })
    return sessions


# ---------------------------------------------------------------------------
# Swarms seeder tests
# ---------------------------------------------------------------------------


class TestSeedFromSwarms:
    def test_seed_from_swarms_registers_templates(self, library: PatternLibrary) -> None:
        patterns = _make_swarms_patterns(3)
        seed_library_from_swarms(library, patterns)
        assert library.summary()["template_count"] == 3

    def test_seed_from_swarms_returns_count(self, library: PatternLibrary) -> None:
        patterns = _make_swarms_patterns(3)
        count = seed_library_from_swarms(library, patterns)
        assert count == 3

    def test_seed_from_swarms_empty_list(self, library: PatternLibrary) -> None:
        count = seed_library_from_swarms(library, [])
        assert count == 0
        assert library.summary()["template_count"] == 0


# ---------------------------------------------------------------------------
# DeerFlow seeder tests
# ---------------------------------------------------------------------------


class TestSeedFromDeerflow:
    def test_seed_from_deerflow_registers_templates(self, library: PatternLibrary) -> None:
        sessions = _make_deerflow_sessions(2)
        seed_library_from_deerflow(library, sessions)
        assert library.summary()["template_count"] == 2

    def test_seed_from_deerflow_returns_count(self, library: PatternLibrary) -> None:
        sessions = _make_deerflow_sessions(2)
        count = seed_library_from_deerflow(library, sessions)
        assert count == 2


# ---------------------------------------------------------------------------
# Atomic skills seeder tests
# ---------------------------------------------------------------------------


class TestSeedFromAtomicSkills:
    def test_seed_registers_all_five(self, library: PatternLibrary) -> None:
        from operon_ai.convergence.catalog import seed_library_from_atomic_skills
        count = seed_library_from_atomic_skills(library)
        assert count == 5
        assert library.summary()["template_count"] == 5

    def test_seed_returns_count(self, library: PatternLibrary) -> None:
        from operon_ai.convergence.catalog import seed_library_from_atomic_skills
        count = seed_library_from_atomic_skills(library)
        assert count == 5

    def test_seed_empty_override(self, library: PatternLibrary) -> None:
        from operon_ai.convergence.catalog import seed_library_from_atomic_skills
        count = seed_library_from_atomic_skills(library, patterns=[])
        assert count == 0

    def test_templates_have_atomic_skill_tag(self, library: PatternLibrary) -> None:
        from operon_ai.convergence.catalog import seed_library_from_atomic_skills
        seed_library_from_atomic_skills(library)
        templates = library.retrieve_templates(tags=("atomic_skill",))
        assert len(templates) == 5
        assert all("atomic_skill" in t.tags for t in templates)

    def test_get_atomic_skill_patterns(self) -> None:
        from operon_ai.convergence.catalog import get_atomic_skill_patterns
        patterns = get_atomic_skill_patterns()
        assert len(patterns) == 5
        names = {p["name"] for p in patterns}
        assert names == {"localize", "edit", "test", "reproduce", "review"}

    def test_review_topology_is_specialist_swarm(self, library: PatternLibrary) -> None:
        """Parallel review skill must map to specialist_swarm, not skill_organism."""
        from operon_ai.convergence.catalog import seed_library_from_atomic_skills
        seed_library_from_atomic_skills(library)
        reviews = library.retrieve_templates(tags=("review", "atomic_skill"))
        assert len(reviews) == 1
        assert reviews[0].topology == "specialist_swarm"

    def test_sequential_skills_are_skill_organism(self, library: PatternLibrary) -> None:
        """Sequential atomic skills must map to skill_organism."""
        from operon_ai.convergence.catalog import seed_library_from_atomic_skills
        seed_library_from_atomic_skills(library)
        seq = library.retrieve_templates(tags=("localization", "atomic_skill"))
        assert len(seq) == 1
        assert seq[0].topology == "skill_organism"


# ---------------------------------------------------------------------------
# ACG survey seeder tests
# ---------------------------------------------------------------------------


class TestSeedFromAcgSurvey:
    def test_seed_from_acg_survey_registers_all(self, library: PatternLibrary) -> None:
        count = seed_library_from_acg_survey(library)
        assert count >= 8
        assert library.summary()["template_count"] >= 8

    def test_acg_templates_have_survey_tag(self, library: PatternLibrary) -> None:
        seed_library_from_acg_survey(library)
        templates = library.retrieve_templates(tags=("acg_survey",))
        assert len(templates) >= 8
        for t in templates:
            assert "acg_survey" in t.tags

    def test_acg_templates_carry_metadata_tags(self, library: PatternLibrary) -> None:
        seed_library_from_acg_survey(library)
        templates = library.retrieve_templates(tags=("acg_survey",))
        for t in templates:
            determination_tags = [tag for tag in t.tags if tag.startswith("determination:")]
            plasticity_tags = [tag for tag in t.tags if tag.startswith("plasticity:")]
            assert len(determination_tags) == 1, f"Missing determination tag on {t.name}"
            assert len(plasticity_tags) == 1, f"Missing plasticity tag on {t.name}"


# ---------------------------------------------------------------------------
# Builtin patterns helper
# ---------------------------------------------------------------------------


class TestGetBuiltinSwarmsPatterns:
    def test_get_builtin_swarms_patterns_returns_list(self) -> None:
        patterns = get_builtin_swarms_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 10

    def test_each_pattern_has_required_keys(self) -> None:
        for pat in get_builtin_swarms_patterns():
            assert "name" in pat
            assert "agents" in pat
            assert "edges" in pat
            assert isinstance(pat["agents"], list)
            assert len(pat["agents"]) >= 2

    def test_agents_have_required_fields(self) -> None:
        for pat in get_builtin_swarms_patterns():
            for agent in pat["agents"]:
                assert "name" in agent
                assert "role" in agent
                assert "skills" in agent


# ---------------------------------------------------------------------------
# Ranking integration test
# ---------------------------------------------------------------------------


class TestSeededLibraryRanking:
    def test_seeded_library_ranks_by_fingerprint(self, library: PatternLibrary) -> None:
        """Seed from all sources, then verify top_templates_for returns results."""
        seed_library_from_swarms(library, get_builtin_swarms_patterns())
        seed_library_from_deerflow(library, _make_deerflow_sessions(2))
        seed_library_from_acg_survey(library)

        # Query for a sequential, small-team task.
        query = TaskFingerprint(
            task_shape="sequential",
            tool_count=2,
            subtask_count=3,
            required_roles=("worker",),
            tags=("swarms",),
        )
        ranked = library.top_templates_for(query, limit=5)

        assert len(ranked) > 0
        # Each entry is (template, score).
        for template, score in ranked:
            assert 0.0 <= score <= 1.0
            assert template.name  # non-empty
