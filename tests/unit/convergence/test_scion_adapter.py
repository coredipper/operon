"""Tests for the Scion convergence adapter."""

from __future__ import annotations

from operon_ai.convergence.scion_adapter import (
    parse_scion_grove,
    scion_agents_to_stages,
    scion_to_template,
)
from operon_ai.convergence.scion_compiler import organism_to_scion
from operon_ai.convergence.types import ExternalTopology
from operon_ai.patterns.repository import PatternTemplate
from operon_ai.patterns.types import SkillStage
from operon_ai import MockProvider, Nucleus, skill_organism


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GROVE_CONFIG: dict = {
    "grove": {"name": "test-grove", "runtime": "docker"},
    "agents": [
        {
            "name": "researcher_0",
            "role": "researcher",
            "template": {
                "system_prompt": "You are the researcher agent.",
                "skills": ["web_search", "summarization"],
            },
            "runtime_profile": "medium",
            "isolation": {"git_worktree": True, "credentials": "isolated"},
        },
        {
            "name": "writer_1",
            "role": "writer",
            "template": {
                "system_prompt": "You are the writer agent.",
                "skills": ["writing", "editing"],
            },
            "runtime_profile": "medium",
            "isolation": {"git_worktree": True, "credentials": "isolated"},
        },
        {
            "name": "reviewer_2",
            "role": "reviewer",
            "template": {
                "system_prompt": "You are the reviewer agent.",
                "skills": ["review", "verification"],
            },
            "runtime_profile": "medium",
            "isolation": {"git_worktree": True, "credentials": "isolated"},
        },
        {
            "name": "operon-watcher",
            "template": {
                "system_prompt": "Monitor agent execution.",
                "skills": ["telemetry_monitoring", "convergence_detection"],
            },
            "runtime_profile": "medium",
            "isolation": {"git_worktree": False, "credentials": "shared"},
        },
    ],
    "messaging": [
        {"from": "researcher_0", "to": "writer_1", "channel": "research_to_write"},
        {"from": "writer_1", "to": "reviewer_2", "channel": "write_to_review"},
    ],
    "watcher": {
        "enabled": True,
        "agent_name": "operon-watcher",
        "telemetry": "otel",
    },
}

# Config without explicit role field (name-only, tests suffix stripping)
_GROVE_NO_ROLE: dict = {
    "grove": {"name": "no-role-grove", "runtime": "docker"},
    "agents": [
        {
            "name": "planner_0",
            "template": {"system_prompt": "Plan.", "skills": ["planning"]},
            "runtime_profile": "medium",
            "isolation": {"git_worktree": True, "credentials": "isolated"},
        },
        {
            "name": "coder_1",
            "template": {"system_prompt": "Code.", "skills": ["code_generation"]},
            "runtime_profile": "medium",
            "isolation": {"git_worktree": True, "credentials": "isolated"},
        },
    ],
    "messaging": [{"from": "planner_0", "to": "coder_1", "channel": "ch"}],
    "watcher": {"enabled": False, "agent_name": "operon-watcher"},
}


# ---------------------------------------------------------------------------
# parse_scion_grove
# ---------------------------------------------------------------------------


def test_parse_scion_grove_basic():
    topo = parse_scion_grove(_GROVE_CONFIG)
    assert isinstance(topo, ExternalTopology)
    assert topo.source == "scion"
    assert topo.pattern_name == "ScionGrove"


def test_parse_scion_grove_filters_watcher():
    topo = parse_scion_grove(_GROVE_CONFIG)
    agent_names = [a["name"] for a in topo.agents]
    assert "operon-watcher" not in agent_names
    assert len(topo.agents) == 3


def test_parse_scion_grove_preserves_edges():
    topo = parse_scion_grove(_GROVE_CONFIG)
    assert len(topo.edges) == 2
    assert ("researcher_0", "writer_1") in topo.edges
    assert ("writer_1", "reviewer_2") in topo.edges


def test_parse_scion_grove_resolves_role_from_field():
    topo = parse_scion_grove(_GROVE_CONFIG)
    roles = [a["role"] for a in topo.agents]
    assert roles == ["researcher", "writer", "reviewer"]


def test_parse_scion_grove_resolves_role_by_stripping_suffix():
    topo = parse_scion_grove(_GROVE_NO_ROLE)
    roles = [a["role"] for a in topo.agents]
    assert roles == ["planner", "coder"]


def test_parse_scion_grove_metadata():
    topo = parse_scion_grove(_GROVE_CONFIG)
    assert topo.metadata["grove_name"] == "test-grove"
    assert topo.metadata["runtime"] == "docker"
    assert topo.metadata["watcher_enabled"] is True


# ---------------------------------------------------------------------------
# scion_agents_to_stages
# ---------------------------------------------------------------------------


def test_scion_agents_to_stages_basic():
    stages = scion_agents_to_stages(_GROVE_CONFIG["agents"])
    assert len(stages) == 3  # watcher filtered
    assert all(isinstance(s, SkillStage) for s in stages)


def test_scion_agents_to_stages_filters_watcher():
    stages = scion_agents_to_stages(_GROVE_CONFIG["agents"])
    names = [s.name for s in stages]
    assert "operon-watcher" not in names


def test_scion_agents_to_stages_resolves_roles():
    stages = scion_agents_to_stages(_GROVE_CONFIG["agents"])
    roles = [s.role for s in stages]
    assert roles == ["researcher", "writer", "reviewer"]


def test_scion_agents_to_stages_mode_from_skills():
    stages = scion_agents_to_stages(_GROVE_CONFIG["agents"])
    # reviewer has "review" + "verification" -> observational -> fixed
    reviewer = next(s for s in stages if s.role == "reviewer")
    assert reviewer.mode == "fixed"


def test_scion_agents_to_stages_strips_suffix_without_role():
    stages = scion_agents_to_stages(_GROVE_NO_ROLE["agents"])
    roles = [s.role for s in stages]
    assert roles == ["planner", "coder"]


# ---------------------------------------------------------------------------
# scion_to_template
# ---------------------------------------------------------------------------


def test_scion_to_template_basic():
    tmpl = scion_to_template(_GROVE_CONFIG)
    assert isinstance(tmpl, PatternTemplate)
    assert "scion" in tmpl.tags


def test_scion_to_template_roles():
    tmpl = scion_to_template(_GROVE_CONFIG)
    roles = tmpl.fingerprint.required_roles
    assert "researcher" in roles
    assert "writer" in roles
    assert "reviewer" in roles


def test_scion_to_template_excludes_watcher():
    tmpl = scion_to_template(_GROVE_CONFIG)
    stage_names = [s["name"] for s in tmpl.stage_specs]
    assert "operon-watcher" not in stage_names


# ---------------------------------------------------------------------------
# Round-trip: organism_to_scion -> parse_scion_grove
# ---------------------------------------------------------------------------


def test_compiler_adapter_round_trip():
    """Compile an organism to Scion, then parse back — roles should survive."""
    nucleus = Nucleus(provider=MockProvider(responses={}))
    stages = [
        SkillStage(name="analyst_0", role="analyst", instructions="Analyze data."),
        SkillStage(name="reporter_1", role="reporter", instructions="Write report."),
    ]
    org = skill_organism(stages=stages, fast_nucleus=nucleus, deep_nucleus=nucleus)

    compiled = organism_to_scion(org)
    topo = parse_scion_grove(compiled)

    assert len(topo.agents) == 2
    roles = [a["role"] for a in topo.agents]
    assert "analyst" in roles
    assert "reporter" in roles
    assert "operon-watcher" not in [a["name"] for a in topo.agents]


def test_compiler_adapter_round_trip_stages():
    """Compile then convert back to stages — roles should survive."""
    nucleus = Nucleus(provider=MockProvider(responses={}))
    stages = [
        SkillStage(name="planner_0", role="planner", instructions="Plan."),
        SkillStage(name="executor_1", role="executor", instructions="Execute."),
    ]
    org = skill_organism(stages=stages, fast_nucleus=nucleus, deep_nucleus=nucleus)

    compiled = organism_to_scion(org)
    recovered = scion_agents_to_stages(compiled["agents"])

    assert len(recovered) == 2
    assert recovered[0].role == "planner"
    assert recovered[1].role == "executor"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_custom_watcher_name_filtered():
    """A grove with a non-default watcher name should still filter it."""
    config = {
        "grove": {"name": "custom-watcher", "runtime": "docker"},
        "agents": [
            {"name": "worker", "role": "worker", "template": {"skills": []}},
            {"name": "my-monitor", "template": {"skills": ["telemetry_monitoring"]}},
        ],
        "messaging": [],
        "watcher": {"enabled": True, "agent_name": "my-monitor", "telemetry": "otel"},
    }
    topo = parse_scion_grove(config)
    assert len(topo.agents) == 1
    assert topo.agents[0]["name"] == "worker"

    # scion_agents_to_stages with explicit watcher_name
    stages = scion_agents_to_stages(config["agents"], watcher_name="my-monitor")
    assert len(stages) == 1
    assert stages[0].role == "worker"


def test_resolve_role_none_falls_back():
    """role=None should fall back to name-based resolution."""
    stages = scion_agents_to_stages([
        {"name": "planner_0", "role": None, "template": {"skills": []}},
    ])
    assert stages[0].role == "planner"


def test_resolve_role_empty_string_falls_back():
    """role='' should fall back to name-based resolution."""
    stages = scion_agents_to_stages([
        {"name": "coder_1", "role": "", "template": {"skills": []}},
    ])
    assert stages[0].role == "coder"


def test_resolve_role_whitespace_falls_back():
    """role='  ' should fall back to name-based resolution."""
    stages = scion_agents_to_stages([
        {"name": "reviewer_2", "role": "   ", "template": {"skills": []}},
    ])
    assert stages[0].role == "reviewer"
