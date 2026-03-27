"""Tests for the AnimaWorks -> Operon convergence adapter."""

from __future__ import annotations

import pytest

from operon_ai.convergence.animaworks_adapter import (
    animaworks_roles_to_stages,
    animaworks_to_template,
    parse_animaworks_org,
)
from operon_ai.convergence.types import ExternalTopology
from operon_ai.patterns.repository import PatternTemplate
from operon_ai.patterns.types import CognitiveMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hierarchical_org() -> dict:
    return {
        "name": "engineering_team",
        "supervisor": {"name": "manager", "role": "manager"},
        "agents": [
            {"name": "engineer_1", "role": "engineer", "skills": ["python", "testing"]},
            {"name": "writer", "role": "writer", "skills": ["documentation"]},
        ],
        "communication": "hierarchical",
    }


@pytest.fixture()
def flat_org() -> dict:
    return {
        "name": "flat_team",
        "supervisor": {"name": "lead", "role": "lead"},
        "agents": [
            {"name": "dev_a", "role": "developer"},
            {"name": "dev_b", "role": "coder"},
            {"name": "dev_c", "role": "engineer"},
        ],
        "communication": "flat",
    }


@pytest.fixture()
def ring_org() -> dict:
    return {
        "name": "review_ring",
        "supervisor": {"name": "auditor", "role": "auditor"},
        "agents": [
            {"name": "analyst_1", "role": "analyst"},
            {"name": "analyst_2", "role": "analyst"},
            {"name": "analyst_3", "role": "analyst"},
        ],
        "communication": "ring",
    }


# ---------------------------------------------------------------------------
# parse_animaworks_org
# ---------------------------------------------------------------------------

class TestParseAnimaworksOrg:
    def test_hierarchical_edges(self, hierarchical_org: dict) -> None:
        topo = parse_animaworks_org(hierarchical_org)

        assert isinstance(topo, ExternalTopology)
        assert topo.source == "animaworks"
        assert topo.pattern_name == "engineering_team"
        # Supervisor -> each agent
        assert ("manager", "engineer_1") in topo.edges
        assert ("manager", "writer") in topo.edges
        assert len(topo.edges) == 2

    def test_agents_include_supervisor(self, hierarchical_org: dict) -> None:
        topo = parse_animaworks_org(hierarchical_org)
        names = {a["name"] for a in topo.agents}
        assert "manager" in names
        assert "engineer_1" in names
        assert "writer" in names
        assert len(topo.agents) == 3

    def test_flat_communication_edges(self, flat_org: dict) -> None:
        topo = parse_animaworks_org(flat_org)
        # Flat: every agent pair gets bidirectional edges (no supervisor edges)
        agent_names = ["dev_a", "dev_b", "dev_c"]
        for i, a in enumerate(agent_names):
            for b in agent_names[i + 1:]:
                assert (a, b) in topo.edges
                assert (b, a) in topo.edges

    def test_ring_communication_edges(self, ring_org: dict) -> None:
        topo = parse_animaworks_org(ring_org)
        # Ring: supervisor -> first, then each -> next, last -> first
        assert ("auditor", "analyst_1") in topo.edges
        assert ("analyst_1", "analyst_2") in topo.edges
        assert ("analyst_2", "analyst_3") in topo.edges
        assert ("analyst_3", "analyst_1") in topo.edges
        assert len(topo.edges) == 4

    def test_metadata_contains_communication(self, hierarchical_org: dict) -> None:
        topo = parse_animaworks_org(hierarchical_org)
        assert topo.metadata["communication"] == "hierarchical"


# ---------------------------------------------------------------------------
# animaworks_roles_to_stages
# ---------------------------------------------------------------------------

class TestAnimaworksRolesToStages:
    def test_known_action_roles(self) -> None:
        roles = [
            {"name": "eng", "role": "engineer", "skills": ["python"]},
            {"name": "mgr", "role": "manager"},
            {"name": "doc", "role": "writer", "skills": ["docs"]},
        ]
        stages = animaworks_roles_to_stages(roles)

        assert len(stages) == 3
        for stage in stages:
            assert stage.mode == "fuzzy"
            assert stage.cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_known_observational_roles(self) -> None:
        roles = [
            {"name": "rev", "role": "reviewer"},
            {"name": "aud", "role": "auditor"},
            {"name": "ana", "role": "analyst"},
        ]
        stages = animaworks_roles_to_stages(roles)

        assert len(stages) == 3
        for stage in stages:
            assert stage.mode == "fixed"
            assert stage.cognitive_mode == CognitiveMode.OBSERVATIONAL

    def test_unknown_role_defaults_to_fuzzy(self) -> None:
        roles = [{"name": "x", "role": "quantum_navigator"}]
        stages = animaworks_roles_to_stages(roles)

        assert len(stages) == 1
        assert stages[0].mode == "fuzzy"
        assert stages[0].cognitive_mode == CognitiveMode.ACTION_ORIENTED

    def test_skills_become_instructions(self) -> None:
        roles = [{"name": "eng", "role": "engineer", "skills": ["python", "testing"]}]
        stages = animaworks_roles_to_stages(roles)
        assert stages[0].instructions == "python, testing"

    def test_handler_returns_placeholder(self) -> None:
        roles = [{"name": "w", "role": "worker"}]
        stages = animaworks_roles_to_stages(roles)
        assert stages[0].handler is not None
        result = stages[0].handler()
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# animaworks_to_template
# ---------------------------------------------------------------------------

class TestAnimaworksToTemplate:
    def test_produces_valid_pattern_template(self, hierarchical_org: dict) -> None:
        template = animaworks_to_template(hierarchical_org)

        assert isinstance(template, PatternTemplate)
        assert template.name == "animaworks_engineering_team"
        assert "animaworks" in template.tags
        assert template.topology == "specialist_swarm"

    def test_stage_specs_match_roles(self, hierarchical_org: dict) -> None:
        template = animaworks_to_template(hierarchical_org)
        spec_names = {s["name"] for s in template.stage_specs}
        assert spec_names == {"manager", "engineer_1", "writer"}

    def test_fingerprint_roles(self, hierarchical_org: dict) -> None:
        template = animaworks_to_template(hierarchical_org)
        fp = template.fingerprint
        assert "manager" in fp.required_roles
        assert "engineer" in fp.required_roles
        assert "writer" in fp.required_roles

    def test_ring_topology_type(self, ring_org: dict) -> None:
        template = animaworks_to_template(ring_org)
        assert template.topology == "skill_organism"

    def test_flat_fingerprint_shape(self, flat_org: dict) -> None:
        template = animaworks_to_template(flat_org)
        assert template.fingerprint.task_shape == "parallel"
