"""Tests for the Swarms convergence adapter."""

import pytest

from operon_ai.convergence.swarms_adapter import (
    analyze_external_topology,
    parse_swarm_topology,
    swarm_to_template,
)
from operon_ai.convergence.types import AdapterResult, ExternalTopology
from operon_ai.core.epistemic import TopologyClass
from operon_ai.patterns.repository import PatternTemplate, TaskFingerprint


# ── Fixtures ──────────────────────────────────────────────────────


def _sequential_topology() -> ExternalTopology:
    """A simple three-agent sequential chain: A -> B -> C."""
    return parse_swarm_topology(
        pattern_name="SequentialWorkflow",
        agent_specs=[
            {"name": "A", "role": "researcher"},
            {"name": "B", "role": "writer"},
            {"name": "C", "role": "reviewer"},
        ],
        edges=[("A", "B"), ("B", "C")],
    )


def _concurrent_topology() -> ExternalTopology:
    """Four independent agents with no edges."""
    return parse_swarm_topology(
        pattern_name="ConcurrentWorkflow",
        agent_specs=[
            {"name": "W1", "role": "worker"},
            {"name": "W2", "role": "worker"},
            {"name": "W3", "role": "worker"},
            {"name": "W4", "role": "worker"},
        ],
        edges=[],
    )


def _large_sequential_topology() -> ExternalTopology:
    """A long sequential chain of 8 agents -- triggers sequential warnings."""
    agents = [{"name": f"S{i}", "role": "processor"} for i in range(8)]
    edges = [(f"S{i}", f"S{i+1}") for i in range(7)]
    return parse_swarm_topology(
        pattern_name="SequentialWorkflow",
        agent_specs=agents,
        edges=edges,
    )


def _hierarchical_topology() -> ExternalTopology:
    """A hierarchical swarm with a coordinator and specialists."""
    return parse_swarm_topology(
        pattern_name="HierarchicalSwarm",
        agent_specs=[
            {"name": "coordinator", "role": "coordinator"},
            {"name": "researcher", "role": "researcher"},
            {"name": "coder", "role": "coder"},
            {"name": "tester", "role": "tester"},
        ],
        edges=[
            ("coordinator", "researcher"),
            ("coordinator", "coder"),
            ("coordinator", "tester"),
            ("researcher", "coordinator"),
            ("coder", "coordinator"),
            ("tester", "coordinator"),
        ],
    )


def _topology_with_capabilities() -> ExternalTopology:
    """Agents with declared capabilities for tool density analysis."""
    return parse_swarm_topology(
        pattern_name="GraphWorkflow",
        agent_specs=[
            {"name": "search", "role": "searcher", "capabilities": ["web_search", "api_call"]},
            {"name": "code", "role": "coder", "capabilities": ["exec_code", "write_fs"]},
            {"name": "review", "role": "reviewer", "capabilities": ["read_fs"]},
        ],
        edges=[("search", "code"), ("code", "review")],
    )


# ── parse_swarm_topology ─────────────────────────────────────────


class TestParseSwarmTopology:
    """Tests for parse_swarm_topology."""

    def test_produces_valid_external_topology(self):
        topo = _sequential_topology()
        assert isinstance(topo, ExternalTopology)
        assert topo.source == "swarms"
        assert topo.pattern_name == "SequentialWorkflow"
        assert len(topo.agents) == 3
        assert len(topo.edges) == 2

    def test_agents_are_tuples_of_dicts(self):
        topo = _sequential_topology()
        assert isinstance(topo.agents, tuple)
        for agent in topo.agents:
            assert isinstance(agent, dict)
            assert "name" in agent
            assert "role" in agent

    def test_edges_are_tuples_of_pairs(self):
        topo = _sequential_topology()
        assert isinstance(topo.edges, tuple)
        for edge in topo.edges:
            assert len(edge) == 2

    def test_metadata_forwarded(self):
        topo = parse_swarm_topology(
            pattern_name="SequentialWorkflow",
            agent_specs=[{"name": "A", "role": "worker"}],
            edges=[],
            sandbox="docker",
            max_retries=3,
        )
        assert topo.metadata["sandbox"] == "docker"
        assert topo.metadata["max_retries"] == 3

    def test_concurrent_no_edges(self):
        topo = _concurrent_topology()
        assert len(topo.edges) == 0
        assert len(topo.agents) == 4


# ── analyze_external_topology ────────────────────────────────────


class TestAnalyzeExternalTopology:
    """Tests for analyze_external_topology."""

    def test_returns_adapter_result(self):
        result = analyze_external_topology(_sequential_topology())
        assert isinstance(result, AdapterResult)

    def test_simple_sequential_low_risk(self):
        """A short sequential chain should have a relatively low risk score."""
        result = analyze_external_topology(_sequential_topology())
        assert result.risk_score < 0.5

    def test_large_sequential_warns_overhead(self):
        """A long sequential chain should warn about sequential overhead."""
        result = analyze_external_topology(_large_sequential_topology())
        overhead_warnings = [w for w in result.warnings if "Sequential overhead" in w]
        assert len(overhead_warnings) > 0

    def test_large_sequential_higher_risk(self):
        """A long chain should score higher risk than a short one."""
        short = analyze_external_topology(_sequential_topology())
        long = analyze_external_topology(_large_sequential_topology())
        assert long.risk_score > short.risk_score

    def test_concurrent_topology_no_sequential_warning(self):
        """Concurrent (no edges) should not trigger sequential overhead warnings."""
        result = analyze_external_topology(_concurrent_topology())
        overhead_warnings = [w for w in result.warnings if "Sequential overhead" in w]
        assert len(overhead_warnings) == 0

    def test_hierarchical_topology_returns_result(self):
        result = analyze_external_topology(_hierarchical_topology())
        assert isinstance(result, AdapterResult)
        assert result.risk_score >= 0.0
        assert result.risk_score <= 1.0

    def test_advice_has_recommended_pattern(self):
        result = analyze_external_topology(_sequential_topology())
        assert result.topology_advice.recommended_pattern in (
            "single_worker",
            "skill_organism",
            "specialist_swarm",
            "reviewer_gate",
        )

    def test_suggested_template_is_present(self):
        result = analyze_external_topology(_sequential_topology())
        assert result.suggested_template is not None
        assert isinstance(result.suggested_template, PatternTemplate)

    def test_many_agents_warns_error_amplification(self):
        """With enough agents the centralized error bound exceeds the threshold."""
        agents = [{"name": f"A{i}", "role": "worker"} for i in range(20)]
        edges = [(f"A{i}", f"A{i+1}") for i in range(19)]
        topo = parse_swarm_topology("SequentialWorkflow", agents, edges)
        result = analyze_external_topology(topo)
        error_warnings = [w for w in result.warnings if "Error amplification" in w]
        assert len(error_warnings) > 0


# ── swarm_to_template ────────────────────────────────────────────


class TestSwarmToTemplate:
    """Tests for swarm_to_template."""

    def test_produces_valid_pattern_template(self):
        template = swarm_to_template(_sequential_topology())
        assert isinstance(template, PatternTemplate)
        assert template.name == "swarms_SequentialWorkflow"
        assert "swarms" in template.tags

    def test_stage_specs_match_agents(self):
        topo = _sequential_topology()
        template = swarm_to_template(topo)
        assert len(template.stage_specs) == len(topo.agents)
        names = {s["name"] for s in template.stage_specs}
        assert names == {"A", "B", "C"}

    def test_fingerprint_shape_from_pattern(self):
        template = swarm_to_template(_sequential_topology())
        assert template.fingerprint.task_shape == "sequential"

        template_c = swarm_to_template(_concurrent_topology())
        assert template_c.fingerprint.task_shape == "parallel"

    def test_fingerprint_has_roles(self):
        template = swarm_to_template(_sequential_topology())
        assert "researcher" in template.fingerprint.required_roles
        assert "writer" in template.fingerprint.required_roles

    def test_template_with_capabilities_counts_tools(self):
        template = swarm_to_template(_topology_with_capabilities())
        assert template.fingerprint.tool_count == 5  # web_search, api_call, exec_code, write_fs, read_fs

    def test_unique_template_ids(self):
        t1 = swarm_to_template(_sequential_topology())
        t2 = swarm_to_template(_sequential_topology())
        assert t1.template_id != t2.template_id
