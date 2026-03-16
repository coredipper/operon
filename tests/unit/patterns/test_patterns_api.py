"""Tests for the pattern-first API layer."""

import pytest

from operon_ai import advise_topology, reviewer_gate, specialist_swarm
from operon_ai.core.epistemic import TopologyClass


def test_reviewer_gate_allows_custom_executor_and_reviewer():
    gate = reviewer_gate(
        executor=lambda prompt: f"EXECUTE::{prompt}",
        reviewer=lambda prompt, candidate: candidate.startswith("EXECUTE::"),
    )

    result = gate.run("Ship the patch")

    assert result.allowed is True
    assert result.output == "EXECUTE::Ship the patch"
    assert result.approval_token is not None
    assert gate.analysis.classification.topology_class == TopologyClass.CENTRALIZED


def test_reviewer_gate_blocks_when_reviewer_rejects():
    gate = reviewer_gate(
        executor=lambda prompt: f"EXECUTE::{prompt}",
        reviewer=lambda prompt, candidate: False,
    )

    result = gate.run("Drop production table")

    assert result.allowed is False
    assert result.status == "blocked"
    assert "reviewer" in result.reason.lower()


def test_reviewer_gate_cache_avoids_reexecution():
    calls = {"count": 0}

    def executor(prompt):
        calls["count"] += 1
        return f"EXECUTE::{prompt}"

    gate = reviewer_gate(
        executor=executor,
        reviewer=lambda prompt, candidate: True,
        enable_cache=True,
    )

    gate.run("Repeatable request")
    cached = gate.run("Repeatable request")

    assert calls["count"] == 1
    assert cached.raw.cached is True


def test_specialist_swarm_runs_and_analyzes():
    swarm = specialist_swarm(
        roles=["research", "risk"],
        workers={
            "research": lambda task, role: f"{role}: found two options",
            "risk": lambda task, role: f"{role}: no blockers found",
        },
        aggregator=lambda task, outputs: " || ".join(outputs.values()),
    )

    result = swarm.run("Assess vendor")

    assert result.outputs["research"] == "research: found two options"
    assert result.outputs["risk"] == "risk: no blockers found"
    assert result.aggregate == "research: found two options || risk: no blockers found"
    assert result.analysis.classification.topology_class == TopologyClass.CENTRALIZED


def test_specialist_swarm_single_argument_aggregator_receives_outputs():
    swarm = specialist_swarm(
        roles=["research", "risk"],
        workers={
            "research": lambda task, role: f"{role}: found two options",
            "risk": lambda task, role: f"{role}: no blockers found",
        },
        aggregator=lambda outputs: sorted(outputs.keys()),
    )

    result = swarm.run("Assess vendor")

    assert result.aggregate == ["research", "risk"]


def test_specialist_swarm_requires_unique_roles():
    with pytest.raises(ValueError, match="unique"):
        specialist_swarm(roles=["risk", "risk"])


def test_advise_topology_sequential_low_error_prefers_reviewer():
    advice = advise_topology(
        task_shape="sequential",
        tool_count=2,
        subtask_count=3,
        error_tolerance=0.02,
    )

    assert advice.recommended_pattern == "single_worker_with_reviewer"
    assert advice.suggested_api == "reviewer_gate(...)"


def test_advise_topology_parallel_prefers_swarm():
    advice = advise_topology(
        task_shape="parallel",
        tool_count=4,
        subtask_count=3,
        error_tolerance=0.1,
    )

    assert advice.recommended_pattern == "specialist_swarm"
    assert advice.suggested_api == "specialist_swarm(...)"
