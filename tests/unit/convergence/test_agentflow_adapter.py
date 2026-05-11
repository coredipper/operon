"""Tests for the agentflow L1 certificate adapter.

Mirrors ``tests/unit/convergence/test_gascity_adapter.py`` — three event
mirrors (``NodeEvent`` / ``EdgeEvent`` / ``EvolveEvent``), three
``evaluate_*`` methods, and JSON envelope round-trip — without
requiring ``agentflow`` to be installed. The adapter consumes
dataclass shapes that mirror agentflow runtime boundaries; the actual
``berabuddies/agentflow`` package is optional.

The dogfood test at the bottom imports the public theorem-name
constants from ``operon-langgraph-gates`` v0.1.0 to confirm that the
adapter can consume them as theorem identifiers; it is skipped when
that package is not installed.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from operon_ai.convergence import verification_to_dolt_envelope
from operon_ai.convergence.agentflow_adapter import (
    AgentflowCertificateAdapter,
    EdgeEvent,
    EvolveEvent,
    NodeEvent,
)
from operon_ai.core.certificate import register_verify_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _passing_quality_node(event: NodeEvent) -> Mapping[str, Any]:
    del event
    return {"scores": [0.9, 0.95, 0.88], "threshold": 0.7}


def _failing_quality_node(event: NodeEvent) -> Mapping[str, Any]:
    del event
    return {"scores": [0.2, 0.3, 0.1], "threshold": 0.7}


def _passing_quality_edge(event: EdgeEvent) -> Mapping[str, Any]:
    del event
    return {"scores": [0.99], "threshold": 0.5}


def _passing_quality_evolve(event: EvolveEvent) -> Mapping[str, Any]:
    """Forward the three pinned hashes verbatim from the EvolveEvent."""
    return {
        "graph_hash": event.graph_hash,
        "traces_hash": event.traces_hash,
        "tuned_agent_hash": event.tuned_agent_hash,
    }


def _make_node_event(phase: str = "post") -> NodeEvent:
    return NodeEvent(
        session_id="sess-1",
        node_name="planner",
        phase=phase,  # type: ignore[arg-type]
        payload={"output": "ok"},
        timestamp="2026-05-11T08:00:00Z",
    )


def _make_edge_event() -> EdgeEvent:
    return EdgeEvent(
        session_id="sess-1",
        from_node="planner",
        to_node="executor",
        payload={"reason": "sequence"},
        timestamp="2026-05-11T08:00:01Z",
    )


def _make_evolve_event() -> EvolveEvent:
    return EvolveEvent(
        session_id="sess-1",
        graph_hash="a" * 16,
        traces_hash="b" * 16,
        tuned_agent_hash="c" * 16,
        timestamp="2026-05-11T08:00:02Z",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    def test_unknown_theorem_raises(self) -> None:
        with pytest.raises(KeyError, match="is not registered"):
            AgentflowCertificateAdapter(theorem="no_such_theorem_exists")

    def test_registered_theorem_from_static_registry_is_accepted(self) -> None:
        adapter = AgentflowCertificateAdapter(theorem="behavioral_quality")
        assert adapter.theorem == "behavioral_quality"

    def test_dynamic_registry_theorems_are_accepted(self) -> None:
        def _always_true(_params):  # noqa: ARG001
            del _params
            return True, {"reason": "dynamic-test"}

        register_verify_fn("dynamic_test_theorem_agentflow", _always_true)
        adapter = AgentflowCertificateAdapter(
            theorem="dynamic_test_theorem_agentflow"
        )
        assert adapter.theorem == "dynamic_test_theorem_agentflow"

    def test_l2_evolve_theorem_is_accepted(self) -> None:
        """The L2 hook's theorem name should be a first-class theorem here too."""
        adapter = AgentflowCertificateAdapter(
            theorem="agentflow_evolve_pinned_inputs"
        )
        assert adapter.theorem == "agentflow_evolve_pinned_inputs"

    def test_default_source_label(self) -> None:
        adapter = AgentflowCertificateAdapter(theorem="behavioral_quality")
        assert adapter.source == "agentflow_adapter"


# ---------------------------------------------------------------------------
# evaluate_node / evaluate_edge / evaluate_evolve
# ---------------------------------------------------------------------------


class TestEvaluateNode:
    def test_passing_harness_yields_holds_true(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_node=_passing_quality_node,
        )
        v = adapter.evaluate_node(_make_node_event())
        assert v.holds is True
        assert v.evidence["mean"] >= 0.7

    def test_failing_harness_yields_holds_false(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_node=_failing_quality_node,
        )
        v = adapter.evaluate_node(_make_node_event())
        assert v.holds is False

    def test_missing_harness_raises(self) -> None:
        adapter = AgentflowCertificateAdapter(theorem="behavioral_quality")
        with pytest.raises(RuntimeError, match="harness_node"):
            adapter.evaluate_node(_make_node_event())

    def test_attach_point_in_conclusion(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_node=_passing_quality_node,
        )
        v = adapter.evaluate_node(_make_node_event())
        assert "node" in v.certificate.conclusion

    def test_pre_and_post_phase_both_supported(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_node=_passing_quality_node,
        )
        v_pre = adapter.evaluate_node(_make_node_event(phase="pre"))
        v_post = adapter.evaluate_node(_make_node_event(phase="post"))
        assert v_pre.holds is True
        assert v_post.holds is True


class TestEvaluateEdge:
    def test_passing_harness_yields_holds_true(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_edge=_passing_quality_edge,
        )
        v = adapter.evaluate_edge(_make_edge_event())
        assert v.holds is True

    def test_missing_harness_raises(self) -> None:
        adapter = AgentflowCertificateAdapter(theorem="behavioral_quality")
        with pytest.raises(RuntimeError, match="harness_edge"):
            adapter.evaluate_edge(_make_edge_event())

    def test_attach_point_in_conclusion(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_edge=_passing_quality_edge,
        )
        v = adapter.evaluate_edge(_make_edge_event())
        assert "edge" in v.certificate.conclusion


class TestEvaluateEvolve:
    def test_passing_harness_yields_holds_true(self) -> None:
        """Forwarding well-formed hashes to the L2 theorem should hold."""
        adapter = AgentflowCertificateAdapter(
            theorem="agentflow_evolve_pinned_inputs",
            harness_evolve=_passing_quality_evolve,
        )
        v = adapter.evaluate_evolve(_make_evolve_event())
        assert v.holds is True

    def test_malformed_hashes_yield_holds_false(self) -> None:
        """Uppercase hex is malformed per the L2 verifier; should not hold."""

        def _bad_harness(event: EvolveEvent) -> Mapping[str, Any]:
            del event
            return {
                "graph_hash": "ABCDEF12",  # uppercase — malformed
                "traces_hash": "b" * 16,
                "tuned_agent_hash": "c" * 16,
            }

        adapter = AgentflowCertificateAdapter(
            theorem="agentflow_evolve_pinned_inputs",
            harness_evolve=_bad_harness,
        )
        v = adapter.evaluate_evolve(_make_evolve_event())
        assert v.holds is False
        assert "graph_hash" in list(v.evidence["malformed"])

    def test_missing_harness_raises(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="agentflow_evolve_pinned_inputs"
        )
        with pytest.raises(RuntimeError, match="harness_evolve"):
            adapter.evaluate_evolve(_make_evolve_event())

    def test_attach_point_in_conclusion(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="agentflow_evolve_pinned_inputs",
            harness_evolve=_passing_quality_evolve,
        )
        v = adapter.evaluate_evolve(_make_evolve_event())
        assert "evolve" in v.certificate.conclusion


# ---------------------------------------------------------------------------
# Dolt audit-trail envelope — reuses gascity's serializer
# ---------------------------------------------------------------------------


class TestDoltEnvelope:
    def test_envelope_keys_are_jsonable(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_node=_passing_quality_node,
        )
        v = adapter.evaluate_node(_make_node_event())
        env = verification_to_dolt_envelope(v, attach_point="node")
        assert set(env.keys()) >= {
            "theorem",
            "holds",
            "conclusion",
            "source",
            "attach_point",
            "parameters",
            "evidence",
        }
        assert env["attach_point"] == "node"
        assert env["source"] == "agentflow_adapter"

    def test_envelope_round_trips_through_json(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_node=_failing_quality_node,
        )
        v = adapter.evaluate_node(_make_node_event())
        env = verification_to_dolt_envelope(v, attach_point="node")
        roundtripped = json.loads(json.dumps(env))
        assert roundtripped["holds"] is False
        assert roundtripped["theorem"] == "behavioral_quality"

    def test_envelope_attach_point_label_matches_caller(self) -> None:
        adapter = AgentflowCertificateAdapter(
            theorem="behavioral_quality",
            harness_edge=_passing_quality_edge,
        )
        v = adapter.evaluate_edge(_make_edge_event())
        env = verification_to_dolt_envelope(v, attach_point="edge")
        assert env["attach_point"] == "edge"


# ---------------------------------------------------------------------------
# Dogfood: consume operon-langgraph-gates v0.1.0 stable theorem-name surface
# ---------------------------------------------------------------------------


class TestOperonLanggraphGatesDogfood:
    """Validates that v0.1.0's STAGNATION_THEOREM / INTEGRITY_THEOREM constants
    are consumable by the agentflow adapter — same load-bearing test that the
    gascity adapter ships, applied to the agentflow attach points.

    Skipped when ``operon-langgraph-gates`` is not installed.
    """

    def test_stagnation_theorem_constant_is_resolvable(self) -> None:
        olg = pytest.importorskip("operon_langgraph_gates")

        def _stagnation_harness(_evt: NodeEvent) -> Mapping[str, Any]:  # noqa: ARG001
            del _evt
            return {"signal_values": [0.1, 0.2, 0.15], "threshold": 0.5}

        adapter = AgentflowCertificateAdapter(
            theorem=olg.STAGNATION_THEOREM,
            harness_node=_stagnation_harness,
        )
        v = adapter.evaluate_node(_make_node_event())
        assert v.holds is True
        assert v.certificate.theorem == "behavioral_stability_windowed"

    def test_integrity_theorem_constant_is_resolvable(self) -> None:
        olg = pytest.importorskip("operon_langgraph_gates")
        adapter = AgentflowCertificateAdapter(theorem=olg.INTEGRITY_THEOREM)
        assert adapter.theorem == olg.INTEGRITY_THEOREM
