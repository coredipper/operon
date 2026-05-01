"""Tests for the Gas City certificate adapter.

These tests verify the adapter's contract — three event mirrors
(``HookEvent`` / ``DispatchEvent`` / ``MailEvent``), three
``evaluate_*`` methods, and a JSON envelope serializer — without
requiring ``gascity`` to be installed (it is a Go runtime; no Python
package exists).

The dogfood test at the bottom imports the public theorem-name
constants from ``operon-langgraph-gates`` v0.1.0 to confirm that the
adapter can consume them as theorem identifiers; it is skipped when
that package is not installed.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from operon_ai.convergence.gascity_adapter import (
    DispatchEvent,
    GascityCertificateAdapter,
    HookEvent,
    MailEvent,
    verification_to_dolt_envelope,
)
from operon_ai.core.certificate import register_verify_fn


# ---------------------------------------------------------------------------
# Fixtures — passing and failing harness variants per attach point.
# ---------------------------------------------------------------------------


def _passing_quality_hook(event: HookEvent) -> Mapping[str, Any]:
    """Hook harness that emits scores well above the quality threshold."""
    del event
    return {"scores": [0.9, 0.95, 0.88], "threshold": 0.7}


def _failing_quality_hook(event: HookEvent) -> Mapping[str, Any]:
    """Hook harness that emits scores below the quality threshold."""
    del event
    return {"scores": [0.2, 0.3, 0.1], "threshold": 0.7}


def _passing_quality_dispatch(event: DispatchEvent) -> Mapping[str, Any]:
    del event
    return {"scores": [0.99], "threshold": 0.5}


def _passing_quality_mail(event: MailEvent) -> Mapping[str, Any]:
    del event
    return {"scores": [0.99], "threshold": 0.5}


def _make_hook_event() -> HookEvent:
    return HookEvent(
        session_id="sess-1",
        hook_kind="PreToolUse",
        payload={"tool": "Read", "args": {"path": "/tmp/x"}},
        timestamp="2026-05-01T08:00:00Z",
    )


def _make_dispatch_event() -> DispatchEvent:
    return DispatchEvent(
        session_id="sess-1",
        nudge_kind="formula",
        payload={"target": "agent-2"},
        timestamp="2026-05-01T08:00:01Z",
    )


def _make_mail_event() -> MailEvent:
    return MailEvent(
        sender="agent-1",
        recipient="agent-2",
        subject="status",
        body={"value": 42},
        timestamp="2026-05-01T08:00:02Z",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    def test_unknown_theorem_raises(self) -> None:
        with pytest.raises(KeyError, match="is not registered"):
            GascityCertificateAdapter(theorem="no_such_theorem_exists")

    def test_registered_theorem_from_static_registry_is_accepted(self) -> None:
        adapter = GascityCertificateAdapter(theorem="behavioral_quality")
        assert adapter.theorem == "behavioral_quality"

    def test_dynamic_registry_theorems_are_accepted(self) -> None:
        """Theorems registered at runtime should be usable by the adapter."""

        def _always_true(_params):  # noqa: ARG001 — verify-fn protocol
            del _params
            return True, {"reason": "dynamic-test"}

        register_verify_fn("dynamic_test_theorem_gascity", _always_true)
        adapter = GascityCertificateAdapter(theorem="dynamic_test_theorem_gascity")
        assert adapter.theorem == "dynamic_test_theorem_gascity"

    def test_default_source_label(self) -> None:
        adapter = GascityCertificateAdapter(theorem="behavioral_quality")
        assert adapter.source == "gascity_adapter"


# ---------------------------------------------------------------------------
# evaluate_hook / evaluate_dispatch / evaluate_mail
# ---------------------------------------------------------------------------


class TestEvaluateHook:
    def test_passing_harness_yields_holds_true(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_hook=_passing_quality_hook,
        )
        v = adapter.evaluate_hook(_make_hook_event())
        assert v.holds is True
        assert v.evidence["mean"] >= 0.7

    def test_failing_harness_yields_holds_false(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_hook=_failing_quality_hook,
        )
        v = adapter.evaluate_hook(_make_hook_event())
        assert v.holds is False

    def test_missing_harness_raises(self) -> None:
        adapter = GascityCertificateAdapter(theorem="behavioral_quality")
        with pytest.raises(RuntimeError, match="harness_hook"):
            adapter.evaluate_hook(_make_hook_event())

    def test_attach_point_in_conclusion(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_hook=_passing_quality_hook,
        )
        v = adapter.evaluate_hook(_make_hook_event())
        assert "hook" in v.certificate.conclusion


class TestEvaluateDispatch:
    def test_passing_harness_yields_holds_true(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_dispatch=_passing_quality_dispatch,
        )
        v = adapter.evaluate_dispatch(_make_dispatch_event())
        assert v.holds is True

    def test_missing_harness_raises(self) -> None:
        adapter = GascityCertificateAdapter(theorem="behavioral_quality")
        with pytest.raises(RuntimeError, match="harness_dispatch"):
            adapter.evaluate_dispatch(_make_dispatch_event())

    def test_attach_point_in_conclusion(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_dispatch=_passing_quality_dispatch,
        )
        v = adapter.evaluate_dispatch(_make_dispatch_event())
        assert "dispatch" in v.certificate.conclusion


class TestEvaluateMail:
    def test_passing_harness_yields_holds_true(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_mail=_passing_quality_mail,
        )
        v = adapter.evaluate_mail(_make_mail_event())
        assert v.holds is True

    def test_missing_harness_raises(self) -> None:
        adapter = GascityCertificateAdapter(theorem="behavioral_quality")
        with pytest.raises(RuntimeError, match="harness_mail"):
            adapter.evaluate_mail(_make_mail_event())

    def test_attach_point_in_conclusion(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_mail=_passing_quality_mail,
        )
        v = adapter.evaluate_mail(_make_mail_event())
        assert "mail" in v.certificate.conclusion


# ---------------------------------------------------------------------------
# Dolt audit-trail envelope
# ---------------------------------------------------------------------------


class TestDoltEnvelope:
    def test_envelope_keys_are_jsonable(self) -> None:
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_hook=_passing_quality_hook,
        )
        v = adapter.evaluate_hook(_make_hook_event())
        env = verification_to_dolt_envelope(v, attach_point="hook")
        assert set(env.keys()) >= {
            "theorem",
            "holds",
            "conclusion",
            "source",
            "attach_point",
            "parameters",
            "evidence",
        }
        assert isinstance(env["holds"], bool)
        assert env["theorem"] == "behavioral_quality"
        assert env["attach_point"] == "hook"

    def test_envelope_round_trips_through_json(self) -> None:
        """Dolt insertion needs a clean json.dumps round-trip."""
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_hook=_failing_quality_hook,
        )
        v = adapter.evaluate_hook(_make_hook_event())
        env = verification_to_dolt_envelope(v, attach_point="hook")
        roundtripped = json.loads(json.dumps(env))
        assert roundtripped["holds"] is False
        assert roundtripped["theorem"] == "behavioral_quality"
        assert "mean" in roundtripped["evidence"]

    def test_envelope_attach_point_label_matches_caller(self) -> None:
        """The serializer takes attach_point as a kwarg — ensure it lands verbatim."""
        adapter = GascityCertificateAdapter(
            theorem="behavioral_quality",
            harness_dispatch=_passing_quality_dispatch,
        )
        v = adapter.evaluate_dispatch(_make_dispatch_event())
        env = verification_to_dolt_envelope(v, attach_point="dispatch")
        assert env["attach_point"] == "dispatch"


# ---------------------------------------------------------------------------
# Dogfood: consume operon-langgraph-gates v0.1.0 stable theorem-name surface
# ---------------------------------------------------------------------------


class TestOperonLanggraphGatesDogfood:
    """Validates that v0.1.0's STAGNATION_THEOREM / INTEGRITY_THEOREM constants
    are consumable by the gascity adapter — i.e. the stable surface lands as
    designed.

    Skipped when ``operon-langgraph-gates`` is not installed.
    """

    def test_stagnation_theorem_constant_is_resolvable(self) -> None:
        olg = pytest.importorskip("operon_langgraph_gates")

        def _stagnation_harness(_evt: HookEvent) -> Mapping[str, Any]:  # noqa: ARG001
            del _evt
            return {"signal_values": [0.1, 0.2, 0.15], "threshold": 0.5}

        adapter = GascityCertificateAdapter(
            theorem=olg.STAGNATION_THEOREM,
            harness_hook=_stagnation_harness,
        )
        v = adapter.evaluate_hook(_make_hook_event())
        # behavioral_stability_windowed: max(values)=0.2 ≤ threshold=0.5 ⇒ holds
        assert v.holds is True
        assert v.certificate.theorem == "behavioral_stability_windowed"

    def test_integrity_theorem_constant_is_resolvable(self) -> None:
        olg = pytest.importorskip("operon_langgraph_gates")
        # Importing operon_langgraph_gates.integrity registers the
        # ``langgraph_state_integrity`` theorem in operon-ai's registry.
        # If the import side-effect is gated, importing the package
        # itself should be enough — confirm by constructing the adapter.
        adapter = GascityCertificateAdapter(theorem=olg.INTEGRITY_THEOREM)
        assert adapter.theorem == olg.INTEGRITY_THEOREM
