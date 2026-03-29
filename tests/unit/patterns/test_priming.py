"""Tests for PrimingView and build_priming_view."""

from datetime import datetime, timezone

import pytest

from operon_ai.memory.bitemporal import BiTemporalFact
from operon_ai.patterns.priming import PrimingView, build_priming_view
from operon_ai.patterns.types import SubstrateView


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

_FACT = BiTemporalFact(
    fact_id="f-1",
    subject="sky",
    predicate="has_color",
    value="blue",
    valid_from=_NOW,
    valid_to=None,
    recorded_from=_NOW,
    recorded_to=None,
    source="test",
)


def _make_substrate(**overrides) -> SubstrateView:
    defaults = dict(facts=(_FACT,), query="what color?", record_time=_NOW)
    defaults.update(overrides)
    return SubstrateView(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_priming_view_is_substrate_view():
    view = PrimingView(facts=(_FACT,), query=None, record_time=_NOW)
    assert isinstance(view, SubstrateView)


def test_priming_view_inherits_fields():
    view = PrimingView(facts=(_FACT,), query="q", record_time=_NOW)
    assert view.facts == (_FACT,)
    assert view.query == "q"
    assert view.record_time == _NOW


def test_priming_view_defaults():
    view = PrimingView(facts=(), query=None, record_time=_NOW)
    assert view.recent_outputs == ()
    assert view.telemetry == ()
    assert view.experience == ()
    assert view.developmental_status is None
    assert view.trust_context == {}


def test_priming_view_with_all_channels():
    outputs = ({"stage": "plan", "result": "ok"},)
    telem = ({"latency_ms": 42},)
    exp = ({"pattern": "retry"},)
    dev_status = {"stage": "juvenile"}
    trust = {"user": 0.9, "system": 0.7}

    view = PrimingView(
        facts=(_FACT,),
        query="q",
        record_time=_NOW,
        recent_outputs=outputs,
        telemetry=telem,
        experience=exp,
        developmental_status=dev_status,
        trust_context=trust,
    )
    assert view.recent_outputs == outputs
    assert view.telemetry == telem
    assert view.experience == exp
    assert view.developmental_status == dev_status
    assert view.trust_context == trust


def test_build_priming_view_from_substrate():
    substrate = _make_substrate()
    outputs = ({"stage": "act", "result": "done"},)
    trust = {"peer": 0.8}

    priming = build_priming_view(
        substrate,
        recent_outputs=outputs,
        trust_context=trust,
    )

    assert isinstance(priming, PrimingView)
    assert isinstance(priming, SubstrateView)
    assert priming.facts == substrate.facts
    assert priming.query == substrate.query
    assert priming.record_time == substrate.record_time
    assert priming.recent_outputs == outputs
    assert priming.trust_context == trust
    # Unset channels default to empty
    assert priming.telemetry == ()
    assert priming.experience == ()
    assert priming.developmental_status is None


def test_priming_view_frozen():
    view = PrimingView(facts=(), query=None, record_time=_NOW)
    with pytest.raises(AttributeError):
        view.facts = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        view.recent_outputs = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        view.trust_context = {}  # type: ignore[misc]


def test_substrate_view_still_works():
    """Plain SubstrateView construction is unaffected by PrimingView's existence."""
    sv = _make_substrate()
    assert sv.facts == (_FACT,)
    assert sv.query == "what color?"
    assert sv.record_time == _NOW
    assert not hasattr(sv, "recent_outputs")


def test_build_priming_view_defaults_trust_to_empty_dict():
    """When trust_context is not provided, it defaults to an empty dict."""
    substrate = _make_substrate()
    priming = build_priming_view(substrate)
    assert priming.trust_context == {}
