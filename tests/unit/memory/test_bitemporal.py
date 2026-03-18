"""Tests for the bi-temporal memory subsystem."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from operon_ai.memory.bitemporal import (
    BiTemporalFact,
    BiTemporalMemory,
    BiTemporalQuery,
    CorrectionResult,
    FactSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory() -> BiTemporalMemory:
    return BiTemporalMemory()


def _t(days: int) -> datetime:
    """Deterministic timestamp: 2025-01-01 + *days* days."""
    return datetime(2025, 1, 1) + timedelta(days=days)


# ---------------------------------------------------------------------------
# Task 1: Core data model
# ---------------------------------------------------------------------------

class TestBiTemporalFact:
    def test_create_fact(self):
        f = BiTemporalFact(
            fact_id="abc",
            subject="client:1",
            predicate="status",
            value="active",
            valid_from=_t(0),
            valid_to=None,
            recorded_from=_t(1),
            recorded_to=None,
            source="crm",
        )
        assert f.subject == "client:1"
        assert f.value == "active"

    def test_frozen(self):
        f = BiTemporalFact(
            fact_id="abc",
            subject="s",
            predicate="p",
            value="v",
            valid_from=_t(0),
            valid_to=None,
            recorded_from=_t(0),
            recorded_to=None,
            source="test",
        )
        with pytest.raises(AttributeError):
            f.value = "new"  # type: ignore[misc]

    def test_defaults(self):
        f = BiTemporalFact(
            fact_id="x",
            subject="s",
            predicate="p",
            value=1,
            valid_from=_t(0),
            valid_to=None,
            recorded_from=_t(0),
            recorded_to=None,
            source="test",
        )
        assert f.confidence == 1.0
        assert f.tags == ()
        assert f.supersedes is None

    def test_custom_tags(self):
        f = BiTemporalFact(
            fact_id="x",
            subject="s",
            predicate="p",
            value=1,
            valid_from=_t(0),
            valid_to=None,
            recorded_from=_t(0),
            recorded_to=None,
            source="test",
            tags=("audit", "compliance"),
        )
        assert f.tags == ("audit", "compliance")

    def test_recorded_to_none_for_active(self):
        f = BiTemporalFact(
            fact_id="x",
            subject="s",
            predicate="p",
            value=1,
            valid_from=_t(0),
            valid_to=None,
            recorded_from=_t(0),
            recorded_to=None,
            source="test",
        )
        assert f.recorded_to is None


class TestBiTemporalQuery:
    def test_create_query(self):
        q = BiTemporalQuery(subject="client:1", at_valid=_t(5))
        assert q.subject == "client:1"
        assert q.at_valid == _t(5)

    def test_frozen(self):
        q = BiTemporalQuery()
        with pytest.raises(AttributeError):
            q.subject = "x"  # type: ignore[misc]


class TestFactSnapshot:
    def test_create_snapshot(self):
        q = BiTemporalQuery(subject="s")
        snap = FactSnapshot(facts=(), query=q, snapshot_time=_t(0))
        assert snap.facts == ()
        assert snap.query is q


# ---------------------------------------------------------------------------
# Task 2: Write semantics
# ---------------------------------------------------------------------------

class TestRecordFact:
    def test_returns_fact(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "v", _t(0), _t(1), "src")
        assert isinstance(f, BiTemporalFact)
        assert f.value == "v"

    def test_active_record(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "v", _t(0), _t(1), "src")
        assert f.recorded_to is None

    def test_unique_ids(self):
        mem = _make_memory()
        f1 = mem.record_fact("s", "p", "v1", _t(0), _t(1), "src")
        f2 = mem.record_fact("s", "p", "v2", _t(0), _t(2), "src")
        assert f1.fact_id != f2.fact_id

    def test_subject_index(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "v", _t(0), _t(1), "src")
        assert f.fact_id in mem._by_subject["s"]


class TestCorrectFact:
    def test_creates_new_fact(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        result = mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        assert result.new_fact.value == "new"
        assert isinstance(result, CorrectionResult)

    def test_closes_old_record(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        result = mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        assert result.old_fact.recorded_to == _t(3)

    def test_supersedes_link(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        result = mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        assert result.new_fact.supersedes == f.fact_id

    def test_result_type(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        result = mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        assert isinstance(result, CorrectionResult)
        assert result.correction_time == _t(3)

    def test_history_preserved(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        # Both the closed old fact and the new fact should exist
        assert len(mem._facts) == 2

    def test_key_error_on_missing(self):
        mem = _make_memory()
        with pytest.raises(KeyError):
            mem.correct_fact("nonexistent", "v", _t(0), _t(1), "src")


class TestInvalidateFact:
    def test_closes_record(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "v", _t(0), _t(1), "src")
        closed = mem.invalidate_fact(f.fact_id, recorded_to=_t(5))
        assert closed.recorded_to == _t(5)

    def test_key_error_on_missing(self):
        mem = _make_memory()
        with pytest.raises(KeyError):
            mem.invalidate_fact("nonexistent")

    def test_default_recorded_to(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "v", _t(0), _t(1), "src")
        closed = mem.invalidate_fact(f.fact_id)
        assert closed.recorded_to is not None


# ---------------------------------------------------------------------------
# Task 3: Point-in-time retrieval
# ---------------------------------------------------------------------------

class TestRetrieveValidAt:
    def test_valid_facts(self):
        mem = _make_memory()
        mem.record_fact("s", "p", "v", _t(0), _t(0), "src")
        results = mem.retrieve_valid_at(at=_t(1), subject="s")
        assert len(results) == 1
        assert results[0].value == "v"

    def test_expired_excluded(self):
        mem = _make_memory()
        mem.record_fact("s", "p", "v", _t(0), _t(0), "src", valid_to=_t(1))
        results = mem.retrieve_valid_at(at=_t(2), subject="s")
        assert len(results) == 0

    def test_future_excluded(self):
        mem = _make_memory()
        mem.record_fact("s", "p", "v", _t(5), _t(0), "src")
        results = mem.retrieve_valid_at(at=_t(2), subject="s")
        assert len(results) == 0

    def test_invalidated_excluded(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "v", _t(0), _t(0), "src")
        mem.invalidate_fact(f.fact_id, recorded_to=_t(3))
        results = mem.retrieve_valid_at(at=_t(1), subject="s")
        assert len(results) == 0


class TestRetrieveKnownAt:
    def test_known_facts(self):
        mem = _make_memory()
        mem.record_fact("s", "p", "v", _t(0), _t(1), "src")
        results = mem.retrieve_known_at(at=_t(2), subject="s")
        assert len(results) == 1

    def test_future_records_excluded(self):
        mem = _make_memory()
        mem.record_fact("s", "p", "v", _t(0), _t(5), "src")
        results = mem.retrieve_known_at(at=_t(2), subject="s")
        assert len(results) == 0

    def test_closed_records_included(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        # At t(2), only the old (now-closed) record was known
        results = mem.retrieve_known_at(at=_t(2), subject="s")
        assert len(results) == 1
        assert results[0].value == "old"


class TestRetrieveBeliefState:
    def test_both_axes(self):
        mem = _make_memory()
        mem.record_fact("s", "p", "v", _t(0), _t(1), "src")
        results = mem.retrieve_belief_state(at_valid=_t(0), at_record=_t(2))
        assert len(results) == 1

    def test_valid_but_unknown(self):
        mem = _make_memory()
        # Fact valid from day 0 but not recorded until day 5
        mem.record_fact("s", "p", "v", _t(0), _t(5), "src")
        # At record-time day 2, system didn't know yet
        results = mem.retrieve_belief_state(at_valid=_t(0), at_record=_t(2))
        assert len(results) == 0

    def test_known_but_invalid(self):
        mem = _make_memory()
        # Fact valid from day 5 but recorded from day 1
        mem.record_fact("s", "p", "v", _t(5), _t(1), "src")
        # At valid-time day 2, fact is not yet valid
        results = mem.retrieve_belief_state(at_valid=_t(2), at_record=_t(2))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Task 4: History and diff
# ---------------------------------------------------------------------------

class TestHistory:
    def test_all_facts_returned(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        hist = mem.history("s")
        assert len(hist) == 2

    def test_sorted_by_recorded_from(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        hist = mem.history("s")
        assert hist[0].recorded_from <= hist[1].recorded_from


class TestDiffBetween:
    def test_valid_time_diff(self):
        mem = _make_memory()
        # Fact becomes valid at day 3
        mem.record_fact("s", "p", "v", _t(3), _t(0), "src")
        diff = mem.diff_between(_t(1), _t(5), axis="valid")
        assert len(diff) == 1
        assert diff[0].value == "v"

    def test_record_time_diff(self):
        mem = _make_memory()
        # Fact recorded at day 3
        mem.record_fact("s", "p", "v", _t(0), _t(3), "src")
        diff = mem.diff_between(_t(1), _t(5), axis="record")
        assert len(diff) == 1

    def test_invalid_axis_raises(self):
        mem = _make_memory()
        with pytest.raises(ValueError, match="axis must be"):
            mem.diff_between(_t(0), _t(1), axis="bogus")


class TestTimelineFor:
    def test_sorted_by_valid_from(self):
        mem = _make_memory()
        mem.record_fact("s", "p", "late", _t(5), _t(0), "src")
        mem.record_fact("s", "p", "early", _t(1), _t(0), "src")
        tl = mem.timeline_for("s")
        assert tl[0].valid_from < tl[1].valid_from

    def test_includes_closed(self):
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        tl = mem.timeline_for("s")
        assert len(tl) == 2


# ---------------------------------------------------------------------------
# Task 7: Release checks
# ---------------------------------------------------------------------------

class TestReleaseChecks:
    def test_append_only(self):
        """Corrections must not remove old facts from the store."""
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        mem.correct_fact(f.fact_id, "new", _t(0), _t(3), "review")
        # Old fact still exists (closed, but present)
        old_in_store = mem._by_id[f.fact_id]
        assert old_in_store.recorded_to is not None
        assert old_in_store.value == "old"

    def test_past_reconstructible(self):
        """Historical belief states must remain reconstructible."""
        mem = _make_memory()
        f = mem.record_fact("s", "p", "old", _t(0), _t(1), "src")
        mem.correct_fact(f.fact_id, "new", _t(0), _t(5), "review")
        # Before correction: system believed "old"
        belief_before = mem.retrieve_belief_state(at_valid=_t(0), at_record=_t(3))
        assert len(belief_before) == 1
        assert belief_before[0].value == "old"
        # After correction: system believes "new"
        belief_after = mem.retrieve_belief_state(at_valid=_t(0), at_record=_t(6))
        assert len(belief_after) == 1
        assert belief_after[0].value == "new"

    def test_axes_disagree(self):
        """Valid-time and record-time queries can produce different results."""
        mem = _make_memory()
        # Fact true in world from day 1, but system learns it on day 5
        mem.record_fact("s", "p", "v", _t(1), _t(5), "src")
        # Valid at day 2 — yes (active record, world-time match)
        valid = mem.retrieve_valid_at(at=_t(2), subject="s")
        assert len(valid) == 1
        # Known at day 2 — no (not recorded yet)
        known = mem.retrieve_known_at(at=_t(2), subject="s")
        assert len(known) == 0
