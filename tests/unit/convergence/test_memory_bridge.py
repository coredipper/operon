"""Tests for the AnimaWorks / DeerFlow -> BiTemporalMemory bridge."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from operon_ai.convergence.memory_bridge import (
    _parse_timestamp,
    bridge_animaworks_memory,
    bridge_deerflow_memory,
)
from operon_ai.memory.bitemporal import BiTemporalFact, BiTemporalMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def btm() -> BiTemporalMemory:
    return BiTemporalMemory()


@pytest.fixture()
def animaworks_entries() -> list[dict]:
    return [
        {
            "id": "mem_001",
            "type": "episodic",
            "content": "User prefers Python",
            "timestamp": "2026-03-01T10:00:00",
            "source_agent": "assistant",
        },
        {
            "id": "mem_002",
            "type": "semantic",
            "content": "Project uses FastAPI",
            "timestamp": "2026-03-02T12:00:00",
            "source_agent": "coder",
        },
        {
            "id": "mem_003",
            "type": "procedural",
            "content": "Always run linter before commit",
            "timestamp": "2026-03-03T08:30:00",
            "source_agent": "reviewer",
        },
    ]


@pytest.fixture()
def deerflow_session() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Search for AI papers",
            "timestamp": "2026-03-01T10:00:00",
        },
        {
            "role": "assistant",
            "content": "Found 42 papers on scaling laws",
            "timestamp": "2026-03-01T10:01:00",
        },
    ]


@pytest.fixture()
def deerflow_vectors() -> list[dict]:
    return [
        {
            "id": "vec_001",
            "content": "AI scaling laws summary",
            "metadata": {"source": "arxiv"},
            "inserted_at": "2026-03-01T10:00:00",
        },
        {
            "id": "vec_002",
            "content": "Transformer architecture overview",
            "metadata": {"source": "arxiv"},
            "inserted_at": "2026-03-01T11:00:00",
        },
    ]


# ---------------------------------------------------------------------------
# AnimaWorks bridge
# ---------------------------------------------------------------------------

class TestBridgeAnimaworksMemory:
    def test_bridge_animaworks_creates_facts(
        self, btm: BiTemporalMemory, animaworks_entries: list[dict],
    ) -> None:
        facts = bridge_animaworks_memory(animaworks_entries, btm)
        assert len(facts) == 3
        assert all(isinstance(f, BiTemporalFact) for f in facts)

    def test_bridge_animaworks_subjects_have_prefix(
        self, btm: BiTemporalMemory, animaworks_entries: list[dict],
    ) -> None:
        facts = bridge_animaworks_memory(animaworks_entries, btm, subject_prefix="aw")
        assert all(f.subject.startswith("aw:") for f in facts)

    def test_bridge_animaworks_handles_missing_fields(
        self, btm: BiTemporalMemory,
    ) -> None:
        entries = [
            {
                "id": "mem_X",
                "type": "episodic",
                "content": "Something happened",
                # no source_agent, no timestamp
            },
        ]
        facts = bridge_animaworks_memory(entries, btm)
        assert len(facts) == 1
        assert facts[0].source == "animaworks:unknown"
        # timestamp should fall back to now (UTC-aware)
        assert facts[0].valid_from.tzinfo is not None

    def test_bridge_animaworks_fact_fields(
        self, btm: BiTemporalMemory,
    ) -> None:
        entries = [
            {
                "id": "mem_100",
                "type": "semantic",
                "content": "Fact content",
                "timestamp": "2026-06-15T09:00:00",
                "source_agent": "planner",
            },
        ]
        facts = bridge_animaworks_memory(entries, btm)
        f = facts[0]
        assert f.subject == "animaworks:mem_100"
        assert f.predicate == "semantic"
        assert f.value == "Fact content"
        assert f.source == "animaworks:planner"
        assert f.valid_from == datetime(2026, 6, 15, 9, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# DeerFlow bridge
# ---------------------------------------------------------------------------

class TestBridgeDeerflowMemory:
    def test_bridge_deerflow_session_and_vector(
        self,
        btm: BiTemporalMemory,
        deerflow_session: list[dict],
        deerflow_vectors: list[dict],
    ) -> None:
        facts = bridge_deerflow_memory(deerflow_session, deerflow_vectors, btm, session_id="test")
        assert len(facts) == 4  # 2 session + 2 vector

    def test_bridge_deerflow_session_only(
        self, btm: BiTemporalMemory, deerflow_session: list[dict],
    ) -> None:
        facts = bridge_deerflow_memory(deerflow_session, [], btm, session_id="test")
        assert len(facts) == 2
        assert all("session" in f.subject for f in facts)

    def test_bridge_deerflow_vector_only(
        self, btm: BiTemporalMemory, deerflow_vectors: list[dict],
    ) -> None:
        # Vector-only import works without session_id.
        facts = bridge_deerflow_memory([], deerflow_vectors, btm)
        assert len(facts) == 2
        assert all(f.predicate == "vector_entry" for f in facts)

    def test_bridge_deerflow_session_subjects_indexed(
        self, btm: BiTemporalMemory, deerflow_session: list[dict],
    ) -> None:
        facts = bridge_deerflow_memory(deerflow_session, [], btm, session_id="test")
        subjects = [f.subject for f in facts]
        # Subjects use session_id + content hash for uniqueness.
        assert all(s.startswith("deerflow:session:") for s in subjects)
        assert len(subjects) == len(set(subjects))  # no duplicates within session
        # With explicit session_id, re-import is stable.
        btm2 = BiTemporalMemory()
        facts2 = bridge_deerflow_memory(deerflow_session, [], btm2, session_id="sess_A")
        btm3 = BiTemporalMemory()
        facts3 = bridge_deerflow_memory(deerflow_session, [], btm3, session_id="sess_A")
        assert [f.subject for f in facts2] == [f.subject for f in facts3]
        # Different session_ids produce different subjects.
        btm4 = BiTemporalMemory()
        facts4 = bridge_deerflow_memory(deerflow_session, [], btm4, session_id="sess_B")
        assert [f.subject for f in facts2] != [f.subject for f in facts4]

    def test_bridge_deerflow_vector_subjects(
        self, btm: BiTemporalMemory, deerflow_vectors: list[dict],
    ) -> None:
        facts = bridge_deerflow_memory([], deerflow_vectors, btm)
        subjects = {f.subject for f in facts}
        assert subjects == {"deerflow:vector:vec_001", "deerflow:vector:vec_002"}

    def test_bridge_deerflow_custom_prefix(
        self, btm: BiTemporalMemory, deerflow_session: list[dict],
    ) -> None:
        facts = bridge_deerflow_memory(
            deerflow_session, [], btm, subject_prefix="df", session_id="test",
        )
        assert all(f.subject.startswith("df:") for f in facts)


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    def test_parse_timestamp_naive(self) -> None:
        dt = _parse_timestamp("2026-03-01T10:00:00")
        assert dt.tzinfo == timezone.utc
        assert dt == datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)

    def test_parse_timestamp_none(self) -> None:
        dt = _parse_timestamp(None)
        assert dt.tzinfo is not None
        # Should be very recent -- within the last 5 seconds
        delta = datetime.now(timezone.utc) - dt
        assert delta.total_seconds() < 5

    def test_parse_timestamp_aware(self) -> None:
        dt = _parse_timestamp("2026-03-01T10:00:00+05:00")
        assert dt.tzinfo is not None
        # Should preserve the original offset, not force UTC
        assert dt.utcoffset().total_seconds() == 5 * 3600


class TestDeerflowSessionIdValidation:
    """Validation of session_id for DeerFlow session imports."""

    def test_blank_session_id_raises(self) -> None:
        btm = BiTemporalMemory()
        with pytest.raises(ValueError, match="non-empty string"):
            bridge_deerflow_memory(
                [{"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"}],
                [], btm, session_id="",
            )

    def test_whitespace_session_id_raises(self) -> None:
        btm = BiTemporalMemory()
        with pytest.raises(ValueError, match="non-empty string"):
            bridge_deerflow_memory(
                [{"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"}],
                [], btm, session_id="   ",
            )

    def test_none_session_id_raises(self) -> None:
        btm = BiTemporalMemory()
        with pytest.raises(ValueError, match="non-empty string"):
            bridge_deerflow_memory(
                [{"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"}],
                [], btm, session_id=None,
            )

    def test_integer_session_id_raises(self) -> None:
        btm = BiTemporalMemory()
        with pytest.raises(ValueError, match="non-empty string"):
            bridge_deerflow_memory(
                [{"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"}],
                [], btm, session_id=42,
            )

    def test_object_session_id_raises(self) -> None:
        btm = BiTemporalMemory()
        with pytest.raises(ValueError, match="non-empty string"):
            bridge_deerflow_memory(
                [{"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"}],
                [], btm, session_id=object(),
            )

    def test_whitespace_stripped_from_subject(self) -> None:
        btm = BiTemporalMemory()
        facts = bridge_deerflow_memory(
            [{"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"}],
            [], btm, session_id="  sess_A  ",
        )
        assert facts[0].subject == "deerflow:session:sess_A:0"
