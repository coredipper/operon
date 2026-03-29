"""Tests for AsyncThink-inspired Fork/Join execution."""

import pytest

from operon_ai.convergence.async_thinking import (
    AsyncOrganizer,
    AsyncThinkResult,
    async_stage_handler,
)


# ── Helpers ──────────────────────────────────────────────────────


def _echo_handler(query: str) -> str:
    """Simple handler that echoes the query with a prefix."""
    return f"result:{query}"


def _upper_handler(query: str) -> str:
    return query.upper()


# ── AsyncOrganizer.fork ─────────────────────────────────────────


class TestAsyncOrganizerFork:
    def test_async_organizer_fork_basic(self):
        """3 sub-queries should produce 3 outputs."""
        org = AsyncOrganizer(capacity=4)
        result = org.fork(
            task="parent",
            sub_queries=["a", "b", "c"],
            handler=_echo_handler,
        )
        assert len(result.outputs) == 3
        assert result.outputs == ("result:a", "result:b", "result:c")
        assert result.fork_count == 3

    def test_async_organizer_concurrency_ratio(self):
        """capacity=4, 2 queries -> eta = 2/4 = 0.5."""
        org = AsyncOrganizer(capacity=4)
        result = org.fork(
            task="parent",
            sub_queries=["x", "y"],
            handler=_echo_handler,
        )
        assert result.concurrency_ratio == pytest.approx(0.5)
        assert result.fork_count == 2

    def test_async_organizer_empty_queries(self):
        """0 queries -> empty outputs, 0 critical path, 0 total_ms."""
        org = AsyncOrganizer(capacity=4)
        result = org.fork(
            task="parent",
            sub_queries=[],
            handler=_echo_handler,
        )
        assert result.outputs == ()
        assert result.critical_path_ms == 0.0
        assert result.total_ms == 0.0
        assert result.fork_count == 0
        assert result.concurrency_ratio == pytest.approx(0.0)


# ── AsyncOrganizer.critical_path_latency ────────────────────────


class TestCriticalPathLatency:
    def test_critical_path_linear(self):
        """A -> B -> C chain should have CPL = 3."""
        dag = {
            "C": ["B"],
            "B": ["A"],
            "A": [],
        }
        org = AsyncOrganizer()
        assert org.critical_path_latency(dag) == pytest.approx(3.0)

    def test_critical_path_parallel(self):
        """A, B, C independent -> CPL = 1."""
        dag = {
            "A": [],
            "B": [],
            "C": [],
        }
        org = AsyncOrganizer()
        assert org.critical_path_latency(dag) == pytest.approx(1.0)

    def test_critical_path_diamond(self):
        """A -> B, A -> C, B -> D, C -> D  =>  CPL = 3."""
        dag = {
            "D": ["B", "C"],
            "B": ["A"],
            "C": ["A"],
            "A": [],
        }
        org = AsyncOrganizer()
        assert org.critical_path_latency(dag) == pytest.approx(3.0)

    def test_critical_path_empty_dag(self):
        """Empty DAG -> CPL = 0."""
        org = AsyncOrganizer()
        assert org.critical_path_latency({}) == pytest.approx(0.0)


# ── async_stage_handler ─────────────────────────────────────────


class TestAsyncStageHandler:
    def test_async_stage_handler_basic(self):
        """Build a handler, call it, verify output dict structure."""
        org = AsyncOrganizer(capacity=4)
        handler = async_stage_handler(
            organizer=org,
            decompose=lambda task: task.split(","),
            handler=_upper_handler,
        )
        result = handler("hello,world")

        assert "output" in result
        assert "async_think" in result
        assert result["output"] == ("HELLO", "WORLD")
        assert result["async_think"]["fork_count"] == 2
        assert result["async_think"]["concurrency_ratio"] == pytest.approx(0.5)

    def test_async_stage_handler_with_join(self):
        """Custom join function combines outputs."""
        org = AsyncOrganizer(capacity=4)
        handler = async_stage_handler(
            organizer=org,
            decompose=lambda task: task.split(","),
            handler=_upper_handler,
            join=lambda outputs: " + ".join(outputs),
        )
        result = handler("foo,bar,baz")

        assert result["output"] == "FOO + BAR + BAZ"
        assert result["async_think"]["fork_count"] == 3


# ── AsyncThinkResult fields ─────────────────────────────────────


class TestAsyncThinkResultFields:
    def test_async_think_result_fields(self):
        """Verify all fields are populated and have correct types."""
        r = AsyncThinkResult(
            outputs=("a", "b"),
            concurrency_ratio=0.5,
            critical_path_ms=12.3,
            total_ms=24.6,
            fork_count=2,
        )
        assert r.outputs == ("a", "b")
        assert isinstance(r.concurrency_ratio, float)
        assert isinstance(r.critical_path_ms, float)
        assert isinstance(r.total_ms, float)
        assert isinstance(r.fork_count, int)
        assert r.fork_count == 2

    def test_async_think_result_frozen(self):
        """AsyncThinkResult is frozen, so attribute assignment should raise."""
        r = AsyncThinkResult(
            outputs=(),
            concurrency_ratio=0.0,
            critical_path_ms=0.0,
            total_ms=0.0,
            fork_count=0,
        )
        with pytest.raises(AttributeError):
            r.fork_count = 99  # type: ignore[misc]
