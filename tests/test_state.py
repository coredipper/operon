"""Tests for state management: ATP_Store, HistoneStore."""

import pytest
from operon_ai.state.metabolism import ATP_Store
from operon_ai.state.histone import HistoneStore


class TestATPStore:
    """Tests for the ATP_Store (metabolic budget) class."""

    def test_initial_budget(self):
        """ATP_Store initializes with the specified budget."""
        store = ATP_Store(budget=100)
        assert store.atp == 100

    def test_consume_success(self):
        """consume() returns True and deducts when sufficient ATP."""
        store = ATP_Store(budget=100)
        result = store.consume(cost=30)
        assert result is True
        assert store.atp == 70

    def test_consume_insufficient(self):
        """consume() returns False when insufficient ATP."""
        store = ATP_Store(budget=10)
        result = store.consume(cost=20)
        assert result is False
        assert store.atp == 10  # Budget unchanged

    def test_consume_exact_budget(self):
        """consume() succeeds when cost equals remaining budget."""
        store = ATP_Store(budget=50)
        result = store.consume(cost=50)
        assert result is True
        assert store.atp == 0

    def test_multiple_consumptions(self):
        """Multiple consume() calls correctly deplete budget."""
        store = ATP_Store(budget=100)
        assert store.consume(cost=25) is True
        assert store.consume(cost=25) is True
        assert store.consume(cost=25) is True
        assert store.atp == 25
        assert store.consume(cost=30) is False  # Would exceed
        assert store.atp == 25

    def test_zero_cost_consumption(self):
        """consume() with zero cost succeeds without change."""
        store = ATP_Store(budget=100)
        result = store.consume(cost=0)
        assert result is True
        assert store.atp == 100


class TestHistoneStore:
    """Tests for the HistoneStore (epigenetic memory) class."""

    def test_initial_state_empty(self):
        """HistoneStore initializes with no methylations."""
        store = HistoneStore()
        assert store.methylations == []

    def test_add_marker(self):
        """add_marker() appends to methylations list."""
        store = HistoneStore()
        store.add_marker("Avoid SQL injection patterns")
        assert len(store.methylations) == 1
        assert "SQL injection" in store.methylations[0]

    def test_add_multiple_markers(self):
        """Multiple markers can be added."""
        store = HistoneStore()
        store.add_marker("Lesson 1")
        store.add_marker("Lesson 2")
        store.add_marker("Lesson 3")
        assert len(store.methylations) == 3

    def test_retrieve_context_empty(self):
        """retrieve_context() returns empty string when no markers."""
        store = HistoneStore()
        context = store.retrieve_context("any query")
        assert context == ""

    def test_retrieve_context_with_markers(self):
        """retrieve_context() returns formatted warnings."""
        store = HistoneStore()
        store.add_marker("Avoid retrying failed API calls")
        store.add_marker("Check for null values")

        context = store.retrieve_context("some query")
        assert "PREVIOUS FAILURES" in context
        assert "Avoid retrying failed API calls" in context
        assert "Check for null values" in context

    def test_markers_persist(self):
        """Markers persist across multiple retrievals."""
        store = HistoneStore()
        store.add_marker("Important lesson")

        # Multiple retrievals should still show the marker
        for _ in range(3):
            context = store.retrieve_context("query")
            assert "Important lesson" in context
