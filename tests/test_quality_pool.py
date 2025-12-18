"""Tests for ubiquitin pool resource management."""
import pytest
from operon_ai.quality import (
    UbiquitinPool, PoolExhaustionStrategy,
    UbiquitinTag, DegronType,
)
from operon_ai.core.types import IntegrityLabel


class TestPoolExhaustionStrategy:
    def test_strategies_exist(self):
        assert PoolExhaustionStrategy.BLOCK.value == "block"
        assert PoolExhaustionStrategy.PASSTHROUGH.value == "passthrough"
        assert PoolExhaustionStrategy.RECYCLE_OLDEST.value == "recycle"


class TestUbiquitinPool:
    def test_create_pool(self):
        pool = UbiquitinPool(capacity=100)
        assert pool.capacity == 100
        assert pool.available == 100

    def test_allocate_returns_tag(self):
        pool = UbiquitinPool(capacity=100)
        tag = pool.allocate(origin="test_agent", confidence=0.9)
        assert tag is not None
        assert tag.origin == "test_agent"
        assert tag.confidence == 0.9
        assert pool.available == 99

    def test_allocate_with_degron(self):
        pool = UbiquitinPool(capacity=100)
        tag = pool.allocate(
            origin="test",
            confidence=0.9,
            degron=DegronType.UNSTABLE,
        )
        assert tag.degron == DegronType.UNSTABLE

    def test_allocate_exhausted_block(self):
        pool = UbiquitinPool(
            capacity=2,
            exhaustion_strategy=PoolExhaustionStrategy.BLOCK,
        )
        pool.allocate(origin="a")
        pool.allocate(origin="b")
        tag = pool.allocate(origin="c")
        assert tag is None
        assert pool.exhaustion_events == 1

    def test_allocate_exhausted_passthrough(self):
        pool = UbiquitinPool(
            capacity=1,
            exhaustion_strategy=PoolExhaustionStrategy.PASSTHROUGH,
        )
        pool.allocate(origin="a")
        tag = pool.allocate(origin="b")
        assert tag is not None  # Still creates tag
        assert tag.origin == "b"

    def test_allocate_exhausted_recycle_oldest(self):
        pool = UbiquitinPool(
            capacity=2,
            exhaustion_strategy=PoolExhaustionStrategy.RECYCLE_OLDEST,
        )
        pool.allocate(origin="a")
        pool.allocate(origin="b")
        # Pool is full, should recycle oldest
        tag = pool.allocate(origin="c")
        assert tag is not None
        assert tag.origin == "c"

    def test_recycle_returns_to_pool(self):
        pool = UbiquitinPool(capacity=10)
        tag = pool.allocate(origin="test", confidence=0.9)
        assert pool.available == 9
        pool.recycle(tag)
        assert pool.available == 10

    def test_recycle_chain_length(self):
        pool = UbiquitinPool(capacity=10)
        tag = UbiquitinTag(
            confidence=0.9,
            origin="test",
            generation=0,
            chain_length=3,
        )
        pool.available = 5
        pool.recycle(tag)
        assert pool.available == 8  # +3 from chain_length

    def test_recycle_capped_at_capacity(self):
        pool = UbiquitinPool(capacity=10)
        pool.available = 9
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0, chain_length=5)
        pool.recycle(tag)
        assert pool.available == 10  # Capped

    def test_status(self):
        pool = UbiquitinPool(capacity=100)
        pool.allocate(origin="test")
        status = pool.status()
        assert status["available"] == 99
        assert status["capacity"] == 100
        assert "utilization" in status
