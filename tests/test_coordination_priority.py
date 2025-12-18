"""Tests for Priority Inheritance."""
import pytest
from operon_ai.coordination.types import ResourceLock, DependencyGraph
from operon_ai.coordination.controller import CellCycleController, OperationContext
from operon_ai.coordination.priority import (
    PriorityInheritance, PriorityBoost,
)


class TestPriorityBoost:
    def test_create_boost(self):
        boost = PriorityBoost(
            operation_id="op1",
            original_priority=1,
            boosted_priority=10,
            reason="blocked by op2",
        )
        assert boost.operation_id == "op1"
        assert boost.original_priority == 1
        assert boost.boosted_priority == 10


class TestPriorityInheritance:
    def test_create_manager(self):
        manager = PriorityInheritance()
        assert len(manager.active_boosts) == 0

    def test_no_boost_needed(self):
        controller = CellCycleController()
        manager = PriorityInheritance()

        # No blocking, no boost needed
        ctx = controller.start_operation("op1", "agent1", priority=5)

        boosts = manager.check_and_boost(controller)
        assert len(boosts) == 0

    def test_boost_when_high_priority_blocked(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)
        manager = PriorityInheritance()

        # Low priority holds resource
        ctx_low = controller.start_operation("op_low", "agent1", priority=1)
        controller.advance(ctx_low)  # G1
        controller.acquire_resource(ctx_low, "file_a")

        # High priority blocked
        ctx_high = controller.start_operation("op_high", "agent2", priority=10)
        controller.advance(ctx_high)  # G1
        controller.acquire_resource(ctx_high, "file_a")  # Blocked

        boosts = manager.check_and_boost(controller)

        assert len(boosts) == 1
        assert boosts[0].operation_id == "op_low"
        assert boosts[0].boosted_priority == 10
        assert ctx_low.priority == 10  # Boosted

    def test_restore_on_release(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)
        manager = PriorityInheritance()

        ctx_low = controller.start_operation("op_low", "agent1", priority=1)
        controller.advance(ctx_low)
        controller.acquire_resource(ctx_low, "file_a")

        ctx_high = controller.start_operation("op_high", "agent2", priority=10)
        controller.advance(ctx_high)
        controller.acquire_resource(ctx_high, "file_a")  # Blocked

        # Boost priority
        manager.check_and_boost(controller)
        assert ctx_low.priority == 10

        # Release resource
        controller.release_resource(ctx_low, "file_a")

        # Restore original priority
        manager.restore_priority(ctx_low)
        assert ctx_low.priority == 1

    def test_transitive_boost(self):
        controller = CellCycleController()
        lock_a = ResourceLock(resource_id="file_a")
        lock_b = ResourceLock(resource_id="file_b")
        controller.register_resource(lock_a)
        controller.register_resource(lock_b)
        manager = PriorityInheritance()

        # Chain: op1 (pri=1) holds A, op2 (pri=5) holds B and blocked on A
        ctx1 = controller.start_operation("op1", "agent1", priority=1)
        controller.advance(ctx1)
        controller.acquire_resource(ctx1, "file_a")

        ctx2 = controller.start_operation("op2", "agent2", priority=5)
        controller.advance(ctx2)
        controller.acquire_resource(ctx2, "file_b")
        controller.acquire_resource(ctx2, "file_a")  # Blocked on op1

        # op3 (pri=10) blocked on op2
        ctx3 = controller.start_operation("op3", "agent3", priority=10)
        controller.advance(ctx3)
        controller.acquire_resource(ctx3, "file_b")  # Blocked on op2

        # Transitive: op3(10) -> op2 -> op1
        # op2 should get boosted to 10
        # op1 should also get boosted to 10
        boosts = manager.check_and_boost(controller)

        # op2 blocked by op1, op2's priority should be inherited by op1
        assert ctx2.priority == 10  # Boosted from 5 to 10
        assert ctx1.priority == 10  # Also boosted from 1 to 10 (transitive)

    def test_multiple_waiters_max_priority(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)
        manager = PriorityInheritance()

        ctx_holder = controller.start_operation("holder", "agent1", priority=1)
        controller.advance(ctx_holder)
        controller.acquire_resource(ctx_holder, "file_a")

        # Multiple waiters with different priorities
        ctx_w1 = controller.start_operation("waiter1", "agent2", priority=5)
        controller.advance(ctx_w1)
        controller.acquire_resource(ctx_w1, "file_a")  # Blocked

        ctx_w2 = controller.start_operation("waiter2", "agent3", priority=8)
        controller.advance(ctx_w2)
        controller.acquire_resource(ctx_w2, "file_a")  # Blocked

        ctx_w3 = controller.start_operation("waiter3", "agent4", priority=3)
        controller.advance(ctx_w3)
        controller.acquire_resource(ctx_w3, "file_a")  # Blocked

        manager.check_and_boost(controller)

        # Holder should get max waiter priority (8)
        assert ctx_holder.priority == 8

    def test_no_boost_if_already_higher(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)
        manager = PriorityInheritance()

        # Holder has higher priority than waiter
        ctx_holder = controller.start_operation("holder", "agent1", priority=10)
        controller.advance(ctx_holder)
        controller.acquire_resource(ctx_holder, "file_a")

        ctx_waiter = controller.start_operation("waiter", "agent2", priority=5)
        controller.advance(ctx_waiter)
        controller.acquire_resource(ctx_waiter, "file_a")

        boosts = manager.check_and_boost(controller)
        assert len(boosts) == 0  # No boost needed

    def test_stats(self):
        manager = PriorityInheritance()
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)

        ctx_low = controller.start_operation("op_low", "agent1", priority=1)
        controller.advance(ctx_low)
        controller.acquire_resource(ctx_low, "file_a")

        ctx_high = controller.start_operation("op_high", "agent2", priority=10)
        controller.advance(ctx_high)
        controller.acquire_resource(ctx_high, "file_a")

        manager.check_and_boost(controller)

        stats = manager.stats()
        assert stats["total_boosts"] == 1
        assert stats["active_boosts"] == 1
