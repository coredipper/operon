"""Tests for Cell Cycle Controller."""
import pytest
from datetime import datetime, timedelta
from operon_ai.coordination.types import (
    Phase, CheckpointResult, ResourceLock, DependencyGraph, LockResult,
)
from operon_ai.coordination.controller import (
    CellCycleController, Checkpoint, OperationContext, OperationResult,
)


class TestCheckpoint:
    def test_create_checkpoint(self):
        checkpoint = Checkpoint(
            phase=Phase.G1,
            condition=lambda ctx: True,
            timeout=timedelta(seconds=10),
        )
        assert checkpoint.phase == Phase.G1
        assert checkpoint.timeout == timedelta(seconds=10)

    def test_checkpoint_evaluate_pass(self):
        checkpoint = Checkpoint(
            phase=Phase.G1,
            condition=lambda ctx: ctx.resources_acquired,
        )
        ctx = OperationContext(operation_id="op1", agent_id="agent1")
        ctx.resources_acquired = True
        result = checkpoint.evaluate(ctx)
        assert result == CheckpointResult.PASSED

    def test_checkpoint_evaluate_fail(self):
        checkpoint = Checkpoint(
            phase=Phase.G1,
            condition=lambda ctx: ctx.resources_acquired,
        )
        ctx = OperationContext(operation_id="op1", agent_id="agent1")
        ctx.resources_acquired = False
        result = checkpoint.evaluate(ctx)
        assert result == CheckpointResult.FAILED


class TestOperationContext:
    def test_create_context(self):
        ctx = OperationContext(
            operation_id="op1",
            agent_id="agent1",
            priority=5,
        )
        assert ctx.operation_id == "op1"
        assert ctx.agent_id == "agent1"
        assert ctx.priority == 5
        assert ctx.phase == Phase.G0

    def test_context_add_acquired_resource(self):
        ctx = OperationContext(operation_id="op1", agent_id="agent1")
        lock = ResourceLock(resource_id="file_a")
        ctx.add_acquired_resource(lock)
        assert "file_a" in ctx.acquired_resources

    def test_context_set_result(self):
        ctx = OperationContext(operation_id="op1", agent_id="agent1")
        ctx.set_result({"output": "data"})
        assert ctx.result == {"output": "data"}


class TestCellCycleController:
    def test_create_controller(self):
        controller = CellCycleController()
        assert len(controller.checkpoints) > 0  # Default checkpoints

    def test_start_operation(self):
        controller = CellCycleController()
        ctx = controller.start_operation(
            operation_id="op1",
            agent_id="agent1",
        )
        assert ctx.operation_id == "op1"
        assert ctx.phase == Phase.G0
        assert "op1" in controller.active_operations

    def test_advance_to_g1(self):
        controller = CellCycleController()
        ctx = controller.start_operation("op1", "agent1")

        result = controller.advance(ctx)
        assert ctx.phase == Phase.G1
        assert result == CheckpointResult.PASSED

    def test_acquire_resource(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)

        ctx = controller.start_operation("op1", "agent1")
        controller.advance(ctx)  # Move to G1

        acquired = controller.acquire_resource(ctx, "file_a")
        assert acquired == LockResult.ACQUIRED
        assert "file_a" in ctx.acquired_resources

    def test_acquire_resource_blocked(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)

        ctx1 = controller.start_operation("op1", "agent1")
        ctx2 = controller.start_operation("op2", "agent2")

        controller.advance(ctx1)  # G1
        controller.advance(ctx2)  # G1

        controller.acquire_resource(ctx1, "file_a")
        result = controller.acquire_resource(ctx2, "file_a")

        assert result == LockResult.BLOCKED

    def test_deadlock_detection(self):
        controller = CellCycleController()
        lock_a = ResourceLock(resource_id="file_a")
        lock_b = ResourceLock(resource_id="file_b")
        controller.register_resource(lock_a)
        controller.register_resource(lock_b)

        ctx1 = controller.start_operation("op1", "agent1")
        ctx2 = controller.start_operation("op2", "agent2")

        controller.advance(ctx1)  # G1
        controller.advance(ctx2)  # G1

        # op1 holds file_a, wants file_b
        controller.acquire_resource(ctx1, "file_a")
        # op2 holds file_b, wants file_a
        controller.acquire_resource(ctx2, "file_b")

        # Now create the deadlock
        controller.acquire_resource(ctx1, "file_b")  # Blocked
        controller.acquire_resource(ctx2, "file_a")  # Blocked

        deadlock = controller.check_deadlock()
        assert deadlock is not None
        assert len(deadlock.agents) == 2

    def test_full_cycle(self):
        controller = CellCycleController()
        ctx = controller.start_operation("op1", "agent1")

        # G0 -> G1
        controller.advance(ctx)
        assert ctx.phase == Phase.G1

        # G1 -> S (mark resources acquired)
        ctx.resources_acquired = True
        controller.advance(ctx)
        assert ctx.phase == Phase.S

        # S -> G2 (set result)
        ctx.set_result({"output": "success"})
        ctx.execution_complete = True
        controller.advance(ctx)
        assert ctx.phase == Phase.G2

        # G2 -> M (mark validated)
        ctx.validation_passed = True
        controller.advance(ctx)
        assert ctx.phase == Phase.M

        # Complete operation
        op_result = controller.complete_operation(ctx)
        assert op_result.success is True
        assert ctx.phase == Phase.G0

    def test_checkpoint_failure_aborts(self):
        controller = CellCycleController()
        ctx = controller.start_operation("op1", "agent1")

        controller.advance(ctx)  # G0 -> G1

        # Try to advance without resources
        ctx.resources_acquired = False
        result = controller.advance(ctx)
        assert result == CheckpointResult.FAILED
        assert ctx.phase == Phase.G1  # Didn't advance

    def test_release_resources_on_complete(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)

        ctx = controller.start_operation("op1", "agent1")
        controller.advance(ctx)  # G1
        controller.acquire_resource(ctx, "file_a")

        assert not lock.is_available

        # Complete full cycle
        ctx.resources_acquired = True
        controller.advance(ctx)  # S
        ctx.set_result({})
        ctx.execution_complete = True
        controller.advance(ctx)  # G2
        ctx.validation_passed = True
        controller.advance(ctx)  # M
        controller.complete_operation(ctx)

        assert lock.is_available  # Released

    def test_abort_operation_releases_resources(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)

        ctx = controller.start_operation("op1", "agent1")
        controller.advance(ctx)
        controller.acquire_resource(ctx, "file_a")

        controller.abort_operation(ctx, reason="test abort")

        assert lock.is_available
        assert ctx.phase == Phase.G0
        assert "op1" not in controller.active_operations

    def test_get_operation_stats(self):
        controller = CellCycleController()
        ctx = controller.start_operation("op1", "agent1")
        controller.advance(ctx)

        stats = controller.stats()
        assert stats["active_operations"] == 1
        assert "phase_distribution" in stats
