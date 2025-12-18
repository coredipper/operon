"""Tests for integrated Coordination System organelle."""
import pytest
from datetime import timedelta
from operon_ai.coordination.types import (
    Phase, CheckpointResult, ResourceLock, LockResult,
)
from operon_ai.coordination.watchdog import ApoptosisReason
from operon_ai.coordination.system import CoordinationSystem, CoordinationResult


class TestCoordinationResult:
    def test_create_result(self):
        result = CoordinationResult(
            operation_id="op1",
            success=True,
            phase_reached=Phase.M,
        )
        assert result.operation_id == "op1"
        assert result.success is True


class TestCoordinationSystem:
    def test_create_system(self):
        system = CoordinationSystem()
        assert system.controller is not None
        assert system.watchdog is not None
        assert system.priority_manager is not None

    def test_register_resource(self):
        system = CoordinationSystem()
        system.register_resource("file_a")
        assert "file_a" in system.controller.resources

    def test_start_operation(self):
        system = CoordinationSystem()
        ctx = system.start_operation("op1", "agent1", priority=5)
        assert ctx.operation_id == "op1"
        assert ctx.agent_id == "agent1"
        assert ctx.priority == 5

    def test_execute_simple_operation(self):
        system = CoordinationSystem()

        def work():
            return {"result": "success"}

        result = system.execute_operation(
            operation_id="op1",
            agent_id="agent1",
            work_fn=work,
        )

        assert result.success is True
        assert result.result == {"result": "success"}

    def test_execute_with_resources(self):
        system = CoordinationSystem()
        system.register_resource("file_a")

        def work():
            return {"used": "file_a"}

        result = system.execute_operation(
            operation_id="op1",
            agent_id="agent1",
            resources=["file_a"],
            work_fn=work,
        )

        assert result.success is True
        # Resource should be released after completion
        assert system.controller.resources["file_a"].is_available

    def test_execute_with_validation(self):
        system = CoordinationSystem()

        def work():
            return {"value": 42}

        def validate(result):
            return result["value"] > 0

        result = system.execute_operation(
            operation_id="op1",
            agent_id="agent1",
            work_fn=work,
            validate_fn=validate,
        )

        assert result.success is True

    def test_execute_validation_fails(self):
        system = CoordinationSystem()

        def work():
            return {"value": -1}

        def validate(result):
            return result["value"] > 0  # Will fail

        result = system.execute_operation(
            operation_id="op1",
            agent_id="agent1",
            work_fn=work,
            validate_fn=validate,
        )

        assert result.success is False
        assert "validation" in result.error.lower()

    def test_execute_work_fails(self):
        system = CoordinationSystem()

        def work():
            raise ValueError("work failed")

        result = system.execute_operation(
            operation_id="op1",
            agent_id="agent1",
            work_fn=work,
        )

        assert result.success is False
        assert "work failed" in result.error

    def test_resource_blocking(self):
        system = CoordinationSystem()
        system.register_resource("shared")

        # Start first operation
        ctx1 = system.start_operation("op1", "agent1")
        system.controller.advance(ctx1)  # G1
        lock_result = system.controller.acquire_resource(ctx1, "shared")
        assert lock_result == LockResult.ACQUIRED

        # Second operation should be blocked
        ctx2 = system.start_operation("op2", "agent2")
        system.controller.advance(ctx2)
        lock_result = system.controller.acquire_resource(ctx2, "shared")
        assert lock_result == LockResult.BLOCKED

    def test_priority_inheritance_applied(self):
        system = CoordinationSystem()
        system.register_resource("shared")

        # Low priority holds resource
        ctx_low = system.start_operation("op_low", "agent1", priority=1)
        system.controller.advance(ctx_low)
        system.controller.acquire_resource(ctx_low, "shared")

        # High priority blocked
        ctx_high = system.start_operation("op_high", "agent2", priority=10)
        system.controller.advance(ctx_high)
        system.controller.acquire_resource(ctx_high, "shared")  # Blocked

        # Run maintenance to apply priority inheritance
        system.run_maintenance()

        # Low priority should be boosted
        assert ctx_low.priority == 10

    def test_watchdog_kills_timeout(self):
        system = CoordinationSystem(
            max_operation_time=timedelta(seconds=1),
        )

        from datetime import datetime
        ctx = system.start_operation("op1", "agent1")
        ctx.created_at = datetime.utcnow() - timedelta(seconds=5)

        events = system.run_maintenance()

        assert len(events["apoptosis"]) == 1
        assert events["apoptosis"][0].reason == ApoptosisReason.TIMEOUT
        assert "op1" not in system.controller.active_operations

    def test_manual_kill(self):
        system = CoordinationSystem()
        ctx = system.start_operation("op1", "agent1")

        event = system.kill_operation("op1", "test kill")

        assert event is not None
        assert event.reason == ApoptosisReason.MANUAL
        assert "op1" not in system.controller.active_operations

    def test_health_check(self):
        system = CoordinationSystem()
        system.register_resource("file_a")
        system.start_operation("op1", "agent1")

        health = system.health()

        assert health["active_operations"] == 1
        assert health["registered_resources"] == 1
        assert "watchdog" in health
        assert "priority" in health

    def test_shutdown_aborts_all(self):
        system = CoordinationSystem()
        system.start_operation("op1", "agent1")
        system.start_operation("op2", "agent2")

        assert len(system.controller.active_operations) == 2

        system.shutdown()

        assert len(system.controller.active_operations) == 0
