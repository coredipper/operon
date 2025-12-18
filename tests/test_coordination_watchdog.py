"""Tests for Watchdog (apoptosis)."""
import pytest
from datetime import datetime, timedelta
from operon_ai.coordination.types import Phase, ResourceLock
from operon_ai.coordination.controller import (
    CellCycleController, OperationContext,
)
from operon_ai.coordination.watchdog import (
    Watchdog, ApoptosisEvent, ApoptosisReason,
)


class TestApoptosisReason:
    def test_reasons_exist(self):
        assert ApoptosisReason.TIMEOUT.value == "timeout"
        assert ApoptosisReason.STARVATION.value == "starvation"
        assert ApoptosisReason.NO_PROGRESS.value == "no_progress"
        assert ApoptosisReason.DEADLOCK.value == "deadlock"
        assert ApoptosisReason.MANUAL.value == "manual"


class TestApoptosisEvent:
    def test_create_event(self):
        event = ApoptosisEvent(
            operation_id="op1",
            agent_id="agent1",
            reason=ApoptosisReason.TIMEOUT,
            details="exceeded 30s timeout",
        )
        assert event.operation_id == "op1"
        assert event.reason == ApoptosisReason.TIMEOUT


class TestWatchdog:
    def test_create_watchdog(self):
        watchdog = Watchdog(
            max_operation_time=timedelta(seconds=30),
            starvation_timeout=timedelta(seconds=10),
            progress_timeout=timedelta(seconds=20),
        )
        assert watchdog.max_operation_time == timedelta(seconds=30)

    def test_check_no_violations(self):
        controller = CellCycleController()
        watchdog = Watchdog(max_operation_time=timedelta(seconds=30))

        ctx = controller.start_operation("op1", "agent1")
        # Just started, no violations

        events = watchdog.check(controller)
        assert len(events) == 0

    def test_detect_operation_timeout(self):
        controller = CellCycleController()
        watchdog = Watchdog(max_operation_time=timedelta(seconds=1))

        ctx = controller.start_operation("op1", "agent1")
        # Simulate old operation
        ctx.created_at = datetime.utcnow() - timedelta(seconds=5)

        events = watchdog.check(controller)
        assert len(events) == 1
        assert events[0].reason == ApoptosisReason.TIMEOUT
        assert events[0].operation_id == "op1"

    def test_detect_starvation(self):
        controller = CellCycleController()
        lock = ResourceLock(resource_id="file_a")
        controller.register_resource(lock)

        watchdog = Watchdog(starvation_timeout=timedelta(seconds=1))

        # First operation holds the resource
        ctx1 = controller.start_operation("op1", "agent1")
        controller.advance(ctx1)  # G1
        controller.acquire_resource(ctx1, "file_a")

        # Second operation is blocked
        ctx2 = controller.start_operation("op2", "agent2")
        controller.advance(ctx2)  # G1
        controller.acquire_resource(ctx2, "file_a")  # Blocked

        # Simulate waiting a long time
        ctx2.phase_entered_at = datetime.utcnow() - timedelta(seconds=5)

        events = watchdog.check(controller)
        assert any(e.reason == ApoptosisReason.STARVATION for e in events)

    def test_detect_no_progress(self):
        controller = CellCycleController()
        watchdog = Watchdog(progress_timeout=timedelta(seconds=1))

        ctx = controller.start_operation("op1", "agent1")
        ctx.resources_acquired = True
        controller.advance(ctx)  # G1
        controller.advance(ctx)  # S - executing

        # Simulate stuck in S phase
        ctx.phase_entered_at = datetime.utcnow() - timedelta(seconds=5)

        events = watchdog.check(controller)
        assert len(events) == 1
        assert events[0].reason == ApoptosisReason.NO_PROGRESS

    def test_detect_deadlock(self):
        controller = CellCycleController()
        lock_a = ResourceLock(resource_id="file_a")
        lock_b = ResourceLock(resource_id="file_b")
        controller.register_resource(lock_a)
        controller.register_resource(lock_b)

        watchdog = Watchdog()

        ctx1 = controller.start_operation("op1", "agent1")
        ctx2 = controller.start_operation("op2", "agent2")

        controller.advance(ctx1)
        controller.advance(ctx2)

        # Create deadlock
        controller.acquire_resource(ctx1, "file_a")
        controller.acquire_resource(ctx2, "file_b")
        controller.acquire_resource(ctx1, "file_b")  # Blocked
        controller.acquire_resource(ctx2, "file_a")  # Blocked - deadlock

        events = watchdog.check(controller)
        assert any(e.reason == ApoptosisReason.DEADLOCK for e in events)

    def test_break_deadlock_by_priority(self):
        controller = CellCycleController()
        lock_a = ResourceLock(resource_id="file_a")
        lock_b = ResourceLock(resource_id="file_b")
        controller.register_resource(lock_a)
        controller.register_resource(lock_b)

        watchdog = Watchdog(deadlock_strategy="priority")

        # Lower priority operation
        ctx1 = controller.start_operation("op1", "agent1", priority=1)
        # Higher priority operation
        ctx2 = controller.start_operation("op2", "agent2", priority=10)

        controller.advance(ctx1)
        controller.advance(ctx2)

        controller.acquire_resource(ctx1, "file_a")
        controller.acquire_resource(ctx2, "file_b")
        controller.acquire_resource(ctx1, "file_b")
        controller.acquire_resource(ctx2, "file_a")

        events = watchdog.check(controller)
        # Should kill lower priority operation
        deadlock_events = [e for e in events if e.reason == ApoptosisReason.DEADLOCK]
        assert len(deadlock_events) == 1
        assert deadlock_events[0].operation_id == "op1"  # Lower priority

    def test_break_deadlock_by_oldest(self):
        controller = CellCycleController()
        lock_a = ResourceLock(resource_id="file_a")
        lock_b = ResourceLock(resource_id="file_b")
        controller.register_resource(lock_a)
        controller.register_resource(lock_b)

        watchdog = Watchdog(deadlock_strategy="oldest")

        ctx1 = controller.start_operation("op1", "agent1")
        ctx1.created_at = datetime.utcnow() - timedelta(hours=1)  # Older

        ctx2 = controller.start_operation("op2", "agent2")
        ctx2.created_at = datetime.utcnow()  # Newer

        controller.advance(ctx1)
        controller.advance(ctx2)

        controller.acquire_resource(ctx1, "file_a")
        controller.acquire_resource(ctx2, "file_b")
        controller.acquire_resource(ctx1, "file_b")
        controller.acquire_resource(ctx2, "file_a")

        events = watchdog.check(controller)
        deadlock_events = [e for e in events if e.reason == ApoptosisReason.DEADLOCK]
        assert len(deadlock_events) == 1
        assert deadlock_events[0].operation_id == "op1"  # Oldest

    def test_execute_terminates_operations(self):
        controller = CellCycleController()
        watchdog = Watchdog(max_operation_time=timedelta(seconds=1))

        ctx = controller.start_operation("op1", "agent1")
        ctx.created_at = datetime.utcnow() - timedelta(seconds=5)

        events = watchdog.execute(controller)

        assert len(events) == 1
        assert "op1" not in controller.active_operations

    def test_watchdog_respects_exempt_operations(self):
        controller = CellCycleController()
        watchdog = Watchdog(max_operation_time=timedelta(seconds=1))

        ctx = controller.start_operation("op1", "agent1")
        ctx.created_at = datetime.utcnow() - timedelta(seconds=5)
        ctx.metadata["watchdog_exempt"] = True

        events = watchdog.check(controller)
        assert len(events) == 0  # Exempt from timeout
