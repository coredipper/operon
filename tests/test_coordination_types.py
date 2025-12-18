"""Tests for coordination system types."""
import pytest
from datetime import datetime, timedelta
from operon_ai.coordination.types import (
    Phase, CheckpointResult, ResourceLock, DependencyGraph,
    LockResult, DeadlockInfo,
)


class TestPhase:
    def test_phase_values(self):
        assert Phase.G0.value == "g0"
        assert Phase.G1.value == "g1"
        assert Phase.S.value == "s"
        assert Phase.G2.value == "g2"
        assert Phase.M.value == "m"

    def test_phase_ordering(self):
        # G0 -> G1 -> S -> G2 -> M -> G0
        assert Phase.G1.next() == Phase.S
        assert Phase.S.next() == Phase.G2
        assert Phase.G2.next() == Phase.M
        assert Phase.M.next() == Phase.G0


class TestCheckpointResult:
    def test_checkpoint_results(self):
        assert CheckpointResult.PASSED.value == "passed"
        assert CheckpointResult.FAILED.value == "failed"
        assert CheckpointResult.WAITING.value == "waiting"
        assert CheckpointResult.TIMEOUT.value == "timeout"


class TestResourceLock:
    def test_create_lock(self):
        lock = ResourceLock(resource_id="file_a")
        assert lock.resource_id == "file_a"
        assert lock.owner is None
        assert lock.is_available

    def test_try_acquire_success(self):
        lock = ResourceLock(resource_id="file_a")
        result = lock.try_acquire(owner="agent_1", priority=1)
        assert result == LockResult.ACQUIRED
        assert lock.owner == "agent_1"
        assert not lock.is_available

    def test_try_acquire_already_owned(self):
        lock = ResourceLock(resource_id="file_a")
        lock.try_acquire(owner="agent_1", priority=1)
        result = lock.try_acquire(owner="agent_2", priority=1)
        assert result == LockResult.BLOCKED
        assert lock.owner == "agent_1"

    def test_try_acquire_reentrant(self):
        lock = ResourceLock(resource_id="file_a")
        lock.try_acquire(owner="agent_1", priority=1)
        result = lock.try_acquire(owner="agent_1", priority=1)  # Same owner
        assert result == LockResult.REENTRANT
        assert lock.hold_count == 2

    def test_try_acquire_preemption(self):
        lock = ResourceLock(resource_id="file_a", allow_preemption=True)
        lock.try_acquire(owner="low_priority", priority=1)
        result = lock.try_acquire(owner="high_priority", priority=10)
        assert result == LockResult.PREEMPTED
        assert lock.owner == "high_priority"

    def test_release(self):
        lock = ResourceLock(resource_id="file_a")
        lock.try_acquire(owner="agent_1", priority=1)
        released = lock.release(owner="agent_1")
        assert released is True
        assert lock.is_available

    def test_release_reentrant(self):
        lock = ResourceLock(resource_id="file_a")
        lock.try_acquire(owner="agent_1", priority=1)
        lock.try_acquire(owner="agent_1", priority=1)  # Reentrant
        lock.release(owner="agent_1")
        assert not lock.is_available  # Still held (count=1)
        lock.release(owner="agent_1")
        assert lock.is_available  # Now released

    def test_release_wrong_owner(self):
        lock = ResourceLock(resource_id="file_a")
        lock.try_acquire(owner="agent_1", priority=1)
        released = lock.release(owner="agent_2")  # Wrong owner
        assert released is False
        assert not lock.is_available

    def test_waiting_list(self):
        lock = ResourceLock(resource_id="file_a")
        lock.try_acquire(owner="agent_1", priority=1)
        lock.try_acquire(owner="agent_2", priority=2)
        lock.try_acquire(owner="agent_3", priority=3)

        assert len(lock.waiting_list) == 2
        # Higher priority should be first
        assert lock.waiting_list[0][0] == "agent_3"  # (owner, priority)

    def test_hold_duration(self):
        lock = ResourceLock(resource_id="file_a")
        lock.try_acquire(owner="agent_1", priority=1)
        duration = lock.hold_duration
        assert duration is not None
        assert duration >= timedelta(0)


class TestDependencyGraph:
    def test_create_graph(self):
        graph = DependencyGraph()
        assert len(graph.edges) == 0

    def test_add_dependency(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="agent_1", blocking="agent_2", resource="file_a")
        assert len(graph.edges) == 1

    def test_remove_dependency(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="agent_1", blocking="agent_2", resource="file_a")
        graph.remove_dependency(waiter="agent_1", blocking="agent_2")
        assert len(graph.edges) == 0

    def test_detect_no_cycle(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="a", blocking="b", resource="r1")
        graph.add_dependency(waiter="b", blocking="c", resource="r2")

        deadlock = graph.detect_cycle()
        assert deadlock is None

    def test_detect_simple_cycle(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="a", blocking="b", resource="r1")
        graph.add_dependency(waiter="b", blocking="a", resource="r2")

        deadlock = graph.detect_cycle()
        assert deadlock is not None
        assert set(deadlock.agents) == {"a", "b"}

    def test_detect_multi_node_cycle(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="a", blocking="b", resource="r1")
        graph.add_dependency(waiter="b", blocking="c", resource="r2")
        graph.add_dependency(waiter="c", blocking="a", resource="r3")

        deadlock = graph.detect_cycle()
        assert deadlock is not None
        assert len(deadlock.agents) == 3

    def test_deadlock_info(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="a", blocking="b", resource="file_a")
        graph.add_dependency(waiter="b", blocking="a", resource="file_b")

        deadlock = graph.detect_cycle()
        assert deadlock is not None
        assert "file_a" in deadlock.resources or "file_b" in deadlock.resources

    def test_get_blocking_chain(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="a", blocking="b", resource="r1")
        graph.add_dependency(waiter="b", blocking="c", resource="r2")

        chain = graph.get_blocking_chain("a")
        assert chain == ["a", "b", "c"]

    def test_clear(self):
        graph = DependencyGraph()
        graph.add_dependency(waiter="a", blocking="b", resource="r1")
        graph.clear()
        assert len(graph.edges) == 0
