"""Cell Cycle Controller for operation coordination."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from .types import (
    Phase, CheckpointResult, LockResult,
    ResourceLock, DependencyGraph, DeadlockInfo,
)


@dataclass
class Checkpoint:
    """
    Checkpoint condition for a cell cycle phase.

    Biological parallel: Cell cycle checkpoints like
    G1/S, G2/M that ensure proper progression.
    """

    phase: Phase
    condition: Callable[["OperationContext"], bool]
    timeout: Optional[timedelta] = None
    name: str = ""

    def evaluate(self, ctx: "OperationContext") -> CheckpointResult:
        """Evaluate checkpoint condition."""
        try:
            if self.condition(ctx):
                return CheckpointResult.PASSED
            return CheckpointResult.FAILED
        except Exception:
            return CheckpointResult.FAILED


@dataclass
class OperationContext:
    """
    Context for an operation going through the cell cycle.
    """

    operation_id: str
    agent_id: str
    priority: int = 0

    # Phase tracking
    phase: Phase = Phase.G0
    phase_entered_at: datetime = field(default_factory=datetime.utcnow)

    # Resource tracking
    acquired_resources: dict[str, ResourceLock] = field(default_factory=dict)
    resources_acquired: bool = False

    # Execution tracking
    result: Any = None
    execution_complete: bool = False

    # Validation tracking
    validation_passed: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def add_acquired_resource(self, lock: ResourceLock) -> None:
        """Track an acquired resource."""
        self.acquired_resources[lock.resource_id] = lock

    def set_result(self, result: Any) -> None:
        """Set operation result."""
        self.result = result

    def enter_phase(self, phase: Phase) -> None:
        """Enter a new phase."""
        self.phase = phase
        self.phase_entered_at = datetime.utcnow()


@dataclass
class OperationResult:
    """Result of a completed operation."""

    operation_id: str
    agent_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: Optional[timedelta] = None


@dataclass
class CellCycleController:
    """
    Manages agent operations through cell cycle checkpoints.

    Coordinates resource acquisition, execution, validation,
    and commitment of operations with deadlock detection.
    """

    # Checkpoints per phase
    checkpoints: dict[Phase, list[Checkpoint]] = field(default_factory=dict)

    # Resources
    resources: dict[str, ResourceLock] = field(default_factory=dict)
    dependency_graph: DependencyGraph = field(default_factory=DependencyGraph)

    # Active operations
    active_operations: dict[str, OperationContext] = field(default_factory=dict)

    def __post_init__(self):
        # Set up default checkpoints if none provided
        if not self.checkpoints:
            self.checkpoints = {
                Phase.G0: [Checkpoint(
                    phase=Phase.G0,
                    condition=lambda ctx: True,  # Always pass G0
                    name="g0_ready",
                )],
                Phase.G1: [Checkpoint(
                    phase=Phase.G1,
                    condition=lambda ctx: ctx.resources_acquired,
                    name="g1_resources",
                )],
                Phase.S: [Checkpoint(
                    phase=Phase.S,
                    condition=lambda ctx: ctx.execution_complete,
                    name="s_execution",
                )],
                Phase.G2: [Checkpoint(
                    phase=Phase.G2,
                    condition=lambda ctx: ctx.validation_passed,
                    name="g2_validation",
                )],
                Phase.M: [Checkpoint(
                    phase=Phase.M,
                    condition=lambda ctx: True,  # Commit if reached
                    name="m_commit",
                )],
            }

    def register_resource(self, lock: ResourceLock) -> None:
        """Register a resource for coordination."""
        self.resources[lock.resource_id] = lock

    def start_operation(
        self,
        operation_id: str,
        agent_id: str,
        priority: int = 0,
    ) -> OperationContext:
        """Start a new operation in G0 phase."""
        ctx = OperationContext(
            operation_id=operation_id,
            agent_id=agent_id,
            priority=priority,
        )
        self.active_operations[operation_id] = ctx
        return ctx

    def advance(self, ctx: OperationContext) -> CheckpointResult:
        """
        Try to advance to next phase.

        Returns checkpoint result indicating success or failure.
        """
        current_phase = ctx.phase

        # Check current phase checkpoints (for transition out)
        checkpoints = self.checkpoints.get(current_phase, [])
        for checkpoint in checkpoints:
            result = checkpoint.evaluate(ctx)
            if result != CheckpointResult.PASSED:
                return result

        # All checkpoints passed - advance
        next_phase = current_phase.next()
        ctx.enter_phase(next_phase)
        return CheckpointResult.PASSED

    def acquire_resource(
        self,
        ctx: OperationContext,
        resource_id: str,
    ) -> LockResult:
        """
        Attempt to acquire a resource for an operation.

        Updates dependency graph if blocked.
        """
        if resource_id not in self.resources:
            raise ValueError(f"Unknown resource: {resource_id}")

        lock = self.resources[resource_id]
        result = lock.try_acquire(owner=ctx.operation_id, priority=ctx.priority)

        if result == LockResult.ACQUIRED or result == LockResult.REENTRANT:
            ctx.add_acquired_resource(lock)
            # Remove any dependency since we now own it
            self.dependency_graph.remove_all_for_agent(ctx.operation_id)

        elif result == LockResult.BLOCKED:
            # Add to dependency graph
            self.dependency_graph.add_dependency(
                waiter=ctx.operation_id,
                blocking=lock.owner,
                resource=resource_id,
            )

        elif result == LockResult.PREEMPTED:
            ctx.add_acquired_resource(lock)
            # Clear old dependencies
            self.dependency_graph.remove_all_for_agent(ctx.operation_id)

        return result

    def release_resource(self, ctx: OperationContext, resource_id: str) -> bool:
        """Release a resource."""
        if resource_id not in ctx.acquired_resources:
            return False

        lock = ctx.acquired_resources[resource_id]
        released = lock.release(owner=ctx.operation_id)

        if released:
            del ctx.acquired_resources[resource_id]
            self.dependency_graph.remove_all_for_agent(ctx.operation_id)

        return released

    def release_all_resources(self, ctx: OperationContext) -> None:
        """Release all resources held by an operation."""
        for resource_id in list(ctx.acquired_resources.keys()):
            self.release_resource(ctx, resource_id)

    def check_deadlock(self) -> Optional[DeadlockInfo]:
        """Check for deadlocks in current operations."""
        return self.dependency_graph.detect_cycle()

    def complete_operation(self, ctx: OperationContext) -> OperationResult:
        """
        Complete an operation (successful).

        Releases all resources and cleans up.
        """
        self.release_all_resources(ctx)
        ctx.enter_phase(Phase.G0)

        if ctx.operation_id in self.active_operations:
            del self.active_operations[ctx.operation_id]

        duration = datetime.utcnow() - ctx.created_at

        return OperationResult(
            operation_id=ctx.operation_id,
            agent_id=ctx.agent_id,
            success=True,
            result=ctx.result,
            duration=duration,
        )

    def abort_operation(
        self,
        ctx: OperationContext,
        reason: str,
    ) -> OperationResult:
        """
        Abort an operation.

        Releases all resources and cleans up.
        """
        self.release_all_resources(ctx)
        ctx.enter_phase(Phase.G0)

        if ctx.operation_id in self.active_operations:
            del self.active_operations[ctx.operation_id]

        duration = datetime.utcnow() - ctx.created_at

        return OperationResult(
            operation_id=ctx.operation_id,
            agent_id=ctx.agent_id,
            success=False,
            error=reason,
            duration=duration,
        )

    def stats(self) -> dict:
        """Return controller statistics."""
        phase_counts = {}
        for ctx in self.active_operations.values():
            phase = ctx.phase.value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        return {
            "active_operations": len(self.active_operations),
            "registered_resources": len(self.resources),
            "phase_distribution": phase_counts,
            "pending_deadlocks": self.dependency_graph.detect_cycle() is not None,
        }
