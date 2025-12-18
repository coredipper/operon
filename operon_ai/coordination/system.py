"""Integrated Coordination System organelle."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Optional

from .types import Phase, CheckpointResult, ResourceLock, LockResult
from .controller import (
    CellCycleController, OperationContext, OperationResult, Checkpoint,
)
from .watchdog import Watchdog, ApoptosisEvent, ApoptosisReason
from .priority import PriorityInheritance, PriorityBoost


@dataclass
class CoordinationResult:
    """Result of coordinated operation execution."""

    operation_id: str
    success: bool
    phase_reached: Phase
    result: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class CoordinationSystem:
    """
    Integrated coordination system organelle.

    Combines CellCycleController, Watchdog, and PriorityInheritance
    into a unified interface for operation coordination.
    """

    # Timeouts (passed to watchdog)
    max_operation_time: Optional[timedelta] = None
    starvation_timeout: Optional[timedelta] = None
    progress_timeout: Optional[timedelta] = None

    # Components (initialized in __post_init__)
    controller: CellCycleController = field(default_factory=CellCycleController)
    watchdog: Watchdog = field(init=False)
    priority_manager: PriorityInheritance = field(default_factory=PriorityInheritance)

    def __post_init__(self):
        self.watchdog = Watchdog(
            max_operation_time=self.max_operation_time,
            starvation_timeout=self.starvation_timeout,
            progress_timeout=self.progress_timeout,
        )

    def register_resource(self, resource_id: str, allow_preemption: bool = False) -> None:
        """Register a resource for coordination."""
        lock = ResourceLock(resource_id=resource_id, allow_preemption=allow_preemption)
        self.controller.register_resource(lock)

    def start_operation(
        self,
        operation_id: str,
        agent_id: str,
        priority: int = 0,
    ) -> OperationContext:
        """Start a new operation."""
        return self.controller.start_operation(operation_id, agent_id, priority)

    def execute_operation(
        self,
        operation_id: str,
        agent_id: str,
        work_fn: Callable[[], Any],
        resources: Optional[list[str]] = None,
        validate_fn: Optional[Callable[[Any], bool]] = None,
        priority: int = 0,
    ) -> CoordinationResult:
        """
        Execute a complete operation through all phases.

        G0 -> G1 (acquire resources) -> S (execute work) -> G2 (validate) -> M (commit)
        """
        import time
        start_time = time.time()

        # Start operation
        ctx = self.controller.start_operation(operation_id, agent_id, priority)

        try:
            # G0 -> G1
            self.controller.advance(ctx)

            # Acquire resources in G1
            resources = resources or []
            for resource_id in resources:
                result = self.controller.acquire_resource(ctx, resource_id)
                if result == LockResult.BLOCKED:
                    # Could wait or fail - for now, fail
                    raise ResourceError(f"Blocked on resource {resource_id}")

            ctx.resources_acquired = True

            # G1 -> S
            checkpoint_result = self.controller.advance(ctx)
            if checkpoint_result != CheckpointResult.PASSED:
                raise CheckpointError(f"G1 checkpoint failed: {checkpoint_result}")

            # Execute work in S phase
            try:
                result = work_fn()
                ctx.set_result(result)
                ctx.execution_complete = True
            except Exception as e:
                raise WorkError(f"Work failed: {e}")

            # S -> G2
            checkpoint_result = self.controller.advance(ctx)
            if checkpoint_result != CheckpointResult.PASSED:
                raise CheckpointError(f"S checkpoint failed: {checkpoint_result}")

            # Validate in G2 phase
            if validate_fn:
                if not validate_fn(result):
                    raise ValidationError("Validation failed")

            ctx.validation_passed = True

            # G2 -> M
            checkpoint_result = self.controller.advance(ctx)
            if checkpoint_result != CheckpointResult.PASSED:
                raise CheckpointError(f"G2 checkpoint failed: {checkpoint_result}")

            # Complete operation
            op_result = self.controller.complete_operation(ctx)

            duration_ms = (time.time() - start_time) * 1000

            return CoordinationResult(
                operation_id=operation_id,
                success=True,
                phase_reached=Phase.M,
                result=result,
                duration_ms=duration_ms,
            )

        except Exception as e:
            # Abort on any error
            self.controller.abort_operation(ctx, reason=str(e))

            duration_ms = (time.time() - start_time) * 1000

            return CoordinationResult(
                operation_id=operation_id,
                success=False,
                phase_reached=ctx.phase,
                error=str(e),
                duration_ms=duration_ms,
            )

    def run_maintenance(self) -> dict:
        """
        Run periodic maintenance tasks.

        - Check for timeouts (watchdog)
        - Apply priority inheritance
        - Detect and handle deadlocks

        Returns dict with events from each subsystem.
        """
        # Priority inheritance
        boosts = self.priority_manager.check_and_boost(self.controller)

        # Watchdog checks and executions
        apoptosis_events = self.watchdog.execute(self.controller)

        # Restore priorities for completed operations
        # (Already handled when operations complete)

        return {
            "priority_boosts": boosts,
            "apoptosis": apoptosis_events,
        }

    def kill_operation(self, operation_id: str, reason: str = "manual") -> Optional[ApoptosisEvent]:
        """Manually kill an operation."""
        return self.watchdog.manual_kill(self.controller, operation_id, reason)

    def health(self) -> dict:
        """Return system health status."""
        return {
            "active_operations": len(self.controller.active_operations),
            "registered_resources": len(self.controller.resources),
            "controller": self.controller.stats(),
            "watchdog": self.watchdog.stats(),
            "priority": self.priority_manager.stats(),
        }

    def shutdown(self) -> None:
        """Gracefully shutdown all operations."""
        for op_id in list(self.controller.active_operations.keys()):
            ctx = self.controller.active_operations.get(op_id)
            if ctx:
                self.controller.abort_operation(ctx, reason="system shutdown")

        self.priority_manager.clear_all(self.controller)


# Custom exceptions for coordination
class CoordinationError(Exception):
    """Base exception for coordination errors."""
    pass


class ResourceError(CoordinationError):
    """Error acquiring resources."""
    pass


class CheckpointError(CoordinationError):
    """Checkpoint failed."""
    pass


class WorkError(CoordinationError):
    """Work function failed."""
    pass


class ValidationError(CoordinationError):
    """Validation failed."""
    pass
