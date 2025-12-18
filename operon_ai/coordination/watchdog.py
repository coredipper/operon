"""Watchdog for operation termination (apoptosis)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from .types import Phase, DeadlockInfo
from .controller import CellCycleController, OperationContext


class ApoptosisReason(Enum):
    """Reason for operation termination."""
    TIMEOUT = "timeout"
    STARVATION = "starvation"
    NO_PROGRESS = "no_progress"
    DEADLOCK = "deadlock"
    MANUAL = "manual"


@dataclass
class ApoptosisEvent:
    """Record of an operation termination."""

    operation_id: str
    agent_id: str
    reason: ApoptosisReason
    details: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Watchdog:
    """
    Monitors operations and triggers termination (apoptosis).

    Biological parallel: Apoptosis - programmed cell death
    that eliminates damaged or stuck cells.
    """

    # Timeout thresholds
    max_operation_time: Optional[timedelta] = None
    starvation_timeout: Optional[timedelta] = None
    progress_timeout: Optional[timedelta] = None

    # Deadlock handling
    deadlock_strategy: str = "priority"  # "priority" or "oldest"

    # History
    events: list[ApoptosisEvent] = field(default_factory=list)

    def check(self, controller: CellCycleController) -> list[ApoptosisEvent]:
        """
        Check for operations that should be terminated.

        Returns list of termination events (does not execute).
        """
        events = []
        now = datetime.utcnow()

        for op_id, ctx in list(controller.active_operations.items()):
            # Check exempt status
            if ctx.metadata.get("watchdog_exempt", False):
                continue

            # Check operation timeout
            if self.max_operation_time:
                elapsed = now - ctx.created_at
                if elapsed > self.max_operation_time:
                    events.append(ApoptosisEvent(
                        operation_id=op_id,
                        agent_id=ctx.agent_id,
                        reason=ApoptosisReason.TIMEOUT,
                        details=f"exceeded {self.max_operation_time} total time",
                    ))
                    continue  # Don't double-count

            # Check starvation (waiting in G1 for resources)
            if self.starvation_timeout and ctx.phase == Phase.G1:
                phase_time = now - ctx.phase_entered_at
                if phase_time > self.starvation_timeout:
                    # Check if actually waiting for resources
                    if not ctx.resources_acquired:
                        events.append(ApoptosisEvent(
                            operation_id=op_id,
                            agent_id=ctx.agent_id,
                            reason=ApoptosisReason.STARVATION,
                            details=f"waiting for resources for {phase_time}",
                        ))
                        continue

            # Check no progress (stuck in S phase)
            if self.progress_timeout and ctx.phase == Phase.S:
                phase_time = now - ctx.phase_entered_at
                if phase_time > self.progress_timeout:
                    events.append(ApoptosisEvent(
                        operation_id=op_id,
                        agent_id=ctx.agent_id,
                        reason=ApoptosisReason.NO_PROGRESS,
                        details=f"no progress in S phase for {phase_time}",
                    ))
                    continue

        # Check for deadlocks
        deadlock = controller.check_deadlock()
        if deadlock:
            victim = self._select_deadlock_victim(controller, deadlock)
            if victim and victim not in [e.operation_id for e in events]:
                ctx = controller.active_operations.get(victim)
                if ctx:
                    events.append(ApoptosisEvent(
                        operation_id=victim,
                        agent_id=ctx.agent_id,
                        reason=ApoptosisReason.DEADLOCK,
                        details=f"deadlock cycle: {deadlock.agents}",
                    ))

        return events

    def _select_deadlock_victim(
        self,
        controller: CellCycleController,
        deadlock: DeadlockInfo,
    ) -> Optional[str]:
        """Select which operation to kill to break deadlock."""
        involved = [
            controller.active_operations.get(op_id)
            for op_id in deadlock.agents
            if op_id in controller.active_operations
        ]
        involved = [ctx for ctx in involved if ctx is not None]

        if not involved:
            return None

        if self.deadlock_strategy == "priority":
            # Kill lowest priority
            victim = min(involved, key=lambda ctx: ctx.priority)
        elif self.deadlock_strategy == "oldest":
            # Kill oldest operation
            victim = min(involved, key=lambda ctx: ctx.created_at)
        else:
            victim = involved[0]

        return victim.operation_id

    def execute(self, controller: CellCycleController) -> list[ApoptosisEvent]:
        """
        Check and terminate operations.

        Returns list of termination events.
        """
        events = self.check(controller)

        for event in events:
            ctx = controller.active_operations.get(event.operation_id)
            if ctx:
                controller.abort_operation(ctx, reason=str(event.reason.value))

        self.events.extend(events)
        return events

    def manual_kill(
        self,
        controller: CellCycleController,
        operation_id: str,
        reason: str = "manual termination",
    ) -> Optional[ApoptosisEvent]:
        """Manually terminate an operation."""
        ctx = controller.active_operations.get(operation_id)
        if not ctx:
            return None

        event = ApoptosisEvent(
            operation_id=operation_id,
            agent_id=ctx.agent_id,
            reason=ApoptosisReason.MANUAL,
            details=reason,
        )

        controller.abort_operation(ctx, reason=reason)
        self.events.append(event)
        return event

    def stats(self) -> dict:
        """Return watchdog statistics."""
        reason_counts = {}
        for event in self.events:
            r = event.reason.value
            reason_counts[r] = reason_counts.get(r, 0) + 1

        return {
            "total_terminations": len(self.events),
            "by_reason": reason_counts,
        }
