"""Priority Inheritance to prevent priority inversion."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .types import ResourceLock, DependencyGraph
from .controller import CellCycleController, OperationContext


@dataclass
class PriorityBoost:
    """Record of a priority boost."""

    operation_id: str
    original_priority: int
    boosted_priority: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PriorityInheritance:
    """
    Manages priority inheritance to prevent priority inversion.

    When a high-priority operation is blocked by a low-priority
    holder, the holder temporarily inherits the waiter's priority.
    """

    # Active boosts: operation_id -> PriorityBoost
    active_boosts: dict[str, PriorityBoost] = field(default_factory=dict)

    # Statistics
    total_boosts: int = 0

    def check_and_boost(self, controller: CellCycleController) -> list[PriorityBoost]:
        """
        Check for priority inversions and apply boosts.

        Returns list of new boosts applied.
        """
        new_boosts = []
        graph = controller.dependency_graph

        # For each waiter, boost the blocking chain
        for waiter_id in list(graph.edges.keys()):
            waiter = controller.active_operations.get(waiter_id)
            if not waiter:
                continue

            # Get chain of blocking operations
            chain = graph.get_blocking_chain(waiter_id)

            # Propagate priority through chain
            max_priority = waiter.priority

            for op_id in chain[1:]:  # Skip waiter itself
                holder = controller.active_operations.get(op_id)
                if not holder:
                    continue

                # Check if boost needed
                if holder.priority < max_priority:
                    # Record original if not already boosted
                    if op_id not in self.active_boosts:
                        original = holder.priority
                    else:
                        original = self.active_boosts[op_id].original_priority

                    # Apply boost
                    holder.priority = max_priority

                    boost = PriorityBoost(
                        operation_id=op_id,
                        original_priority=original,
                        boosted_priority=max_priority,
                        reason=f"blocked by {waiter_id} (pri={waiter.priority})",
                    )

                    self.active_boosts[op_id] = boost
                    self.total_boosts += 1
                    new_boosts.append(boost)

                # Continue propagation with max of holder's priority
                max_priority = max(max_priority, holder.priority)

        return new_boosts

    def restore_priority(self, ctx: OperationContext) -> Optional[int]:
        """
        Restore original priority after boost.

        Returns original priority if restored, None if not boosted.
        """
        if ctx.operation_id not in self.active_boosts:
            return None

        boost = self.active_boosts.pop(ctx.operation_id)
        original = boost.original_priority
        ctx.priority = original
        return original

    def get_boost(self, operation_id: str) -> Optional[PriorityBoost]:
        """Get active boost for an operation."""
        return self.active_boosts.get(operation_id)

    def is_boosted(self, operation_id: str) -> bool:
        """Check if operation is currently boosted."""
        return operation_id in self.active_boosts

    def clear_all(self, controller: CellCycleController) -> int:
        """
        Clear all boosts and restore original priorities.

        Returns count of boosts cleared.
        """
        count = 0
        for op_id, boost in list(self.active_boosts.items()):
            ctx = controller.active_operations.get(op_id)
            if ctx:
                ctx.priority = boost.original_priority
            count += 1

        self.active_boosts.clear()
        return count

    def stats(self) -> dict:
        """Return priority inheritance statistics."""
        return {
            "total_boosts": self.total_boosts,
            "active_boosts": len(self.active_boosts),
            "operations_boosted": list(self.active_boosts.keys()),
        }
