"""Naive alternative to ATP_Store: flat decrementing counter.

This is the minimal reasonable budget that any production system would
use.  Start at N, subtract costs, return False when empty.  No states,
no currencies, no regeneration, no debt, no scaling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimpleBudget:
    """Flat budget counter — the simplest possible resource manager."""

    budget: int
    _consumed: int = 0
    _failed: int = 0
    _ops: int = 0

    def consume(self, cost: int, operation: str = "", priority: int = 0) -> bool:
        """Deduct cost. Returns False if insufficient funds."""
        self._ops += 1
        if self.budget >= cost:
            self.budget -= cost
            self._consumed += cost
            return True
        self._failed += 1
        return False

    def get_balance(self) -> int:
        return self.budget

    def get_statistics(self) -> dict:
        return {
            "balance": self.budget,
            "consumed": self._consumed,
            "operations": self._ops,
            "failed": self._failed,
        }
