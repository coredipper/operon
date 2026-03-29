"""HeartbeatDaemon — periodic idle-time consolidation for WatcherComponent.

Extends WatcherComponent to track runs between heartbeats and trigger
SleepConsolidation when enough runs accumulate and sufficient time has
elapsed since the last consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .watcher import WatcherComponent


@dataclass
class HeartbeatDaemon(WatcherComponent):
    """WatcherComponent with periodic idle-time consolidation.

    Tracks runs since last consolidation. When heartbeat() is called
    (typically between runs), triggers SleepConsolidation if conditions
    are met.
    """

    consolidation: Any = None  # SleepConsolidation | None
    heartbeat_interval_s: float = 300.0
    min_runs_before_consolidate: int = 3
    _run_count_since_consolidate: int = field(default=0, init=False)
    _last_consolidation: datetime | None = field(default=None, init=False)
    _heartbeat_results: list[dict] = field(default_factory=list, init=False)

    def on_run_start(self, task: str, shared_state: dict) -> None:
        """Increment run counter and delegate to parent."""
        super().on_run_start(task, shared_state)
        self._run_count_since_consolidate += 1

    def heartbeat(self) -> dict[str, Any]:
        """Check if consolidation is due and trigger if conditions met.

        Returns dict with keys: triggered (bool), reason (str),
        result (ConsolidationResult | None), runs_since_last (int)
        """
        now = datetime.now(UTC)
        result_dict: dict[str, Any] = {
            "triggered": False,
            "reason": "",
            "result": None,
            "runs_since_last": self._run_count_since_consolidate,
        }

        if self.consolidation is None:
            result_dict["reason"] = "no consolidation configured"
            return result_dict

        if self._run_count_since_consolidate < self.min_runs_before_consolidate:
            result_dict["reason"] = (
                f"only {self._run_count_since_consolidate} runs "
                f"(need {self.min_runs_before_consolidate})"
            )
            return result_dict

        if self._last_consolidation is not None:
            elapsed = (now - self._last_consolidation).total_seconds()
            if elapsed < self.heartbeat_interval_s:
                result_dict["reason"] = (
                    f"too soon ({elapsed:.0f}s < {self.heartbeat_interval_s:.0f}s)"
                )
                return result_dict

        # Trigger consolidation
        consolidation_result = self.consolidation.consolidate()
        self._run_count_since_consolidate = 0
        self._last_consolidation = now
        result_dict["triggered"] = True
        result_dict["reason"] = "conditions met"
        result_dict["result"] = consolidation_result
        self._heartbeat_results.append(result_dict)
        return result_dict

    def summary(self) -> dict[str, Any]:
        """Extend watcher summary with heartbeat stats."""
        base = super().summary()
        base["heartbeat"] = {
            "runs_since_consolidate": self._run_count_since_consolidate,
            "last_consolidation": (
                self._last_consolidation.isoformat()
                if self._last_consolidation
                else None
            ),
            "total_heartbeats_triggered": len(self._heartbeat_results),
        }
        return base
