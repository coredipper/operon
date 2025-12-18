"""Regulatory T-Cell - tolerance and suppression."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from .types import ThreatLevel, ResponseAction
from .tcell import ImmuneResponse


@dataclass
class SuppressionRule:
    """
    Rule for suppressing immune responses.

    Biological parallel: Mechanisms by which Tregs suppress
    T-cell activation in specific contexts.
    """

    name: str
    condition: Callable[[ImmuneResponse, "ToleranceRecord"], bool]
    max_severity: ThreatLevel = ThreatLevel.CONFIRMED
    duration: Optional[timedelta] = None  # None = permanent rule

    def can_suppress(self, response: ImmuneResponse) -> bool:
        """Check if rule can suppress this threat level."""
        severity_order = [
            ThreatLevel.NONE,
            ThreatLevel.SUSPICIOUS,
            ThreatLevel.CONFIRMED,
            ThreatLevel.CRITICAL,
        ]
        return (
            severity_order.index(response.threat_level) <=
            severity_order.index(self.max_severity)
        )


@dataclass
class ToleranceRecord:
    """
    Tolerance tracking for a single agent.

    Tracks inspection history and temporary states
    that affect suppression decisions.
    """

    agent_id: str

    # Inspection history
    clean_inspections: int = 0  # Consecutive clean inspections
    total_inspections: int = 0

    # Temporary states
    last_update: Optional[datetime] = None
    update_tolerance_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))

    # Custom tolerances
    tolerated_violations: set[str] = field(default_factory=set)

    @property
    def recent_update(self) -> bool:
        """Check if agent was recently updated."""
        if self.last_update is None:
            return False
        elapsed = datetime.utcnow() - self.last_update
        return elapsed < self.update_tolerance_duration

    def record_inspection(self, clean: bool) -> None:
        """Record an inspection result."""
        self.total_inspections += 1
        if clean:
            self.clean_inspections += 1
        else:
            self.clean_inspections = 0  # Reset streak

    def mark_updated(self) -> None:
        """Mark agent as recently updated."""
        self.last_update = datetime.utcnow()

    def is_stable(self, threshold: int) -> bool:
        """Check if agent has stable history."""
        return self.clean_inspections >= threshold

    def add_tolerated_violation(self, pattern: str) -> None:
        """Add a known acceptable violation pattern."""
        self.tolerated_violations.add(pattern)


@dataclass
class SuppressionResult:
    """Result of Treg evaluation."""

    suppressed: bool
    original_action: ResponseAction
    modified_action: ResponseAction
    suppression_reason: Optional[str] = None


@dataclass
class RegulatoryTCell:
    """
    Suppresses overactive immune responses.

    Biological parallel: Regulatory T-cells (Tregs) that
    prevent autoimmunity by suppressing T-cell activation.
    """

    rules: list[SuppressionRule] = field(default_factory=list)
    stability_threshold: int = 100  # Clean inspections for auto-tolerance

    # Agent records
    records: dict[str, ToleranceRecord] = field(default_factory=dict)

    def register_agent(self, agent_id: str) -> ToleranceRecord:
        """Register a new agent for tolerance tracking."""
        record = ToleranceRecord(agent_id=agent_id)
        self.records[agent_id] = record
        return record

    def get_record(self, agent_id: str) -> Optional[ToleranceRecord]:
        """Get tolerance record for agent."""
        return self.records.get(agent_id)

    def evaluate(
        self,
        response: ImmuneResponse,
        record: ToleranceRecord,
    ) -> SuppressionResult:
        """
        Evaluate whether to suppress an immune response.

        Returns modified action if suppressed.
        """
        original_action = response.action

        # Never suppress CRITICAL or SHUTDOWN
        if response.threat_level == ThreatLevel.CRITICAL:
            return SuppressionResult(
                suppressed=False,
                original_action=original_action,
                modified_action=original_action,
            )

        # Check stability-based auto-tolerance
        if record.is_stable(self.stability_threshold):
            if response.threat_level == ThreatLevel.SUSPICIOUS:
                return SuppressionResult(
                    suppressed=True,
                    original_action=original_action,
                    modified_action=ResponseAction.IGNORE,
                    suppression_reason="stable_agent",
                )

        # Check suppression rules
        for rule in self.rules:
            if rule.can_suppress(response) and rule.condition(response, record):
                # Downgrade action
                modified = self._downgrade_action(response.action)
                return SuppressionResult(
                    suppressed=True,
                    original_action=original_action,
                    modified_action=modified,
                    suppression_reason=rule.name,
                )

        # No suppression
        return SuppressionResult(
            suppressed=False,
            original_action=original_action,
            modified_action=original_action,
        )

    def _downgrade_action(self, action: ResponseAction) -> ResponseAction:
        """Downgrade an action by one level."""
        downgrades = {
            ResponseAction.SHUTDOWN: ResponseAction.ISOLATE,
            ResponseAction.ISOLATE: ResponseAction.MONITOR,
            ResponseAction.MONITOR: ResponseAction.IGNORE,
            ResponseAction.ALERT: ResponseAction.MONITOR,
            ResponseAction.IGNORE: ResponseAction.IGNORE,
        }
        return downgrades.get(action, ResponseAction.MONITOR)
