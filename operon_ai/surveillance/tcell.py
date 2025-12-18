"""T-Cell - two-signal surveillance responder."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .types import (
    Signal1, Signal2, ThreatLevel, ResponseAction,
    MHCPeptide, ActivationState,
)
from .thymus import BaselineProfile


@dataclass
class ImmuneResponse:
    """Response from T-cell inspection."""

    agent_id: str
    threat_level: ThreatLevel
    action: ResponseAction
    signal1: Signal1
    signal2: Signal2
    violations: list[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_anergic: bool = False


@dataclass
class TCell:
    """
    Two-signal surveillance responder.

    Biological parallel: T-cells that require both MHC recognition
    (Signal 1) and co-stimulation (Signal 2) before activation.
    This prevents autoimmunity (false positives).
    """

    profile: BaselineProfile
    repeated_anomaly_threshold: int = 3
    anergy_threshold: int = 5

    state: ActivationState = field(init=False)
    anomaly_count: int = 0
    anergy_count: int = 0
    manual_flag: Optional[str] = None

    def __post_init__(self):
        self.state = ActivationState(
            agent_id=self.profile.agent_id,
            anergy_threshold=self.anergy_threshold,
        )

    @property
    def is_anergic(self) -> bool:
        """Is the T-cell desensitized from too many false alarms?"""
        return self.anergy_count >= self.anergy_threshold

    def inspect(self, peptide: MHCPeptide) -> ImmuneResponse:
        """
        Inspect peptide and determine response.

        Two-signal activation:
        - Signal 1: Baseline violations (anomaly detected)
        - Signal 2: Canary failure, repeated anomaly, cross-validation, or manual flag
        """
        # Check for anergy first
        if self.is_anergic:
            return ImmuneResponse(
                agent_id=self.profile.agent_id,
                threat_level=ThreatLevel.NONE,
                action=ResponseAction.IGNORE,
                signal1=Signal1.UNKNOWN,
                signal2=Signal2.NONE,
                violations=[],
                is_anergic=True,
            )

        # Check baseline for violations (Signal 1)
        violations = self.profile.check(peptide)
        signal1 = Signal1.NON_SELF if violations else Signal1.SELF

        # Check for Signal 2 sources
        signal2 = Signal2.NONE

        # Manual flag
        if self.manual_flag:
            signal2 = Signal2.MANUAL_FLAG

        # Canary failure
        if (peptide.canary_accuracy is not None and
            peptide.canary_accuracy < self.profile.canary_accuracy_min):
            signal2 = Signal2.CANARY_FAILED

        # Track repeated anomalies
        if signal1 == Signal1.NON_SELF:
            self.anomaly_count += 1
            if self.anomaly_count >= self.repeated_anomaly_threshold:
                signal2 = Signal2.REPEATED_ANOMALY
        else:
            self.anomaly_count = 0

        # Update state
        self.state.signal1 = signal1
        self.state.signal1_violations = violations
        self.state.signal2 = signal2

        # Determine threat level and action
        threat_level, action = self._determine_response(
            signal1, signal2, len(violations), peptide,
        )

        return ImmuneResponse(
            agent_id=self.profile.agent_id,
            threat_level=threat_level,
            action=action,
            signal1=signal1,
            signal2=signal2,
            violations=violations,
        )

    def _determine_response(
        self,
        signal1: Signal1,
        signal2: Signal2,
        violation_count: int,
        peptide: MHCPeptide,
    ) -> tuple[ThreatLevel, ResponseAction]:
        """Determine threat level and recommended action."""

        # No anomaly - all clear
        if signal1 == Signal1.SELF:
            return ThreatLevel.NONE, ResponseAction.IGNORE

        # Anomaly without confirmation - suspicious, monitor
        if signal2 == Signal2.NONE:
            return ThreatLevel.SUSPICIOUS, ResponseAction.MONITOR

        # Both signals - activated
        # Severity based on violation count and canary accuracy
        is_critical = (
            violation_count >= 3 or
            (peptide.canary_accuracy is not None and peptide.canary_accuracy < 0.5)
        )

        if is_critical:
            return ThreatLevel.CRITICAL, ResponseAction.SHUTDOWN
        else:
            return ThreatLevel.CONFIRMED, ResponseAction.ISOLATE

    def flag_manually(self, reason: str) -> None:
        """Manually flag agent for Signal 2."""
        self.manual_flag = reason

    def reset(self) -> None:
        """Reset T-cell state (e.g., after response is handled)."""
        self.state = ActivationState(
            agent_id=self.profile.agent_id,
            anergy_threshold=self.anergy_threshold,
        )
        self.anomaly_count = 0
        self.manual_flag = None

    def reset_without_confirmation(self) -> None:
        """
        Reset after Signal 1 without Signal 2 (false alarm).

        Tracks toward anergy (desensitization).
        """
        if self.state.signal1 == Signal1.NON_SELF and self.state.signal2 == Signal2.NONE:
            self.anergy_count += 1

        self.state = ActivationState(
            agent_id=self.profile.agent_id,
            anergy_threshold=self.anergy_threshold,
        )
        self.anomaly_count = 0
