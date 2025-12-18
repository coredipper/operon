# operon_ai/surveillance/types.py
"""Core types for the surveillance system."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import statistics


class Signal1(Enum):
    """MHC recognition results."""
    SELF = "self"
    NON_SELF = "non_self"
    UNKNOWN = "unknown"


class Signal2(Enum):
    """Co-stimulatory confirmation."""
    NONE = "none"
    CANARY_FAILED = "canary"
    CROSS_VALIDATED = "cross"
    REPEATED_ANOMALY = "repeat"
    MANUAL_FLAG = "manual"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    NONE = "none"
    SUSPICIOUS = "suspicious"
    CONFIRMED = "confirmed"
    CRITICAL = "critical"


class ResponseAction(Enum):
    """Recommended response actions."""
    IGNORE = "ignore"
    MONITOR = "monitor"
    ISOLATE = "isolate"
    SHUTDOWN = "shutdown"
    ALERT = "alert"


@dataclass(frozen=True)
class MHCPeptide:
    """
    Behavioral fingerprint displayed by an agent.

    Like MHC presenting protein fragments, this presents
    statistical signatures of agent behavior for inspection.
    """

    agent_id: str
    timestamp: datetime

    # Output characteristics
    output_length_mean: float
    output_length_std: float
    response_time_mean: float
    response_time_std: float

    # Semantic markers
    vocabulary_hash: str
    structure_hash: str
    confidence_mean: float
    confidence_std: float

    # Error patterns
    error_rate: float
    error_types: tuple[str, ...]

    # Canary results
    canary_accuracy: Optional[float] = None

    def similarity(self, other: MHCPeptide) -> float:
        """Calculate similarity score (0.0 = different, 1.0 = identical)."""
        if self.agent_id != other.agent_id:
            return 0.0

        scores = []

        def compare_stat(a_mean, a_std, b_mean, b_std) -> float:
            if a_std == 0 and b_std == 0:
                return 1.0 if a_mean == b_mean else 0.0
            diff = abs(a_mean - b_mean)
            tolerance = max(a_std, b_std, 0.01) * 2
            return max(0.0, 1.0 - (diff / tolerance))

        scores.append(compare_stat(
            self.output_length_mean, self.output_length_std,
            other.output_length_mean, other.output_length_std
        ))
        scores.append(compare_stat(
            self.response_time_mean, self.response_time_std,
            other.response_time_mean, other.response_time_std
        ))
        scores.append(compare_stat(
            self.confidence_mean, self.confidence_std,
            other.confidence_mean, other.confidence_std
        ))

        # Hash matches (binary with partial credit)
        scores.append(1.0 if self.vocabulary_hash == other.vocabulary_hash else 0.3)
        scores.append(1.0 if self.structure_hash == other.structure_hash else 0.3)

        # Error rate comparison
        scores.append(1.0 - min(1.0, abs(self.error_rate - other.error_rate) * 10))

        return statistics.mean(scores)


@dataclass
class ActivationState:
    """Current activation state of a T-cell watching an agent."""

    agent_id: str
    signal1: Signal1 = Signal1.SELF
    signal1_violations: list[str] = field(default_factory=list)
    signal2: Signal2 = Signal2.NONE
    signal2_evidence: Optional[str] = None

    anomaly_without_confirmation_count: int = 0
    anergy_threshold: int = 3

    @property
    def is_activated(self) -> bool:
        """Activation requires both signals."""
        return self.signal1 == Signal1.NON_SELF and self.signal2 != Signal2.NONE
