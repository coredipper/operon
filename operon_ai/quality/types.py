"""Core types for the quality control system."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from operon_ai.core.types import IntegrityLabel

T = TypeVar("T")


class ChainType(Enum):
    """Ubiquitin chain types with different signals."""
    K48 = "k48"      # Standard degradation signal
    K63 = "k63"      # Non-degradation signaling
    K11 = "k11"      # Time-sensitive operations
    MONO = "mono"    # Minimal modification


class DegronType(Enum):
    """Data-specific degradation rates."""
    STABLE = "stable"       # Long half-life (config, validated refs)
    NORMAL = "normal"       # Standard agent outputs
    UNSTABLE = "unstable"   # Transient state, cache
    IMMEDIATE = "immediate" # Sensitive data, PII


class DegradationResult(Enum):
    """Result of proteasome inspection."""
    PASSED = "passed"
    REPAIRED = "repaired"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    QUEUED_REVIEW = "queued"
    RESCUED = "rescued"


@dataclass(frozen=True)
class UbiquitinTag:
    """Provenance tag attached to data flowing through the system."""

    confidence: float
    origin: str
    generation: int
    chain_type: ChainType = ChainType.K48
    degron: DegronType = DegronType.NORMAL
    chain_length: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED

    def with_confidence(self, new_confidence: float) -> UbiquitinTag:
        """Return new tag with updated confidence (clamped to 0-1)."""
        clamped = max(0.0, min(1.0, new_confidence))
        return UbiquitinTag(
            confidence=clamped,
            origin=self.origin,
            generation=self.generation,
            chain_type=self.chain_type,
            degron=self.degron,
            chain_length=self.chain_length,
            timestamp=self.timestamp,
            integrity=self.integrity,
        )

    def restore_confidence(self, amount: float) -> UbiquitinTag:
        """Return new tag with confidence increased by amount."""
        return self.with_confidence(self.confidence + amount)

    def reduce_confidence(self, factor: float) -> UbiquitinTag:
        """Return new tag with confidence multiplied by factor."""
        return self.with_confidence(self.confidence * factor)

    def increment_generation(self) -> UbiquitinTag:
        """Return new tag with generation incremented."""
        return UbiquitinTag(
            confidence=self.confidence,
            origin=self.origin,
            generation=self.generation + 1,
            chain_type=self.chain_type,
            degron=self.degron,
            chain_length=self.chain_length,
            timestamp=self.timestamp,
            integrity=self.integrity,
        )

    def with_integrity(self, integrity: IntegrityLabel) -> UbiquitinTag:
        """Return new tag with updated integrity label."""
        return UbiquitinTag(
            confidence=self.confidence,
            origin=self.origin,
            generation=self.generation,
            chain_type=self.chain_type,
            degron=self.degron,
            chain_length=self.chain_length,
            timestamp=self.timestamp,
            integrity=integrity,
        )

    def effective_threshold(self, base: float) -> float:
        """Calculate degron-adjusted threshold."""
        multipliers = {
            DegronType.STABLE: 0.5,
            DegronType.NORMAL: 1.0,
            DegronType.UNSTABLE: 1.5,
            DegronType.IMMEDIATE: 3.0,
        }
        return base * multipliers[self.degron]


@dataclass
class TaggedData(Generic[T]):
    """Data paired with its provenance tag."""

    data: T
    tag: UbiquitinTag

    def map(self, func: Callable[[T], T]) -> TaggedData[T]:
        """Apply transformation preserving tag."""
        return TaggedData(data=func(self.data), tag=self.tag)

    def with_tag(self, tag: UbiquitinTag) -> TaggedData[T]:
        """Return new TaggedData with different tag."""
        return TaggedData(data=self.data, tag=tag)

    def clone_for_fanout(self) -> TaggedData[T]:
        """Create independent copy for branching pipelines."""
        new_tag = UbiquitinTag(
            confidence=self.tag.confidence,
            origin=self.tag.origin,
            generation=self.tag.generation,
            chain_type=self.tag.chain_type,
            degron=self.tag.degron,
            chain_length=self.tag.chain_length,
            timestamp=self.tag.timestamp,
            integrity=self.tag.integrity,
        )
        return TaggedData(data=self.data, tag=new_tag)
