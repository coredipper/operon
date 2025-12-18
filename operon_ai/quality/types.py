"""Core types for the quality control system."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar, Optional

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


class PoolExhaustionStrategy(Enum):
    """Strategy when ubiquitin pool is exhausted."""
    BLOCK = "block"           # Refuse to allocate
    PASSTHROUGH = "passthrough"  # Create without pool tracking
    RECYCLE_OLDEST = "recycle"   # Force-recycle oldest tags


@dataclass
class UbiquitinPool:
    """Manages ubiquitin tag allocation and recycling."""

    capacity: int = 1000
    available: int = field(init=False)
    exhaustion_strategy: PoolExhaustionStrategy = PoolExhaustionStrategy.BLOCK

    # Tracking for RECYCLE_OLDEST
    active_tags: list[tuple[datetime, UbiquitinTag]] = field(default_factory=list)

    # Metrics
    allocated_total: int = 0
    recycled_total: int = 0
    exhaustion_events: int = 0

    def __post_init__(self):
        self.available = self.capacity

    def allocate(
        self,
        origin: str,
        confidence: float = 1.0,
        degron: DegronType = DegronType.NORMAL,
        integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED,
    ) -> Optional[UbiquitinTag]:
        """Allocate a new tag from the pool."""

        if self.available < 1:
            self.exhaustion_events += 1

            if self.exhaustion_strategy == PoolExhaustionStrategy.BLOCK:
                return None

            elif self.exhaustion_strategy == PoolExhaustionStrategy.RECYCLE_OLDEST:
                if not self._force_recycle():
                    return None

            elif self.exhaustion_strategy == PoolExhaustionStrategy.PASSTHROUGH:
                # Create tag without consuming from pool
                return self._create_tag(origin, confidence, degron, integrity)

        self.available -= 1
        self.allocated_total += 1
        tag = self._create_tag(origin, confidence, degron, integrity)
        self.active_tags.append((datetime.utcnow(), tag))
        return tag

    def _create_tag(
        self,
        origin: str,
        confidence: float,
        degron: DegronType,
        integrity: IntegrityLabel,
    ) -> UbiquitinTag:
        """Create a new tag instance."""
        return UbiquitinTag(
            confidence=confidence,
            origin=origin,
            generation=0,
            degron=degron,
            integrity=integrity,
        )

    def recycle(self, tag: UbiquitinTag) -> None:
        """Return a tag to the pool."""
        self.available = min(self.capacity, self.available + tag.chain_length)
        self.recycled_total += tag.chain_length
        # Remove from active tracking
        self.active_tags = [
            (ts, t) for ts, t in self.active_tags
            if not (t.timestamp == tag.timestamp and t.origin == tag.origin)
        ]

    def _force_recycle(self) -> bool:
        """Force-recycle oldest tag. Returns True if successful."""
        if not self.active_tags:
            return False

        # Sort by timestamp, recycle oldest
        self.active_tags.sort(key=lambda x: x[0])
        _, oldest = self.active_tags.pop(0)
        self.available += oldest.chain_length
        self.recycled_total += oldest.chain_length
        return True

    def status(self) -> dict:
        """Return pool status metrics."""
        utilization = 1 - (self.available / self.capacity) if self.capacity > 0 else 0
        return {
            "available": self.available,
            "capacity": self.capacity,
            "utilization": f"{utilization:.1%}",
            "allocated_total": self.allocated_total,
            "recycled_total": self.recycled_total,
            "exhaustion_events": self.exhaustion_events,
            "active_tags": len(self.active_tags),
        }
