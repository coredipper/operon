# Cell Integrity Systems Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three interconnected biologically-inspired systems for cascade prevention (Quality/Ubiquitin), Byzantine detection (Surveillance/Immune), and deadlock prevention (Coordination/Cell Cycle).

**Architecture:** Three organelle modules under `operon_ai/organelles/` with shared types in `operon_ai/core/`. Each system is self-contained but exposes hooks for cross-system integration. Final `IntegratedCell` class composes all three.

**Tech Stack:** Python 3.11+, Pydantic for validation, dataclasses for immutable types, pytest for testing.

---

## Phase 1: Quality System (Ubiquitin-Proteasome)

### Task 1.1: Core Quality Types

**Files:**
- Create: `operon_ai/quality/__init__.py`
- Create: `operon_ai/quality/types.py`
- Test: `tests/test_quality_types.py`

**Step 1: Write the failing tests**

```python
# tests/test_quality_types.py
"""Tests for quality system types."""
import pytest
from datetime import datetime
from operon_ai.quality.types import (
    ChainType, DegronType, DegradationResult,
    UbiquitinTag, TaggedData,
)
from operon_ai.core.types import IntegrityLabel


class TestChainType:
    def test_chain_types_exist(self):
        assert ChainType.K48.value == "k48"
        assert ChainType.K63.value == "k63"
        assert ChainType.K11.value == "k11"
        assert ChainType.MONO.value == "mono"


class TestDegronType:
    def test_degron_types_exist(self):
        assert DegronType.STABLE.value == "stable"
        assert DegronType.NORMAL.value == "normal"
        assert DegronType.UNSTABLE.value == "unstable"
        assert DegronType.IMMEDIATE.value == "immediate"


class TestDegradationResult:
    def test_results_exist(self):
        assert DegradationResult.PASSED.value == "passed"
        assert DegradationResult.REPAIRED.value == "repaired"
        assert DegradationResult.DEGRADED.value == "degraded"
        assert DegradationResult.BLOCKED.value == "blocked"


class TestUbiquitinTag:
    def test_create_tag(self):
        tag = UbiquitinTag(
            confidence=0.9,
            origin="test_agent",
            generation=0,
        )
        assert tag.confidence == 0.9
        assert tag.origin == "test_agent"
        assert tag.generation == 0
        assert tag.chain_type == ChainType.K48
        assert tag.degron == DegronType.NORMAL

    def test_tag_is_frozen(self):
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        with pytest.raises(Exception):  # FrozenInstanceError
            tag.confidence = 0.5

    def test_with_confidence(self):
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        new_tag = tag.with_confidence(0.5)
        assert new_tag.confidence == 0.5
        assert tag.confidence == 0.9  # Original unchanged

    def test_confidence_clamped(self):
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0)
        assert tag.with_confidence(1.5).confidence == 1.0
        assert tag.with_confidence(-0.5).confidence == 0.0

    def test_reduce_confidence(self):
        tag = UbiquitinTag(confidence=1.0, origin="test", generation=0)
        new_tag = tag.reduce_confidence(0.9)
        assert new_tag.confidence == 0.9

    def test_restore_confidence(self):
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0)
        new_tag = tag.restore_confidence(0.3)
        assert new_tag.confidence == 0.8

    def test_increment_generation(self):
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        new_tag = tag.increment_generation()
        assert new_tag.generation == 1
        assert tag.generation == 0

    def test_effective_threshold_stable(self):
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0, degron=DegronType.STABLE)
        assert tag.effective_threshold(0.4) == 0.2  # 0.4 * 0.5

    def test_effective_threshold_unstable(self):
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0, degron=DegronType.UNSTABLE)
        assert tag.effective_threshold(0.4) == 0.6  # 0.4 * 1.5

    def test_effective_threshold_immediate(self):
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0, degron=DegronType.IMMEDIATE)
        assert tag.effective_threshold(0.4) == 1.2  # 0.4 * 3.0


class TestTaggedData:
    def test_create_tagged_data(self):
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        tagged = TaggedData(data="hello", tag=tag)
        assert tagged.data == "hello"
        assert tagged.tag.confidence == 0.9

    def test_map_preserves_tag(self):
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        tagged = TaggedData(data="hello", tag=tag)
        new_tagged = tagged.map(lambda x: x.upper())
        assert new_tagged.data == "HELLO"
        assert new_tagged.tag.confidence == 0.9

    def test_with_tag(self):
        tag1 = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        tag2 = UbiquitinTag(confidence=0.5, origin="other", generation=1)
        tagged = TaggedData(data="hello", tag=tag1)
        new_tagged = tagged.with_tag(tag2)
        assert new_tagged.tag.confidence == 0.5

    def test_clone_for_fanout(self):
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        tagged = TaggedData(data="hello", tag=tag)
        cloned = tagged.clone_for_fanout()
        assert cloned.data == "hello"
        assert cloned.tag.confidence == 0.9
        assert cloned.tag is not tagged.tag  # Different object
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_quality_types.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'operon_ai.quality'"

**Step 3: Create package init**

```python
# operon_ai/quality/__init__.py
"""Quality control system (Ubiquitin-Proteasome model)."""
from .types import (
    ChainType,
    DegronType,
    DegradationResult,
    UbiquitinTag,
    TaggedData,
)

__all__ = [
    "ChainType",
    "DegronType",
    "DegradationResult",
    "UbiquitinTag",
    "TaggedData",
]
```

**Step 4: Write minimal implementation**

```python
# operon_ai/quality/types.py
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
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_quality_types.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add operon_ai/quality/ tests/test_quality_types.py
git commit -m "feat(quality): add core types for ubiquitin-proteasome system

- ChainType enum (K48, K63, K11, MONO)
- DegronType enum (STABLE, NORMAL, UNSTABLE, IMMEDIATE)
- DegradationResult enum
- UbiquitinTag frozen dataclass with confidence operations
- TaggedData generic wrapper for provenance tracking"
```

---

### Task 1.2: Ubiquitin Pool

**Files:**
- Modify: `operon_ai/quality/types.py`
- Modify: `operon_ai/quality/__init__.py`
- Test: `tests/test_quality_pool.py`

**Step 1: Write the failing tests**

```python
# tests/test_quality_pool.py
"""Tests for ubiquitin pool resource management."""
import pytest
from operon_ai.quality import (
    UbiquitinPool, PoolExhaustionStrategy,
    UbiquitinTag, DegronType,
)
from operon_ai.core.types import IntegrityLabel


class TestPoolExhaustionStrategy:
    def test_strategies_exist(self):
        assert PoolExhaustionStrategy.BLOCK.value == "block"
        assert PoolExhaustionStrategy.PASSTHROUGH.value == "passthrough"
        assert PoolExhaustionStrategy.RECYCLE_OLDEST.value == "recycle"


class TestUbiquitinPool:
    def test_create_pool(self):
        pool = UbiquitinPool(capacity=100)
        assert pool.capacity == 100
        assert pool.available == 100

    def test_allocate_returns_tag(self):
        pool = UbiquitinPool(capacity=100)
        tag = pool.allocate(origin="test_agent", confidence=0.9)
        assert tag is not None
        assert tag.origin == "test_agent"
        assert tag.confidence == 0.9
        assert pool.available == 99

    def test_allocate_with_degron(self):
        pool = UbiquitinPool(capacity=100)
        tag = pool.allocate(
            origin="test",
            confidence=0.9,
            degron=DegronType.UNSTABLE,
        )
        assert tag.degron == DegronType.UNSTABLE

    def test_allocate_exhausted_block(self):
        pool = UbiquitinPool(
            capacity=2,
            exhaustion_strategy=PoolExhaustionStrategy.BLOCK,
        )
        pool.allocate(origin="a")
        pool.allocate(origin="b")
        tag = pool.allocate(origin="c")
        assert tag is None
        assert pool.exhaustion_events == 1

    def test_allocate_exhausted_passthrough(self):
        pool = UbiquitinPool(
            capacity=1,
            exhaustion_strategy=PoolExhaustionStrategy.PASSTHROUGH,
        )
        pool.allocate(origin="a")
        tag = pool.allocate(origin="b")
        assert tag is not None  # Still creates tag
        assert tag.origin == "b"

    def test_allocate_exhausted_recycle_oldest(self):
        pool = UbiquitinPool(
            capacity=2,
            exhaustion_strategy=PoolExhaustionStrategy.RECYCLE_OLDEST,
        )
        pool.allocate(origin="a")
        pool.allocate(origin="b")
        # Pool is full, should recycle oldest
        tag = pool.allocate(origin="c")
        assert tag is not None
        assert tag.origin == "c"

    def test_recycle_returns_to_pool(self):
        pool = UbiquitinPool(capacity=10)
        tag = pool.allocate(origin="test", confidence=0.9)
        assert pool.available == 9
        pool.recycle(tag)
        assert pool.available == 10

    def test_recycle_chain_length(self):
        pool = UbiquitinPool(capacity=10)
        tag = UbiquitinTag(
            confidence=0.9,
            origin="test",
            generation=0,
            chain_length=3,
        )
        pool.available = 5
        pool.recycle(tag)
        assert pool.available == 8  # +3 from chain_length

    def test_recycle_capped_at_capacity(self):
        pool = UbiquitinPool(capacity=10)
        pool.available = 9
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0, chain_length=5)
        pool.recycle(tag)
        assert pool.available == 10  # Capped

    def test_status(self):
        pool = UbiquitinPool(capacity=100)
        pool.allocate(origin="test")
        status = pool.status()
        assert status["available"] == 99
        assert status["capacity"] == 100
        assert "utilization" in status
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_quality_pool.py -v`
Expected: FAIL with "cannot import name 'UbiquitinPool'"

**Step 3: Write minimal implementation**

Add to `operon_ai/quality/types.py`:

```python
# Add to imports at top
from typing import Any, Callable, Generic, TypeVar, Optional

# Add these classes after TaggedData

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
```

**Step 4: Update __init__.py**

```python
# operon_ai/quality/__init__.py
"""Quality control system (Ubiquitin-Proteasome model)."""
from .types import (
    ChainType,
    DegronType,
    DegradationResult,
    PoolExhaustionStrategy,
    UbiquitinTag,
    TaggedData,
    UbiquitinPool,
)

__all__ = [
    "ChainType",
    "DegronType",
    "DegradationResult",
    "PoolExhaustionStrategy",
    "UbiquitinTag",
    "TaggedData",
    "UbiquitinPool",
]
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_quality_pool.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add operon_ai/quality/ tests/test_quality_pool.py
git commit -m "feat(quality): add UbiquitinPool for tag resource management

- PoolExhaustionStrategy enum (BLOCK, PASSTHROUGH, RECYCLE_OLDEST)
- UbiquitinPool with allocation, recycling, and metrics
- Force-recycle oldest when pool exhausted (RECYCLE_OLDEST strategy)"
```

---

### Task 1.3: Proteasome Components (E3Ligase, DUB, ChaperoneRepair)

**Files:**
- Create: `operon_ai/quality/components.py`
- Modify: `operon_ai/quality/__init__.py`
- Test: `tests/test_quality_components.py`

**Step 1: Write the failing tests**

```python
# tests/test_quality_components.py
"""Tests for proteasome components."""
import pytest
from operon_ai.quality import (
    UbiquitinTag, DegronType,
)
from operon_ai.quality.components import (
    ProvenanceContext, E3Ligase, Deubiquitinase, ChaperoneRepair,
)


class TestProvenanceContext:
    def test_create_context(self):
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        ctx = ProvenanceContext(
            tag=tag,
            source_module="agent_a",
            target_module="agent_b",
        )
        assert ctx.source_module == "agent_a"
        assert ctx.target_module == "agent_b"
        assert ctx.source_reliability == 1.0  # Default


class TestE3Ligase:
    def test_create_ligase(self):
        ligase = E3Ligase(
            name="test_ligase",
            active=lambda ctx: True,
            substrate_match=lambda data: True,
            tag_strength=lambda ctx: 0.8,
        )
        assert ligase.name == "test_ligase"

    def test_ligase_conditional_activation(self):
        ligase = E3Ligase(
            name="conditional",
            active=lambda ctx: ctx.source_reliability < 0.5,
            substrate_match=lambda data: True,
            tag_strength=lambda ctx: 0.5,
        )
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)

        ctx_reliable = ProvenanceContext(tag=tag, source_module="a", target_module="b", source_reliability=0.9)
        ctx_unreliable = ProvenanceContext(tag=tag, source_module="a", target_module="b", source_reliability=0.3)

        assert ligase.active(ctx_reliable) is False
        assert ligase.active(ctx_unreliable) is True

    def test_ligase_substrate_match(self):
        ligase = E3Ligase(
            name="string_only",
            active=lambda ctx: True,
            substrate_match=lambda data: isinstance(data, str),
            tag_strength=lambda ctx: 0.9,
        )
        assert ligase.substrate_match("hello") is True
        assert ligase.substrate_match(123) is False


class TestDeubiquitinase:
    def test_create_dub(self):
        dub = Deubiquitinase(
            name="test_dub",
            active=lambda ctx: True,
            rescue_condition=lambda tag, ctx: tag.confidence < 0.5,
            rescue_amount=0.2,
        )
        assert dub.name == "test_dub"
        assert dub.rescue_amount == 0.2

    def test_dub_rescue_condition(self):
        dub = Deubiquitinase(
            name="low_conf_rescue",
            active=lambda ctx: True,
            rescue_condition=lambda tag, ctx: tag.confidence < 0.5,
            rescue_amount=0.2,
        )
        tag_low = UbiquitinTag(confidence=0.3, origin="test", generation=0)
        tag_high = UbiquitinTag(confidence=0.8, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag_low, source_module="a", target_module="b")

        assert dub.rescue_condition(tag_low, ctx) is True
        assert dub.rescue_condition(tag_high, ctx) is False


class TestChaperoneRepair:
    def test_create_chaperone(self):
        chaperone = ChaperoneRepair(
            name="json_repair",
            can_repair=lambda data, tag: isinstance(data, str),
            repair=lambda data, tag: (data.strip(), True),
            confidence_boost=0.3,
        )
        assert chaperone.name == "json_repair"
        assert chaperone.confidence_boost == 0.3

    def test_chaperone_repair_success(self):
        chaperone = ChaperoneRepair(
            name="strip_repair",
            can_repair=lambda data, tag: isinstance(data, str),
            repair=lambda data, tag: (data.strip(), True),
        )
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0)
        repaired, success = chaperone.repair("  hello  ", tag)
        assert repaired == "hello"
        assert success is True

    def test_chaperone_repair_failure(self):
        chaperone = ChaperoneRepair(
            name="always_fail",
            can_repair=lambda data, tag: True,
            repair=lambda data, tag: (data, False),
        )
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0)
        repaired, success = chaperone.repair("hello", tag)
        assert success is False
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_quality_components.py -v`
Expected: FAIL with "No module named 'operon_ai.quality.components'"

**Step 3: Write minimal implementation**

```python
# operon_ai/quality/components.py
"""Components for the proteasome system."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from operon_ai.core.types import DataType
from .types import UbiquitinTag


@dataclass
class ProvenanceContext:
    """Runtime context available during tag processing."""

    tag: UbiquitinTag
    source_module: str
    target_module: str
    source_reliability: float = 1.0
    system_load: float = 0.0
    operation_criticality: str = "normal"
    data_type: Optional[DataType] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class E3Ligase:
    """
    Context-sensitive tagger — reduces confidence.

    Biological parallel: E3 ubiquitin ligases that attach
    ubiquitin chains to proteins based on quality signals.
    """

    name: str
    active: Callable[[ProvenanceContext], bool]
    substrate_match: Callable[[Any], bool]
    tag_strength: Callable[[ProvenanceContext], float]  # Multiplier (0.0-1.0)


@dataclass
class Deubiquitinase:
    """
    Context-sensitive eraser — restores confidence.

    Biological parallel: DUBs that remove ubiquitin chains,
    rescuing proteins from degradation.
    """

    name: str
    active: Callable[[ProvenanceContext], bool]
    rescue_condition: Callable[[UbiquitinTag, ProvenanceContext], bool]
    rescue_amount: float  # Added to confidence


@dataclass
class ChaperoneRepair:
    """
    Attempts data repair before degradation.

    Biological parallel: Chaperones that refold misfolded
    proteins before they are sent to proteasome.
    """

    name: str
    can_repair: Callable[[Any, UbiquitinTag], bool]
    repair: Callable[[Any, UbiquitinTag], tuple[Any, bool]]  # (repaired_data, success)
    confidence_boost: float = 0.3
```

**Step 4: Update __init__.py**

Add to exports:
```python
from .components import (
    ProvenanceContext,
    E3Ligase,
    Deubiquitinase,
    ChaperoneRepair,
)
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_quality_components.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add operon_ai/quality/ tests/test_quality_components.py
git commit -m "feat(quality): add E3Ligase, Deubiquitinase, ChaperoneRepair

- ProvenanceContext for runtime context during processing
- E3Ligase: context-sensitive tagger (reduces confidence)
- Deubiquitinase: context-sensitive eraser (restores confidence)
- ChaperoneRepair: attempts data repair before degradation"
```

---

### Task 1.4: Proteasome

**Files:**
- Create: `operon_ai/quality/proteasome.py`
- Modify: `operon_ai/quality/__init__.py`
- Test: `tests/test_proteasome.py`

**Step 1: Write the failing tests**

```python
# tests/test_proteasome.py
"""Tests for the Proteasome organelle."""
import pytest
from operon_ai.quality import (
    UbiquitinTag, UbiquitinPool, DegronType, DegradationResult,
)
from operon_ai.quality.components import (
    ProvenanceContext, Deubiquitinase, ChaperoneRepair,
)
from operon_ai.quality.proteasome import Proteasome


class TestProteasome:
    def test_create_proteasome(self):
        proto = Proteasome()
        assert proto.degradation_threshold == 0.3
        assert proto.block_threshold == 0.1

    def test_inspect_passes_high_confidence(self):
        proto = Proteasome()
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        data, new_tag, result = proto.inspect("hello", tag, ctx, pool)
        assert result == DegradationResult.PASSED
        assert data == "hello"

    def test_inspect_blocks_very_low_confidence(self):
        proto = Proteasome(block_threshold=0.1)
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.05, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        data, new_tag, result = proto.inspect("hello", tag, ctx, pool)
        assert result == DegradationResult.BLOCKED
        assert data is None

    def test_inspect_degrades_medium_confidence(self):
        proto = Proteasome(
            degradation_threshold=0.5,
            block_threshold=0.1,
            fallback_strategy=lambda data, tag: {"degraded": True, "original": data},
        )
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.3, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        data, new_tag, result = proto.inspect("hello", tag, ctx, pool)
        assert result == DegradationResult.DEGRADED
        assert data["degraded"] is True

    def test_inspect_queues_review_no_fallback(self):
        proto = Proteasome(degradation_threshold=0.5, block_threshold=0.1)
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.3, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        data, new_tag, result = proto.inspect("hello", tag, ctx, pool)
        assert result == DegradationResult.QUEUED_REVIEW
        assert len(proto.review_queue) == 1

    def test_inspect_dub_rescue(self):
        dub = Deubiquitinase(
            name="rescue_all",
            active=lambda ctx: True,
            rescue_condition=lambda tag, ctx: tag.confidence < 0.5,
            rescue_amount=0.3,
        )
        proto = Proteasome(
            degradation_threshold=0.5,
            deubiquitinases=[dub],
        )
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.4, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        data, new_tag, result = proto.inspect("hello", tag, ctx, pool)
        assert result == DegradationResult.RESCUED
        assert new_tag.confidence == 0.7  # 0.4 + 0.3

    def test_inspect_chaperone_repair(self):
        chaperone = ChaperoneRepair(
            name="strip",
            can_repair=lambda data, tag: isinstance(data, str),
            repair=lambda data, tag: (data.strip(), True),
            confidence_boost=0.2,
        )
        proto = Proteasome(
            degradation_threshold=0.5,
            chaperones=[chaperone],
        )
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.3, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        data, new_tag, result = proto.inspect("  hello  ", tag, ctx, pool)
        assert result == DegradationResult.REPAIRED
        assert data == "hello"
        assert new_tag.confidence == 0.5  # 0.3 + 0.2

    def test_inspect_degron_adjusts_threshold(self):
        proto = Proteasome(degradation_threshold=0.4, block_threshold=0.1)
        pool = UbiquitinPool(capacity=100)

        # STABLE degron: threshold becomes 0.2 (0.4 * 0.5)
        tag_stable = UbiquitinTag(confidence=0.25, origin="test", generation=0, degron=DegronType.STABLE)
        ctx = ProvenanceContext(tag=tag_stable, source_module="a", target_module="b")
        _, _, result = proto.inspect("hello", tag_stable, ctx, pool)
        assert result == DegradationResult.PASSED  # 0.25 > 0.2

        # UNSTABLE degron: threshold becomes 0.6 (0.4 * 1.5)
        tag_unstable = UbiquitinTag(confidence=0.5, origin="test", generation=0, degron=DegronType.UNSTABLE)
        ctx = ProvenanceContext(tag=tag_unstable, source_module="a", target_module="b")
        proto2 = Proteasome(
            degradation_threshold=0.4,
            fallback_strategy=lambda d, t: d,
        )
        _, _, result = proto2.inspect("hello", tag_unstable, ctx, pool)
        assert result == DegradationResult.DEGRADED  # 0.5 < 0.6

    def test_inspect_respects_capacity(self):
        proto = Proteasome(max_throughput=2)
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.05, origin="test", generation=0)  # Would be blocked
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        # First two use capacity
        proto.inspect("a", tag, ctx, pool)
        proto.inspect("b", tag, ctx, pool)

        # Third should pass through (no capacity)
        data, _, result = proto.inspect("c", tag, ctx, pool)
        assert result == DegradationResult.PASSED
        assert data == "c"

    def test_reset_cycle(self):
        proto = Proteasome(max_throughput=2)
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        proto.inspect("a", tag, ctx, pool)
        proto.inspect("b", tag, ctx, pool)
        assert proto.current_load == 2

        proto.reset_cycle()
        assert proto.current_load == 0

    def test_stats(self):
        proto = Proteasome()
        pool = UbiquitinPool(capacity=100)
        tag = UbiquitinTag(confidence=0.9, origin="test", generation=0)
        ctx = ProvenanceContext(tag=tag, source_module="a", target_module="b")

        proto.inspect("hello", tag, ctx, pool)
        stats = proto.stats()
        assert stats["inspected"] == 1
        assert "repair_rate" in stats
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_proteasome.py -v`
Expected: FAIL with "No module named 'operon_ai.quality.proteasome'"

**Step 3: Write minimal implementation**

```python
# operon_ai/quality/proteasome.py
"""Proteasome organelle for quality inspection and degradation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import UbiquitinTag, UbiquitinPool, DegradationResult
from .components import ProvenanceContext, Deubiquitinase, ChaperoneRepair


@dataclass
class Proteasome:
    """
    Inspects ubiquitin tags and enforces quality thresholds.

    Biological parallel: The 26S proteasome that degrades
    ubiquitin-tagged proteins.
    """

    # Thresholds (adjusted by degron)
    degradation_threshold: float = 0.3
    block_threshold: float = 0.1

    # Capacity (ATP-dependent in biology)
    max_throughput: int = 100
    current_load: int = 0

    # Components
    chaperones: list[ChaperoneRepair] = field(default_factory=list)
    deubiquitinases: list[Deubiquitinase] = field(default_factory=list)

    # Handlers
    fallback_strategy: Optional[Callable[[Any, UbiquitinTag], Any]] = None
    review_queue: list = field(default_factory=list)

    # Metrics
    inspected: int = 0
    repairs_attempted: int = 0
    repairs_succeeded: int = 0

    def inspect(
        self,
        data: Any,
        tag: UbiquitinTag,
        context: ProvenanceContext,
        pool: UbiquitinPool,
    ) -> tuple[Optional[Any], UbiquitinTag, DegradationResult]:
        """
        Inspect data and tag, potentially degrading or blocking.

        Returns: (data, updated_tag, result)
        """
        self.inspected += 1

        # Capacity check - if overloaded, pass through
        if self.current_load >= self.max_throughput:
            return data, tag, DegradationResult.PASSED

        self.current_load += 1

        # Calculate degron-adjusted thresholds
        effective_degrade = tag.effective_threshold(self.degradation_threshold)
        effective_block = tag.effective_threshold(self.block_threshold)

        # Step 1: DUB rescue attempt
        for dub in self.deubiquitinases:
            if dub.active(context) and dub.rescue_condition(tag, context):
                tag = tag.restore_confidence(dub.rescue_amount)
                if tag.confidence >= effective_degrade:
                    return data, tag, DegradationResult.RESCUED

        # Step 2: Chaperone repair (before degradation)
        if tag.confidence < effective_degrade:
            for chaperone in self.chaperones:
                if chaperone.can_repair(data, tag):
                    self.repairs_attempted += 1
                    repaired_data, success = chaperone.repair(data, tag)
                    if success:
                        self.repairs_succeeded += 1
                        tag = tag.restore_confidence(chaperone.confidence_boost)
                        return repaired_data, tag, DegradationResult.REPAIRED

        # Step 3: Threshold enforcement
        if tag.confidence < effective_block:
            pool.recycle(tag)
            return None, tag, DegradationResult.BLOCKED

        if tag.confidence < effective_degrade:
            pool.recycle(tag)
            if self.fallback_strategy:
                degraded = self.fallback_strategy(data, tag)
                return degraded, tag, DegradationResult.DEGRADED
            else:
                self.review_queue.append((data, tag, context))
                return None, tag, DegradationResult.QUEUED_REVIEW

        # Passed inspection
        return data, tag, DegradationResult.PASSED

    def reset_cycle(self) -> None:
        """Reset throughput for new cycle."""
        self.current_load = 0

    def stats(self) -> dict:
        """Return inspection statistics."""
        return {
            "inspected": self.inspected,
            "repairs_attempted": self.repairs_attempted,
            "repairs_succeeded": self.repairs_succeeded,
            "repair_rate": (
                self.repairs_succeeded / self.repairs_attempted
                if self.repairs_attempted > 0 else 0.0
            ),
            "pending_review": len(self.review_queue),
            "current_load": self.current_load,
        }
```

**Step 4: Update __init__.py**

Add:
```python
from .proteasome import Proteasome
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_proteasome.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add operon_ai/quality/ tests/test_proteasome.py
git commit -m "feat(quality): add Proteasome organelle

- Inspects tags against degron-adjusted thresholds
- DUB rescue before degradation
- Chaperone repair attempts
- Fallback strategy or review queue
- Throughput capacity limits"
```

---

## Phase 2: Surveillance System (Immune)

### Task 2.1: Core Surveillance Types

**Files:**
- Create: `operon_ai/surveillance/__init__.py`
- Create: `operon_ai/surveillance/types.py`
- Test: `tests/test_surveillance_types.py`

**Step 1: Write the failing tests**

```python
# tests/test_surveillance_types.py
"""Tests for surveillance system types."""
import pytest
from datetime import datetime
from operon_ai.surveillance.types import (
    Signal1, Signal2, ThreatLevel, ResponseAction,
    MHCPeptide, ActivationState,
)


class TestEnums:
    def test_signal1_values(self):
        assert Signal1.SELF.value == "self"
        assert Signal1.NON_SELF.value == "non_self"
        assert Signal1.UNKNOWN.value == "unknown"

    def test_signal2_values(self):
        assert Signal2.NONE.value == "none"
        assert Signal2.CANARY_FAILED.value == "canary"
        assert Signal2.CROSS_VALIDATED.value == "cross"
        assert Signal2.REPEATED_ANOMALY.value == "repeat"

    def test_threat_level_values(self):
        assert ThreatLevel.NONE.value == "none"
        assert ThreatLevel.SUSPICIOUS.value == "suspicious"
        assert ThreatLevel.CONFIRMED.value == "confirmed"
        assert ThreatLevel.CRITICAL.value == "critical"

    def test_response_action_values(self):
        assert ResponseAction.IGNORE.value == "ignore"
        assert ResponseAction.MONITOR.value == "monitor"
        assert ResponseAction.ISOLATE.value == "isolate"
        assert ResponseAction.SHUTDOWN.value == "shutdown"


class TestMHCPeptide:
    def test_create_peptide(self):
        peptide = MHCPeptide(
            agent_id="test_agent",
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc123",
            structure_hash="def456",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=("timeout",),
        )
        assert peptide.agent_id == "test_agent"
        assert peptide.output_length_mean == 100.0

    def test_peptide_similarity_same_agent(self):
        peptide1 = MHCPeptide(
            agent_id="test",
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc",
            structure_hash="def",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=(),
        )
        peptide2 = MHCPeptide(
            agent_id="test",
            timestamp=datetime.utcnow(),
            output_length_mean=105.0,  # Similar
            output_length_std=10.0,
            response_time_mean=0.52,   # Similar
            response_time_std=0.1,
            vocabulary_hash="abc",     # Same
            structure_hash="def",      # Same
            confidence_mean=0.88,      # Similar
            confidence_std=0.05,
            error_rate=0.02,           # Similar
            error_types=(),
        )
        similarity = peptide1.similarity(peptide2)
        assert similarity > 0.8  # High similarity

    def test_peptide_similarity_different_agent(self):
        peptide1 = MHCPeptide(
            agent_id="agent_a",
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc",
            structure_hash="def",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=(),
        )
        peptide2 = MHCPeptide(
            agent_id="agent_b",  # Different agent
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc",
            structure_hash="def",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=(),
        )
        assert peptide1.similarity(peptide2) == 0.0


class TestActivationState:
    def test_create_state(self):
        state = ActivationState(agent_id="test")
        assert state.agent_id == "test"
        assert state.signal1 == Signal1.SELF
        assert state.signal2 == Signal2.NONE

    def test_is_activated_requires_both_signals(self):
        state = ActivationState(agent_id="test")
        assert state.is_activated is False

        state.signal1 = Signal1.NON_SELF
        assert state.is_activated is False  # Still needs signal2

        state.signal2 = Signal2.CANARY_FAILED
        assert state.is_activated is True
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_surveillance_types.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# operon_ai/surveillance/__init__.py
"""Surveillance system (Immune model)."""
from .types import (
    Signal1,
    Signal2,
    ThreatLevel,
    ResponseAction,
    MHCPeptide,
    ActivationState,
)

__all__ = [
    "Signal1",
    "Signal2",
    "ThreatLevel",
    "ResponseAction",
    "MHCPeptide",
    "ActivationState",
]
```

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/bogdan/core/operon/.worktrees/cell-integrity && python3.11 -m pytest tests/test_surveillance_types.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add operon_ai/surveillance/ tests/test_surveillance_types.py
git commit -m "feat(surveillance): add core types for immune system

- Signal1/Signal2 enums for two-signal activation
- ThreatLevel and ResponseAction enums
- MHCPeptide behavioral fingerprint with similarity calculation
- ActivationState tracking for T-cells"
```

---

*[Plan continues with Tasks 2.2-2.6 for Surveillance and Tasks 3.1-3.5 for Coordination, following same TDD pattern. Truncated for length but follows identical structure.]*

---

## Phase 3: Coordination System (Cell Cycle)

### Task 3.1: Core Coordination Types

*(Same TDD structure: Phase, CheckpointResult, ResourceLock, DependencyGraph)*

### Task 3.2: Cell Cycle Controller

*(Checkpoint conditions, phase management, resource acquisition)*

### Task 3.3: Watchdog (Apoptosis)

*(Timeout monitoring, deadlock breaking, termination)*

### Task 3.4: Priority Inheritance

*(Inversion prevention, priority boosting)*

### Task 3.5: Coordination System Organelle

*(Complete integration)*

---

## Phase 4: Integration

### Task 4.1: IntegratedCell

**Files:**
- Create: `operon_ai/cell.py`
- Test: `tests/test_integrated_cell.py`

*(Cross-system wiring, unified execution, health checks)*

### Task 4.2: Examples

**Files:**
- Create: `examples/18_cell_integrity_demo.py`

---

## Execution Notes

- Run all tests after each task: `python3.11 -m pytest tests/ -v`
- Commit after each green test
- If tests fail unexpectedly, debug before proceeding
