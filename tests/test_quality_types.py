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
        assert abs(tag.effective_threshold(0.4) - 0.6) < 1e-9  # 0.4 * 1.5

    def test_effective_threshold_immediate(self):
        tag = UbiquitinTag(confidence=0.5, origin="test", generation=0, degron=DegronType.IMMEDIATE)
        assert abs(tag.effective_threshold(0.4) - 1.2) < 1e-9  # 0.4 * 3.0


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
