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
