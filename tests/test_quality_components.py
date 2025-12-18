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
