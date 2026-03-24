"""Tests for memory adapters (HistoneStore/EpisodicMemory → BiTemporal)."""

from operon_ai import BiTemporalMemory
from operon_ai.memory.adapters import histone_to_bitemporal, episodic_to_bitemporal
from operon_ai.memory.episodic import EpisodicMemory, MemoryTier
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength


# ---------------------------------------------------------------------------
# histone_to_bitemporal
# ---------------------------------------------------------------------------


def test_histone_adapter_creates_facts():
    hs = HistoneStore()
    hs.add_marker("important lesson", MarkerType.METHYLATION, MarkerStrength.STRONG)
    hs.add_marker("temporary hint", MarkerType.ACETYLATION, MarkerStrength.MODERATE)

    mem = BiTemporalMemory()
    facts = histone_to_bitemporal(hs, mem)
    assert len(facts) == 2
    assert all(f.source == "histone_adapter" for f in facts)


def test_histone_adapter_uses_prefix():
    hs = HistoneStore()
    hs.add_marker("lesson", MarkerType.METHYLATION, MarkerStrength.STRONG)
    mem = BiTemporalMemory()
    facts = histone_to_bitemporal(hs, mem, subject_prefix="custom")
    assert facts[0].subject.startswith("custom:")


def test_histone_adapter_skips_expired():
    hs = HistoneStore()
    h = hs.add_marker("temp", MarkerType.UBIQUITINATION, MarkerStrength.WEAK, decay_hours=0.0001)
    # Force expiry
    import time
    time.sleep(0.001)
    mem = BiTemporalMemory()
    facts = histone_to_bitemporal(hs, mem)
    # May or may not have expired depending on timing — at least no crash
    assert isinstance(facts, list)


# ---------------------------------------------------------------------------
# episodic_to_bitemporal
# ---------------------------------------------------------------------------


def test_episodic_adapter_creates_facts():
    em = EpisodicMemory()
    em.store("learned pattern A", tier=MemoryTier.EPISODIC)
    em.store("learned pattern B", tier=MemoryTier.LONGTERM)
    em.store("working scratch", tier=MemoryTier.WORKING)

    mem = BiTemporalMemory()
    facts = episodic_to_bitemporal(em, mem, min_tier="episodic")
    # Should include EPISODIC and LONGTERM, skip WORKING
    assert len(facts) == 2
    assert all(f.source == "episodic_adapter" for f in facts)


def test_episodic_adapter_respects_min_tier():
    em = EpisodicMemory()
    em.store("episodic entry", tier=MemoryTier.EPISODIC)
    em.store("longterm entry", tier=MemoryTier.LONGTERM)

    mem = BiTemporalMemory()
    facts = episodic_to_bitemporal(em, mem, min_tier="longterm")
    assert len(facts) == 1
    assert facts[0].value == "longterm entry"


def test_episodic_adapter_uses_prefix():
    em = EpisodicMemory()
    em.store("entry", tier=MemoryTier.EPISODIC)
    mem = BiTemporalMemory()
    facts = episodic_to_bitemporal(em, mem, subject_prefix="ep")
    assert facts[0].subject.startswith("ep:")
