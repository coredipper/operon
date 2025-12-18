"""Tests for the episodic memory system."""

import pytest
import tempfile
import os
from pathlib import Path


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        from operon_ai.memory import MemoryEntry, MemoryTier

        entry = MemoryEntry(
            content="test memory",
            tier=MemoryTier.WORKING,
        )
        assert entry.content == "test memory"
        assert entry.tier == MemoryTier.WORKING
        assert entry.access_count == 0

    def test_memory_entry_decay(self):
        from operon_ai.memory import MemoryEntry, MemoryTier

        entry = MemoryEntry(content="test", tier=MemoryTier.WORKING, decay_rate=0.1)
        initial_strength = entry.strength
        entry.decay()
        assert entry.strength < initial_strength


class TestEpisodicMemory:
    """Test the EpisodicMemory system."""

    def test_store_and_retrieve(self):
        from operon_ai.memory import EpisodicMemory, MemoryTier

        memory = EpisodicMemory()
        memory.store("Hello world", tier=MemoryTier.WORKING)

        results = memory.retrieve("Hello")
        assert len(results) > 0
        assert "Hello world" in results[0].content

    def test_memory_tiers(self):
        from operon_ai.memory import EpisodicMemory, MemoryTier

        memory = EpisodicMemory()
        memory.store("working memory", tier=MemoryTier.WORKING)
        memory.store("long term memory", tier=MemoryTier.LONGTERM)

        working = memory.get_tier(MemoryTier.WORKING)
        longterm = memory.get_tier(MemoryTier.LONGTERM)

        assert len(working) == 1
        assert len(longterm) == 1

    def test_histone_marks(self):
        from operon_ai.memory import EpisodicMemory, MemoryTier

        memory = EpisodicMemory()
        entry = memory.store("test content", tier=MemoryTier.EPISODIC)

        memory.add_mark(entry.id, "reliability", 0.8)
        memory.add_mark(entry.id, "importance", 0.9)

        retrieved = memory.get_by_id(entry.id)
        assert retrieved.histone_marks["reliability"] == 0.8
        assert retrieved.histone_marks["importance"] == 0.9

    def test_persistence(self):
        from operon_ai.memory import EpisodicMemory, MemoryTier

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate memory
            memory1 = EpisodicMemory(persistence_path=tmpdir)
            memory1.store("persistent memory", tier=MemoryTier.LONGTERM)
            memory1.save()

            # Load in new instance
            memory2 = EpisodicMemory(persistence_path=tmpdir)
            memory2.load()

            results = memory2.retrieve("persistent")
            assert len(results) > 0
