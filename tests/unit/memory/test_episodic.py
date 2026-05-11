import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from operon_ai.memory.episodic import EpisodicMemory, MemoryEntry, MemoryTier


class TestMemoryEntry:
    def test_initialization(self):
        entry = MemoryEntry(content="test content", tier=MemoryTier.WORKING)
        assert entry.content == "test content"
        assert entry.tier == MemoryTier.WORKING
        assert isinstance(entry.id, str)
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.last_accessed, datetime)
        assert entry.access_count == 0
        assert entry.strength == 1.0
        assert entry.decay_rate == 0.1
        assert entry.histone_marks == {}

    def test_decay(self):
        # WORKING memory decays
        entry = MemoryEntry(content="test", tier=MemoryTier.WORKING, decay_rate=0.2)
        entry.decay()
        assert entry.strength == 0.8
        entry.decay()
        assert pytest.approx(entry.strength) == 0.6

        # LONGTERM memory does not decay
        entry_lt = MemoryEntry(content="test2", tier=MemoryTier.LONGTERM, decay_rate=0.0)
        entry_lt.decay()
        assert entry_lt.strength == 1.0

        # Strength doesn't go below 0.0
        entry_fast = MemoryEntry(content="fast", tier=MemoryTier.WORKING, strength=0.1, decay_rate=0.5)
        entry_fast.decay()
        assert entry_fast.strength == 0.0

    def test_access(self):
        entry = MemoryEntry(content="test", tier=MemoryTier.WORKING, strength=0.8)
        initial_accessed = entry.last_accessed

        entry.access()
        assert entry.access_count == 1
        assert pytest.approx(entry.strength) == 0.85
        assert entry.last_accessed > initial_accessed

        # Strength caps at 1.0
        entry_strong = MemoryEntry(content="test", tier=MemoryTier.WORKING, strength=0.98)
        entry_strong.access()
        assert entry_strong.strength == 1.0

    def test_to_and_from_dict(self):
        entry = MemoryEntry(
            content="test serialization",
            tier=MemoryTier.EPISODIC,
            strength=0.9,
            decay_rate=0.05,
            histone_marks={"importance": 0.8}
        )

        data = entry.to_dict()
        assert data["content"] == "test serialization"
        assert data["tier"] == "episodic"
        assert data["strength"] == 0.9
        assert data["decay_rate"] == 0.05
        assert data["histone_marks"] == {"importance": 0.8}

        new_entry = MemoryEntry.from_dict(data)
        assert new_entry.id == entry.id
        assert new_entry.content == entry.content
        assert new_entry.tier == entry.tier
        assert new_entry.strength == entry.strength
        assert new_entry.decay_rate == entry.decay_rate
        assert new_entry.histone_marks == entry.histone_marks
        assert new_entry.created_at == entry.created_at
        assert new_entry.last_accessed == entry.last_accessed


class TestEpisodicMemory:
    @pytest.fixture
    def memory(self):
        return EpisodicMemory()

    @pytest.fixture
    def memory_with_persistence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield EpisodicMemory(persistence_path=temp_dir)

    def test_store(self, memory):
        entry = memory.store("hello world", tier=MemoryTier.EPISODIC, histone_marks={"reliability": 0.9})
        assert entry.id in memory.memories
        assert entry.content == "hello world"
        assert entry.tier == MemoryTier.EPISODIC
        assert entry.histone_marks == {"reliability": 0.9}
        assert entry.decay_rate == EpisodicMemory.DECAY_RATES[MemoryTier.EPISODIC]

    def test_retrieve(self, memory):
        memory.store("python programming", tier=MemoryTier.WORKING, histone_marks={"importance": 1.0})
        memory.store("java programming", tier=MemoryTier.WORKING, histone_marks={"importance": 0.5})
        memory.store("cooking recipes", tier=MemoryTier.WORKING)

        # Test basic retrieval
        results = memory.retrieve("programming")
        assert len(results) == 2

        # Test case insensitivity
        results = memory.retrieve("PROGRAMming")
        assert len(results) == 2

        # Test sorting by importance (python programming has higher importance)
        assert results[0].content == "python programming"
        assert results[1].content == "java programming"

        # Test access increases strength
        assert results[0].access_count == 2
        assert results[1].access_count == 2

        # Test min_strength filter
        entry = memory.store("low strength programming")
        entry.strength = 0.05
        results_filtered = memory.retrieve("programming", min_strength=0.1)
        # Should not retrieve "low strength programming"
        assert all("low strength" not in r.content for r in results_filtered)

    def test_get_by_id(self, memory):
        entry = memory.store("find me")
        assert memory.get_by_id(entry.id) == entry
        assert memory.get_by_id("invalid-id") is None

    def test_get_tier(self, memory):
        memory.store("w1", tier=MemoryTier.WORKING)
        memory.store("w2", tier=MemoryTier.WORKING)
        memory.store("e1", tier=MemoryTier.EPISODIC)
        memory.store("l1", tier=MemoryTier.LONGTERM)

        assert len(memory.get_tier(MemoryTier.WORKING)) == 2
        assert len(memory.get_tier(MemoryTier.EPISODIC)) == 1
        assert len(memory.get_tier(MemoryTier.LONGTERM)) == 1

    def test_add_mark(self, memory):
        entry = memory.store("mark me")
        memory.add_mark(entry.id, "reliability", 0.95)
        assert memory.memories[entry.id].histone_marks["reliability"] == 0.95

    def test_promote(self, memory):
        entry = memory.store("promote me", tier=MemoryTier.WORKING)
        memory.promote(entry.id, MemoryTier.LONGTERM)

        promoted = memory.get_by_id(entry.id)
        assert promoted.tier == MemoryTier.LONGTERM
        assert promoted.decay_rate == EpisodicMemory.DECAY_RATES[MemoryTier.LONGTERM]

    def test_decay_all(self, memory):
        e1 = memory.store("w1", tier=MemoryTier.WORKING)
        e2 = memory.store("e1", tier=MemoryTier.EPISODIC)
        e3 = memory.store("l1", tier=MemoryTier.LONGTERM)

        # Set strength low to test removal
        e1.strength = 0.15  # will go to 0.0 and be removed (0.15 - 0.2 < 0)
        e2.strength = 0.5   # will go to 0.45

        memory.decay_all()

        assert e1.id not in memory.memories
        assert e2.id in memory.memories
        assert memory.memories[e2.id].strength == 0.45
        assert memory.memories[e3.id].strength == 1.0

    def test_save_and_load(self, memory_with_persistence):
        # Store in different tiers
        memory_with_persistence.store("long memory 1", tier=MemoryTier.LONGTERM)
        memory_with_persistence.store("long memory 2", tier=MemoryTier.LONGTERM)
        memory_with_persistence.store("short memory", tier=MemoryTier.WORKING)

        # Save should only save LONGTERM
        memory_with_persistence.save()

        # Create a new memory instance with the same path
        new_memory = EpisodicMemory(persistence_path=memory_with_persistence.persistence_path)
        new_memory.load()

        assert len(new_memory.memories) == 2
        contents = [e.content for e in new_memory.memories.values()]
        assert "long memory 1" in contents
        assert "long memory 2" in contents
        assert "short memory" not in contents

    def test_format_context(self, memory):
        memory.store("user likes python", histone_marks={"reliability": 0.9})
        memory.store("user hates java", histone_marks={"reliability": 0.6})

        context = memory.format_context("user")
        assert "Relevant memories:" in context
        assert "[90% reliable] user likes python" in context
        assert "[60% reliable] user hates java" in context

        # Test empty context
        assert memory.format_context("something completely different") == ""
