"""Tests for Immune Memory."""
import pytest
from datetime import datetime, timedelta
from operon_ai.surveillance.types import ThreatLevel, ResponseAction
from operon_ai.surveillance.memory import (
    ImmuneMemory, ThreatSignature,
)


def make_signature(
    agent_id: str = "test",
    vocabulary_hash: str = "abc123",
    structure_hash: str = "def456",
    violation_types: tuple[str, ...] = ("output_length",),
    threat_level: ThreatLevel = ThreatLevel.CONFIRMED,
    effective_response: ResponseAction = ResponseAction.ISOLATE,
) -> ThreatSignature:
    """Helper to create test signatures."""
    return ThreatSignature(
        agent_id=agent_id,
        vocabulary_hash=vocabulary_hash,
        structure_hash=structure_hash,
        violation_types=violation_types,
        threat_level=threat_level,
        effective_response=effective_response,
    )


class TestThreatSignature:
    def test_create_signature(self):
        sig = make_signature()
        assert sig.agent_id == "test"
        assert sig.vocabulary_hash == "abc123"
        assert sig.threat_level == ThreatLevel.CONFIRMED

    def test_signature_match_same(self):
        sig1 = make_signature()
        sig2 = make_signature()
        assert sig1.matches(sig2) is True

    def test_signature_match_different_vocab(self):
        sig1 = make_signature(vocabulary_hash="abc123")
        sig2 = make_signature(vocabulary_hash="xyz789")
        assert sig1.matches(sig2) is False

    def test_signature_partial_match(self):
        sig1 = make_signature(violation_types=("output_length", "vocabulary"))
        sig2 = make_signature(violation_types=("output_length",))
        # Partial match if some violations overlap
        assert sig1.matches(sig2, partial=True) is True


class TestImmuneMemory:
    def test_create_memory(self):
        memory = ImmuneMemory(capacity=100)
        assert memory.capacity == 100
        assert len(memory.signatures) == 0

    def test_store_signature(self):
        memory = ImmuneMemory()
        sig = make_signature()
        memory.store(sig)
        assert len(memory.signatures) == 1

    def test_recall_exact_match(self):
        memory = ImmuneMemory()
        sig = make_signature(vocabulary_hash="abc123")
        memory.store(sig)

        query = make_signature(vocabulary_hash="abc123")
        recalled = memory.recall(query)
        assert recalled is not None
        assert recalled.vocabulary_hash == "abc123"

    def test_recall_no_match(self):
        memory = ImmuneMemory()
        sig = make_signature(vocabulary_hash="abc123")
        memory.store(sig)

        query = make_signature(vocabulary_hash="xyz789")
        recalled = memory.recall(query)
        assert recalled is None

    def test_recall_updates_access_time(self):
        memory = ImmuneMemory()
        sig = make_signature()
        memory.store(sig)

        old_access = memory.signatures[0].last_accessed
        memory.recall(sig)
        new_access = memory.signatures[0].last_accessed
        assert new_access >= old_access

    def test_capacity_prunes_oldest(self):
        memory = ImmuneMemory(capacity=2)

        sig1 = make_signature(vocabulary_hash="hash1")
        sig2 = make_signature(vocabulary_hash="hash2")
        sig3 = make_signature(vocabulary_hash="hash3")

        memory.store(sig1)
        memory.store(sig2)
        memory.store(sig3)  # Should prune oldest

        assert len(memory.signatures) == 2
        # sig1 should be gone
        hashes = [s.vocabulary_hash for s in memory.signatures]
        assert "hash1" not in hashes
        assert "hash2" in hashes
        assert "hash3" in hashes

    def test_prune_by_age(self):
        memory = ImmuneMemory()
        sig = make_signature()
        memory.store(sig)

        # Manually age the signature
        memory.signatures[0].created_at = datetime.utcnow() - timedelta(days=100)

        memory.prune_old(max_age=timedelta(days=30))
        assert len(memory.signatures) == 0

    def test_export_import(self):
        memory1 = ImmuneMemory()
        memory1.store(make_signature(vocabulary_hash="hash1"))
        memory1.store(make_signature(vocabulary_hash="hash2"))

        exported = memory1.export_signatures()

        memory2 = ImmuneMemory()
        memory2.import_signatures(exported)

        assert len(memory2.signatures) == 2

    def test_get_stats(self):
        memory = ImmuneMemory()
        memory.store(make_signature())

        stats = memory.stats()
        assert stats["stored"] == 1
        assert "capacity" in stats
