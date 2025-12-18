"""Immune Memory - threat pattern storage."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json

from .types import ThreatLevel, ResponseAction


@dataclass
class ThreatSignature:
    """
    Stored signature of a known threat.

    Biological parallel: Memory B-cell or T-cell that remembers
    a specific pathogen pattern for faster future response.
    """

    agent_id: str
    vocabulary_hash: str
    structure_hash: str
    violation_types: tuple[str, ...]
    threat_level: ThreatLevel
    effective_response: ResponseAction

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    recall_count: int = 0

    def matches(self, other: "ThreatSignature", partial: bool = False) -> bool:
        """Check if signatures match."""
        if self.agent_id != other.agent_id:
            return False

        if not partial:
            return (
                self.vocabulary_hash == other.vocabulary_hash and
                self.structure_hash == other.structure_hash
            )

        # Partial match - check if any violation types overlap
        return bool(set(self.violation_types) & set(other.violation_types))

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.utcnow()
        self.recall_count += 1

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "agent_id": self.agent_id,
            "vocabulary_hash": self.vocabulary_hash,
            "structure_hash": self.structure_hash,
            "violation_types": list(self.violation_types),
            "threat_level": self.threat_level.value,
            "effective_response": self.effective_response.value,
            "created_at": self.created_at.isoformat(),
            "recall_count": self.recall_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThreatSignature":
        """Import from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            vocabulary_hash=data["vocabulary_hash"],
            structure_hash=data["structure_hash"],
            violation_types=tuple(data["violation_types"]),
            threat_level=ThreatLevel(data["threat_level"]),
            effective_response=ResponseAction(data["effective_response"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            recall_count=data.get("recall_count", 0),
        )


@dataclass
class ImmuneMemory:
    """
    Stores threat patterns for accelerated response.

    Biological parallel: Immune memory that enables faster
    secondary response to previously encountered pathogens.
    """

    capacity: int = 1000
    signatures: list[ThreatSignature] = field(default_factory=list)

    def store(self, signature: ThreatSignature) -> None:
        """Store a new threat signature."""
        # Prune if at capacity
        if len(self.signatures) >= self.capacity:
            self._prune_least_accessed()

        self.signatures.append(signature)

    def recall(self, query: ThreatSignature, partial: bool = False) -> Optional[ThreatSignature]:
        """
        Recall stored signature matching query.

        Returns None if no match found.
        """
        for sig in self.signatures:
            if sig.matches(query, partial=partial):
                sig.touch()
                return sig
        return None

    def recall_by_hashes(
        self,
        agent_id: str,
        vocabulary_hash: str,
        structure_hash: str,
    ) -> Optional[ThreatSignature]:
        """Recall by hash values directly."""
        for sig in self.signatures:
            if (sig.agent_id == agent_id and
                sig.vocabulary_hash == vocabulary_hash and
                sig.structure_hash == structure_hash):
                sig.touch()
                return sig
        return None

    def _prune_least_accessed(self) -> None:
        """Remove least recently accessed signature."""
        if not self.signatures:
            return

        # Find oldest by last_accessed
        oldest = min(self.signatures, key=lambda s: s.last_accessed)
        self.signatures.remove(oldest)

    def prune_old(self, max_age: timedelta) -> int:
        """Remove signatures older than max_age. Returns count removed."""
        cutoff = datetime.utcnow() - max_age
        before = len(self.signatures)
        self.signatures = [s for s in self.signatures if s.created_at > cutoff]
        return before - len(self.signatures)

    def export_signatures(self) -> list[dict]:
        """Export all signatures for persistence."""
        return [sig.to_dict() for sig in self.signatures]

    def import_signatures(self, data: list[dict]) -> int:
        """Import signatures from exported data. Returns count imported."""
        imported = 0
        for item in data:
            sig = ThreatSignature.from_dict(item)
            if len(self.signatures) < self.capacity:
                self.signatures.append(sig)
                imported += 1
        return imported

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "stored": len(self.signatures),
            "capacity": self.capacity,
            "utilization": f"{len(self.signatures) / self.capacity:.1%}",
            "total_recalls": sum(s.recall_count for s in self.signatures),
        }
