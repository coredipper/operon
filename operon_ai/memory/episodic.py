"""
Episodic Memory System with Histone Marks.

Implements a three-tier memory hierarchy:
- Working: In-session, fast decay (like human working memory)
- Episodic: Learned feedback, slow decay (like episodic memory)
- Long-term: Persisted to disk, no decay (like long-term memory)

Histone marks allow attaching metadata (reliability, importance, etc.)
that affects retrieval ranking and decay behavior.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable


class MemoryTier(Enum):
    """Memory tier determines decay rate and persistence."""
    WORKING = "working"      # Fast decay, in-session only
    EPISODIC = "episodic"    # Slow decay, learned feedback
    LONGTERM = "longterm"    # No decay, persisted


@dataclass
class MemoryEntry:
    """
    Single memory unit with epigenetic marks.

    Histone marks are metadata that affect memory behavior:
    - reliability: How trustworthy this memory is (0-1)
    - importance: How relevant to current context (0-1)
    - emotional_valence: Positive/negative association (-1 to 1)
    """
    content: str
    tier: MemoryTier
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    strength: float = 1.0
    decay_rate: float = 0.1
    histone_marks: dict[str, float] = field(default_factory=dict)

    def decay(self) -> None:
        """Apply decay to memory strength."""
        if self.tier != MemoryTier.LONGTERM:
            self.strength = max(0.0, self.strength - self.decay_rate)

    def access(self) -> None:
        """Record memory access (strengthens memory)."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # Accessing strengthens memory slightly
        self.strength = min(1.0, self.strength + 0.05)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "tier": self.tier.value,
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "strength": self.strength,
            "decay_rate": self.decay_rate,
            "histone_marks": self.histone_marks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Deserialize from dictionary."""
        return cls(
            content=data["content"],
            tier=MemoryTier(data["tier"]),
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            strength=data["strength"],
            decay_rate=data["decay_rate"],
            histone_marks=data.get("histone_marks", {}),
        )


class EpisodicMemory:
    """
    Three-tier episodic memory system.

    Manages working, episodic, and long-term memories with:
    - Automatic decay over time
    - Histone mark annotations
    - Persistence to disk for long-term memories
    - Relevance-based retrieval
    """

    # Decay rates by tier
    DECAY_RATES = {
        MemoryTier.WORKING: 0.2,    # Fast decay
        MemoryTier.EPISODIC: 0.05,  # Slow decay
        MemoryTier.LONGTERM: 0.0,   # No decay
    }

    def __init__(self, persistence_path: str | Path | None = None):
        self.memories: dict[str, MemoryEntry] = {}
        self.persistence_path = Path(persistence_path) if persistence_path else None

        if self.persistence_path:
            self.persistence_path.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.WORKING,
        histone_marks: dict[str, float] | None = None,
    ) -> MemoryEntry:
        """Store a new memory."""
        entry = MemoryEntry(
            content=content,
            tier=tier,
            decay_rate=self.DECAY_RATES[tier],
            histone_marks=histone_marks or {},
        )
        self.memories[entry.id] = entry
        return entry

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        min_strength: float = 0.1,
    ) -> list[MemoryEntry]:
        """
        Retrieve memories relevant to query.

        Uses simple substring matching. In production, use embeddings.
        """
        query_lower = query.lower()

        matches = []
        for entry in self.memories.values():
            if entry.strength < min_strength:
                continue

            # Simple relevance: substring match
            if query_lower in entry.content.lower():
                entry.access()
                matches.append(entry)

        # Sort by strength * importance mark (if present)
        def score(e: MemoryEntry) -> float:
            importance = e.histone_marks.get("importance", 1.0)
            reliability = e.histone_marks.get("reliability", 1.0)
            return e.strength * importance * reliability

        matches.sort(key=score, reverse=True)
        return matches[:limit]

    def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Get memory by ID."""
        return self.memories.get(memory_id)

    def get_tier(self, tier: MemoryTier) -> list[MemoryEntry]:
        """Get all memories in a tier."""
        return [e for e in self.memories.values() if e.tier == tier]

    def add_mark(self, memory_id: str, mark_name: str, value: float) -> None:
        """Add or update a histone mark on a memory."""
        if entry := self.memories.get(memory_id):
            entry.histone_marks[mark_name] = value

    def promote(self, memory_id: str, to_tier: MemoryTier) -> None:
        """Promote memory to a higher tier."""
        if entry := self.memories.get(memory_id):
            entry.tier = to_tier
            entry.decay_rate = self.DECAY_RATES[to_tier]

    def decay_all(self) -> None:
        """Apply decay to all memories."""
        for entry in self.memories.values():
            entry.decay()

        # Remove memories with zero strength
        self.memories = {
            k: v for k, v in self.memories.items()
            if v.strength > 0
        }

    def save(self) -> None:
        """Persist long-term memories to disk."""
        if not self.persistence_path:
            return

        longterm = self.get_tier(MemoryTier.LONGTERM)
        data = [e.to_dict() for e in longterm]

        filepath = self.persistence_path / "longterm.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load long-term memories from disk."""
        if not self.persistence_path:
            return

        filepath = self.persistence_path / "longterm.json"
        if not filepath.exists():
            return

        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            entry = MemoryEntry.from_dict(item)
            self.memories[entry.id] = entry

    def format_context(self, query: str, max_entries: int = 3) -> str:
        """Format relevant memories as context string."""
        memories = self.retrieve(query, limit=max_entries)

        if not memories:
            return ""

        lines = ["Relevant memories:"]
        for m in memories:
            reliability = m.histone_marks.get("reliability", 1.0)
            lines.append(f"- [{reliability:.0%} reliable] {m.content}")

        return "\n".join(lines)
