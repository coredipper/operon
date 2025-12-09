"""
Histone: Epigenetic Memory System
=================================

Biological Analogy:
- DNA methylation: Permanent silencing of genes (strong markers)
- Histone acetylation: Temporary activation (weak markers)
- Histone phosphorylation: Signal-dependent changes (transient markers)
- Chromatin remodeling: Restructuring access to memory
- Epigenetic inheritance: Passing memory to child agents
- Genomic imprinting: Parent-specific memory expression

The Histone system provides sophisticated memory management for agents,
storing learned lessons, preferences, and constraints that bias future
behavior without changing the fundamental "DNA" (configuration).
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import re


class MarkerType(Enum):
    """Types of epigenetic markers with different strengths and decay rates."""
    METHYLATION = "methylation"       # Permanent, strong silencing
    ACETYLATION = "acetylation"       # Temporary, moderate activation
    PHOSPHORYLATION = "phosphorylation"  # Transient, signal-dependent
    UBIQUITINATION = "ubiquitination"   # Tags for degradation/processing


class MarkerStrength(Enum):
    """Strength levels for markers."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    PERMANENT = 4


@dataclass
class EpigeneticMarker:
    """
    A single epigenetic marker (memory unit).

    Contains the lesson/constraint plus metadata about how it should
    affect behavior and how long it should persist.
    """
    content: str
    marker_type: MarkerType
    strength: MarkerStrength
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    decay_hours: float | None = None  # None = permanent
    tags: list[str] = field(default_factory=list)
    context: str = ""  # What triggered this marker
    confidence: float = 1.0  # How confident we are in this marker

    def is_expired(self) -> bool:
        """Check if this marker has decayed."""
        if self.decay_hours is None:
            return False
        age = datetime.now() - self.created_at
        return age > timedelta(hours=self.decay_hours)

    def get_effective_strength(self) -> float:
        """Get strength adjusted for decay and access."""
        base = self.strength.value

        # Boost for frequent access (reinforcement)
        access_boost = min(1.0, self.access_count * 0.1)

        # Decay over time
        if self.decay_hours:
            age = (datetime.now() - self.created_at).total_seconds() / 3600
            decay_factor = max(0, 1 - (age / self.decay_hours))
        else:
            decay_factor = 1.0

        return (base + access_boost) * decay_factor * self.confidence

    def get_hash(self) -> str:
        """Get unique hash for this marker."""
        return hashlib.md5(self.content.encode()).hexdigest()[:8]


@dataclass
class RetrievalResult:
    """Result of a context retrieval operation."""
    markers: list[EpigeneticMarker]
    formatted_context: str
    relevance_scores: dict[str, float]
    total_markers: int
    active_markers: int


class HistoneStore:
    """
    Advanced Epigenetic Memory System.

    Stores and manages learned lessons, constraints, and behavioral biases
    that influence agent behavior without changing configuration.

    Features:

    1. Multiple Marker Types
       - METHYLATION: Permanent, strong constraints
       - ACETYLATION: Temporary boosts
       - PHOSPHORYLATION: Transient, signal-dependent
       - UBIQUITINATION: Marks for processing/removal

    2. Decay and Reinforcement
       - Markers can decay over time
       - Accessing markers reinforces them
       - Unused markers fade

    3. Semantic Retrieval
       - Query with keywords/patterns
       - Tag-based filtering
       - Relevance scoring

    4. Inheritance
       - Copy markers to child agents
       - Selective inheritance by strength
       - Genomic imprinting

    5. Context Windowing
       - Limit context size
       - Priority-based selection
       - Recency weighting

    Example:
        >>> histones = HistoneStore()
        >>> histones.methylate("Never run DELETE without WHERE clause")
        >>> histones.acetylate("User prefers verbose output", decay_hours=24)
        >>> context = histones.retrieve_context("database query")
        >>> print(context.formatted_context)
    """

    # Default decay rates by marker type (hours)
    DEFAULT_DECAY = {
        MarkerType.METHYLATION: None,      # Permanent
        MarkerType.ACETYLATION: 168,       # 1 week
        MarkerType.PHOSPHORYLATION: 24,    # 1 day
        MarkerType.UBIQUITINATION: 1,      # 1 hour (processing marker)
    }

    def __init__(
        self,
        max_markers: int = 1000,
        context_window: int = 10,
        decay_check_interval: int = 100,  # Check decay every N operations
        on_marker_expired: Callable[[EpigeneticMarker], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Histone Store.

        Args:
            max_markers: Maximum markers to store
            context_window: Default number of markers in retrieval
            decay_check_interval: How often to check for expired markers
            on_marker_expired: Callback when markers expire
            silent: Suppress console output
        """
        self.max_markers = max_markers
        self.context_window = context_window
        self.decay_check_interval = decay_check_interval
        self.on_marker_expired = on_marker_expired
        self.silent = silent

        # Marker storage
        self._markers: dict[str, EpigeneticMarker] = {}  # hash -> marker
        self._operation_count = 0

        # Statistics
        self._total_added = 0
        self._total_expired = 0
        self._total_retrievals = 0

        # Legacy compatibility
        self.methylations: list[str] = []  # For backward compat

    def add_marker(
        self,
        content: str,
        marker_type: MarkerType = MarkerType.METHYLATION,
        strength: MarkerStrength = MarkerStrength.MODERATE,
        decay_hours: float | None = None,
        tags: list[str] | None = None,
        context: str = "",
        confidence: float = 1.0,
    ) -> str:
        """
        Add an epigenetic marker.

        Args:
            content: The lesson or constraint to remember
            marker_type: Type of marker (affects decay and strength)
            strength: How strong this marker is
            decay_hours: Override default decay (None = use type default)
            tags: Keywords for retrieval
            context: What triggered this marker
            confidence: How confident we are (0-1)

        Returns:
            Hash of the created marker
        """
        self._operation_count += 1
        self._maybe_check_decay()

        # Use default decay if not specified
        if decay_hours is None:
            decay_hours = self.DEFAULT_DECAY.get(marker_type)

        marker = EpigeneticMarker(
            content=content,
            marker_type=marker_type,
            strength=strength,
            decay_hours=decay_hours,
            tags=tags or [],
            context=context,
            confidence=confidence,
        )

        marker_hash = marker.get_hash()

        # Check for duplicate
        if marker_hash in self._markers:
            # Reinforce existing marker
            existing = self._markers[marker_hash]
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            if strength.value > existing.strength.value:
                existing.strength = strength
            if not self.silent:
                print(f"📜 [Epigenetics] Reinforced: '{content[:30]}...'")
            return marker_hash

        # Check capacity
        if len(self._markers) >= self.max_markers:
            self._evict_weakest()

        self._markers[marker_hash] = marker
        self._total_added += 1

        # Legacy compatibility
        if marker_type == MarkerType.METHYLATION:
            self.methylations.append(content)

        if not self.silent:
            emoji = {
                MarkerType.METHYLATION: "🔒",
                MarkerType.ACETYLATION: "🔓",
                MarkerType.PHOSPHORYLATION: "⚡",
                MarkerType.UBIQUITINATION: "🏷️",
            }.get(marker_type, "📜")
            print(f"{emoji} [Epigenetics] Added {marker_type.value}: '{content[:40]}...'")

        return marker_hash

    def methylate(
        self,
        lesson: str,
        strength: MarkerStrength = MarkerStrength.STRONG,
        tags: list[str] | None = None,
        context: str = "",
    ) -> str:
        """
        Add a permanent methylation marker (strong silencing).

        Use for critical constraints that should never be forgotten.
        """
        return self.add_marker(
            content=lesson,
            marker_type=MarkerType.METHYLATION,
            strength=strength,
            tags=tags,
            context=context,
        )

    def acetylate(
        self,
        lesson: str,
        decay_hours: float = 168,  # 1 week default
        strength: MarkerStrength = MarkerStrength.MODERATE,
        tags: list[str] | None = None,
        context: str = "",
    ) -> str:
        """
        Add a temporary acetylation marker (activation).

        Use for preferences and temporary biases.
        """
        return self.add_marker(
            content=lesson,
            marker_type=MarkerType.ACETYLATION,
            strength=strength,
            decay_hours=decay_hours,
            tags=tags,
            context=context,
        )

    def phosphorylate(
        self,
        lesson: str,
        decay_hours: float = 24,  # 1 day default
        tags: list[str] | None = None,
        context: str = "",
    ) -> str:
        """
        Add a transient phosphorylation marker (signal-dependent).

        Use for context-specific adjustments.
        """
        return self.add_marker(
            content=lesson,
            marker_type=MarkerType.PHOSPHORYLATION,
            strength=MarkerStrength.WEAK,
            decay_hours=decay_hours,
            tags=tags,
            context=context,
        )

    def ubiquitinate(
        self,
        content: str,
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a ubiquitination marker (tag for processing).

        Use to mark items for special handling or removal.
        """
        return self.add_marker(
            content=content,
            marker_type=MarkerType.UBIQUITINATION,
            strength=MarkerStrength.WEAK,
            decay_hours=1,  # Short-lived
            tags=tags,
        )

    def retrieve_context(
        self,
        query: str = "",
        tags: list[str] | None = None,
        marker_types: list[MarkerType] | None = None,
        min_strength: MarkerStrength = MarkerStrength.WEAK,
        limit: int | None = None,
        include_expired: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve relevant context based on query.

        Args:
            query: Keywords to search for
            tags: Filter by tags
            marker_types: Filter by marker types
            min_strength: Minimum marker strength
            limit: Maximum markers to return
            include_expired: Include expired markers

        Returns:
            RetrievalResult with markers and formatted context
        """
        self._operation_count += 1
        self._total_retrievals += 1
        self._maybe_check_decay()

        limit = limit or self.context_window
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score and filter markers
        scored_markers: list[tuple[float, EpigeneticMarker]] = []

        for marker in self._markers.values():
            # Filter by expiration
            if not include_expired and marker.is_expired():
                continue

            # Filter by marker type
            if marker_types and marker.marker_type not in marker_types:
                continue

            # Filter by minimum strength
            if marker.strength.value < min_strength.value:
                continue

            # Filter by tags
            if tags and not any(t in marker.tags for t in tags):
                continue

            # Calculate relevance score
            score = self._calculate_relevance(marker, query_lower, query_words)

            if score > 0 or not query:  # Include all if no query
                scored_markers.append((score, marker))

        # Sort by score (descending) and take top N
        scored_markers.sort(key=lambda x: x[0], reverse=True)
        top_markers = scored_markers[:limit]

        # Update access counts
        for _, marker in top_markers:
            marker.access_count += 1
            marker.last_accessed = datetime.now()

        # Build result
        markers = [m for _, m in top_markers]
        relevance_scores = {m.get_hash(): s for s, m in top_markers}

        # Format context string
        if not markers:
            formatted = ""
        else:
            lines = []
            for marker in markers:
                strength_indicator = "!" * marker.strength.value
                type_emoji = {
                    MarkerType.METHYLATION: "🔒",
                    MarkerType.ACETYLATION: "🔓",
                    MarkerType.PHOSPHORYLATION: "⚡",
                    MarkerType.UBIQUITINATION: "🏷️",
                }.get(marker.marker_type, "•")
                lines.append(f"{type_emoji} [{strength_indicator}] {marker.content}")

            formatted = "⚠️ EPIGENETIC MEMORY:\n" + "\n".join(f"  - {line}" for line in lines)

        return RetrievalResult(
            markers=markers,
            formatted_context=formatted,
            relevance_scores=relevance_scores,
            total_markers=len(self._markers),
            active_markers=len([m for m in self._markers.values() if not m.is_expired()]),
        )

    def _calculate_relevance(
        self,
        marker: EpigeneticMarker,
        query_lower: str,
        query_words: set[str]
    ) -> float:
        """Calculate relevance score for a marker."""
        score = 0.0

        content_lower = marker.content.lower()
        content_words = set(content_lower.split())

        # Exact substring match
        if query_lower and query_lower in content_lower:
            score += 5.0

        # Word overlap
        overlap = query_words & content_words
        score += len(overlap) * 2.0

        # Tag match
        tag_words = set(t.lower() for t in marker.tags)
        tag_overlap = query_words & tag_words
        score += len(tag_overlap) * 3.0

        # Boost by strength
        score *= marker.get_effective_strength()

        # Recency boost (more recent = higher score)
        age_hours = (datetime.now() - marker.last_accessed).total_seconds() / 3600
        recency_factor = max(0.5, 1 - (age_hours / 168))  # Decay over 1 week
        score *= recency_factor

        return score

    def remove_marker(self, marker_hash: str) -> bool:
        """Remove a marker by hash."""
        if marker_hash in self._markers:
            marker = self._markers.pop(marker_hash)
            # Legacy compat
            if marker.content in self.methylations:
                self.methylations.remove(marker.content)
            return True
        return False

    def clear_type(self, marker_type: MarkerType):
        """Remove all markers of a specific type."""
        to_remove = [
            h for h, m in self._markers.items()
            if m.marker_type == marker_type
        ]
        for h in to_remove:
            self.remove_marker(h)

    def _maybe_check_decay(self):
        """Periodically check for expired markers."""
        if self._operation_count % self.decay_check_interval == 0:
            self._purge_expired()

    def _purge_expired(self):
        """Remove all expired markers."""
        expired = [
            (h, m) for h, m in self._markers.items()
            if m.is_expired()
        ]

        for marker_hash, marker in expired:
            del self._markers[marker_hash]
            self._total_expired += 1

            if self.on_marker_expired:
                self.on_marker_expired(marker)

            if not self.silent and expired:
                print(f"🗑️ [Epigenetics] Expired {len(expired)} markers")

    def _evict_weakest(self):
        """Evict the weakest marker to make room."""
        if not self._markers:
            return

        weakest_hash = min(
            self._markers.keys(),
            key=lambda h: self._markers[h].get_effective_strength()
        )
        self.remove_marker(weakest_hash)

    def inherit_to(
        self,
        child: 'HistoneStore',
        min_strength: MarkerStrength = MarkerStrength.STRONG,
        marker_types: list[MarkerType] | None = None,
    ):
        """
        Pass markers to a child histone store (epigenetic inheritance).

        Args:
            child: The child store to inherit to
            min_strength: Minimum strength to inherit
            marker_types: Types to inherit (default: methylation only)
        """
        marker_types = marker_types or [MarkerType.METHYLATION]
        inherited = 0

        for marker in self._markers.values():
            if marker.marker_type not in marker_types:
                continue
            if marker.strength.value < min_strength.value:
                continue
            if marker.is_expired():
                continue

            # Add to child with reduced confidence
            child.add_marker(
                content=marker.content,
                marker_type=marker.marker_type,
                strength=marker.strength,
                decay_hours=marker.decay_hours,
                tags=marker.tags.copy(),
                context=f"Inherited from parent: {marker.context}",
                confidence=marker.confidence * 0.8,  # Slight reduction
            )
            inherited += 1

        if not self.silent:
            print(f"🧬 [Epigenetics] Inherited {inherited} markers to child")

    def export_markers(self) -> list[dict]:
        """Export all markers for serialization."""
        return [
            {
                "content": m.content,
                "marker_type": m.marker_type.value,
                "strength": m.strength.value,
                "created_at": m.created_at.isoformat(),
                "decay_hours": m.decay_hours,
                "tags": m.tags,
                "context": m.context,
                "confidence": m.confidence,
                "access_count": m.access_count,
            }
            for m in self._markers.values()
        ]

    def import_markers(self, markers: list[dict]):
        """Import markers from serialized format."""
        for m in markers:
            self.add_marker(
                content=m["content"],
                marker_type=MarkerType(m["marker_type"]),
                strength=MarkerStrength(m["strength"]),
                decay_hours=m.get("decay_hours"),
                tags=m.get("tags", []),
                context=m.get("context", ""),
                confidence=m.get("confidence", 1.0),
            )

    def get_statistics(self) -> dict:
        """Get histone statistics."""
        by_type = {}
        by_strength = {}

        for marker in self._markers.values():
            t = marker.marker_type.value
            by_type[t] = by_type.get(t, 0) + 1

            s = marker.strength.name
            by_strength[s] = by_strength.get(s, 0) + 1

        active_count = len([m for m in self._markers.values() if not m.is_expired()])

        return {
            "total_markers": len(self._markers),
            "active_markers": active_count,
            "expired_markers": len(self._markers) - active_count,
            "total_added": self._total_added,
            "total_expired": self._total_expired,
            "total_retrievals": self._total_retrievals,
            "by_type": by_type,
            "by_strength": by_strength,
        }

    def clear(self):
        """Clear all markers."""
        self._markers.clear()
        self.methylations.clear()
