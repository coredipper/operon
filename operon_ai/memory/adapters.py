"""Memory adapters — one-way bridges from existing memory systems to BiTemporalMemory.

These adapters create new bi-temporal facts from existing memory entries
without modifying the source. They are the integration layer between
Operon's epigenetic/episodic memory and the append-only auditable fact store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .bitemporal import BiTemporalFact, BiTemporalMemory

if TYPE_CHECKING:
    from ..memory.episodic import EpisodicMemory, MemoryTier
    from ..state.histone import HistoneStore


def histone_to_bitemporal(
    histone_store: HistoneStore,
    bitemporal: BiTemporalMemory,
    *,
    subject_prefix: str = "histone",
) -> list[BiTemporalFact]:
    """Bridge HistoneStore markers into bi-temporal facts.

    Each marker becomes a fact with:
      - subject: "{prefix}:{marker_hash}"
      - predicate: marker_type value (e.g., "methylation")
      - value: marker content string
      - valid_from/recorded_from: marker created_at
      - source: "histone_adapter"
      - tags: ("histone", marker_type)

    Returns list of created facts.
    """
    created: list[BiTemporalFact] = []
    for marker_hash, marker in histone_store._markers.items():
        if marker.is_expired():
            continue
        fact = bitemporal.record_fact(
            subject=f"{subject_prefix}:{marker_hash[:8]}",
            predicate=marker.marker_type.value,
            value=marker.content,
            valid_from=marker.created_at,
            recorded_from=marker.created_at,
            source="histone_adapter",
            confidence=marker.confidence,
            tags=(subject_prefix, marker.marker_type.value),
        )
        created.append(fact)
    return created


def episodic_to_bitemporal(
    episodic_memory: EpisodicMemory,
    bitemporal: BiTemporalMemory,
    *,
    subject_prefix: str = "episodic",
    min_tier: str = "episodic",
) -> list[BiTemporalFact]:
    """Bridge EpisodicMemory entries into bi-temporal facts.

    Filters entries by minimum tier (default: EPISODIC, skipping WORKING).
    Each entry becomes a fact with:
      - subject: "{prefix}:{entry.id}"
      - predicate: tier value (e.g., "episodic", "longterm")
      - value: entry content string
      - valid_from/recorded_from: entry created_at
      - source: "episodic_adapter"
      - tags: ("episodic", tier)

    Returns list of created facts.
    """
    from ..memory.episodic import MemoryTier

    tier_order = {"working": 0, "episodic": 1, "longterm": 2}
    min_order = tier_order.get(min_tier, 1)

    created: list[BiTemporalFact] = []
    for entry in episodic_memory.memories.values():
        if tier_order.get(entry.tier.value, 0) < min_order:
            continue
        if entry.strength <= 0:
            continue
        fact = bitemporal.record_fact(
            subject=f"{subject_prefix}:{entry.id}",
            predicate=entry.tier.value,
            value=entry.content,
            valid_from=entry.created_at,
            recorded_from=entry.created_at,
            source="episodic_adapter",
            tags=(subject_prefix, entry.tier.value),
        )
        created.append(fact)
    return created
