"""
Memory: Epigenetic and Bi-Temporal Memory for AI Agents
=======================================================

Provides a three-tier memory system inspired by biological memory:
- Working Memory: Short-term, fast decay
- Episodic Memory: Medium-term, learns from feedback
- Long-term Memory: Persistent, no decay

And a bi-temporal factual memory for tracking world-time vs system-time:
- BiTemporalMemory: Append-only fact store with dual time axes
"""

from .episodic import (
    MemoryTier,
    MemoryEntry,
    EpisodicMemory,
)
from .bitemporal import (
    BiTemporalFact,
    BiTemporalQuery,
    BiTemporalMemory,
    FactSnapshot,
    CorrectionResult,
)
from .adapters import (
    histone_to_bitemporal,
    episodic_to_bitemporal,
)

__all__ = [
    "MemoryTier",
    "MemoryEntry",
    "EpisodicMemory",
    "BiTemporalFact",
    "BiTemporalQuery",
    "BiTemporalMemory",
    "FactSnapshot",
    "CorrectionResult",
    "histone_to_bitemporal",
    "episodic_to_bitemporal",
]
