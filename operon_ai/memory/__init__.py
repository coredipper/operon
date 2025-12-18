"""
Memory: Epigenetic Memory Hierarchy for AI Agents
=================================================

Provides a three-tier memory system inspired by biological memory:
- Working Memory: Short-term, fast decay
- Episodic Memory: Medium-term, learns from feedback
- Long-term Memory: Persistent, no decay
"""

from .episodic import (
    MemoryTier,
    MemoryEntry,
    EpisodicMemory,
)

__all__ = [
    "MemoryTier",
    "MemoryEntry",
    "EpisodicMemory",
]
