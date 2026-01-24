"""
Health monitoring for epistemic state.

This module implements health metrics inspired by the Free Energy Principle,
measuring an agent's "epistemic health" through Bayesian Surprise.
"""

from .epiplexity import (
    EpiplexityMonitor,
    EpiplexityState,
    EpiplexityResult,
    HealthStatus,
    EmbeddingProvider,
    MockEmbeddingProvider,
)

__all__ = [
    "EpiplexityMonitor",
    "EpiplexityState",
    "EpiplexityResult",
    "HealthStatus",
    "EmbeddingProvider",
    "MockEmbeddingProvider",
]
