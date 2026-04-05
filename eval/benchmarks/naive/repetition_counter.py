"""Naive alternative to EpiplexityMonitor: cosine similarity + timeout.

This is a reasonable production-quality stagnation detector that many
agent frameworks use.  It checks whether recent outputs are too similar
(cosine similarity in a sliding window) and applies a fixed timeout.

It is NOT a strawman — cosine-similarity-based repetition detection is
the standard approach in real systems.  The question is whether the
Bayesian two-signal approach (novelty + perplexity) in EpiplexityMonitor
catches stagnation patterns that this simpler detector misses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

from operon_ai.health.epiplexity import EmbeddingProvider


@dataclass
class NaiveStagnationResult:
    """Result from the repetition counter."""

    status: str  # "healthy" or "stagnant"
    avg_similarity: float
    steps_since_change: int
    window_size: int


@dataclass
class RepetitionCounter:
    """Cosine-similarity-based stagnation detector.

    Maintains a sliding window of embeddings.  If the average pairwise
    cosine similarity exceeds a threshold, declares STAGNANT.
    Also applies a fixed timeout: if status hasn't changed in N steps,
    declares STAGNANT.

    Uses the same EmbeddingProvider as EpiplexityMonitor for fair comparison.
    """

    embedding_provider: EmbeddingProvider
    window_size: int = 10
    similarity_threshold: float = 0.85
    timeout_steps: int = 20

    _embeddings: deque[list[float]] = field(default_factory=lambda: deque(maxlen=100))
    _high_sim_steps: int = field(default=0, repr=False)
    _total: int = field(default=0, repr=False)

    def measure(self, message: str) -> NaiveStagnationResult:
        """Measure stagnation for a new message."""
        embedding = self.embedding_provider.embed(message)
        self._embeddings.append(embedding)
        self._total += 1

        window = list(self._embeddings)[-self.window_size:]
        avg_sim = self._average_pairwise_similarity(window)

        # Track consecutive steps with elevated similarity
        if avg_sim > self.similarity_threshold * 0.7:
            self._high_sim_steps += 1
        else:
            self._high_sim_steps = 0

        # Determine status
        if avg_sim > self.similarity_threshold:
            status = "stagnant"
        elif self._high_sim_steps >= self.timeout_steps:
            status = "stagnant"
        else:
            status = "healthy"

        return NaiveStagnationResult(
            status=status,
            avg_similarity=avg_sim,
            steps_since_change=self._high_sim_steps,
            window_size=len(window),
        )

    def reset(self) -> None:
        self._embeddings.clear()
        self._high_sim_steps = 0
        self._total = 0

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _average_pairwise_similarity(self, embeddings: list[list[float]]) -> float:
        """Average cosine similarity across all pairs in the window."""
        if len(embeddings) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                total += self._cosine_similarity(embeddings[i], embeddings[j])
                count += 1
        return total / count if count > 0 else 0.0
