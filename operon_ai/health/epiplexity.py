"""
Epiplexity: Epistemic Health Monitoring via Bayesian Surprise
==============================================================

Biological Analogy:
- Trophic Factors: Growth signals that maintain neuronal health. Without novel
  stimulation, neurons atrophy. Similarly, agents need "informational nutrition"
  to maintain healthy reasoning.
- Free Energy Principle: Living systems minimize surprise while maintaining
  viability. An agent with no surprise is either perfectly adapted (unlikely)
  or epistemically stagnant (loop pathology).

The key insight: If an agent's outputs stabilize (low embedding distance) while
its perplexity remains high (model is uncertain), it's in a pathological loop.
Healthy agents either:
1. Converge (low perplexity) → task complete
2. Explore (high embedding novelty) → making progress

Epiplexity measures the *absence* of this healthy pattern.

References:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Article Section 5: The Epistemic Starvation Pathology
"""
from dataclasses import dataclass, field
from typing import Protocol, Sequence
from enum import Enum
from collections import deque
import math
import hashlib


class HealthStatus(Enum):
    """Epistemic health status."""
    HEALTHY = "healthy"           # Normal operation
    CONVERGING = "converging"     # Low epiplexity, likely task completion
    EXPLORING = "exploring"       # High novelty, making progress
    STAGNANT = "stagnant"         # High epiplexity, possible loop
    CRITICAL = "critical"         # Sustained high epiplexity, intervention needed


@dataclass
class EpiplexityResult:
    """Result of a single epiplexity measurement."""

    # Raw components
    embedding_novelty: float  # 1 - cos(e_t, e_{t-1}), range [0, 2]
    normalized_perplexity: float  # σ(H), range [0, 1]

    # Combined score
    epiplexity: float  # Ê_t, range [0, 1]

    # Windowed integral
    epiplexic_integral: float  # E_w = mean of window

    # Interpretation
    status: HealthStatus
    window_size: int
    threshold: float

    @property
    def is_healthy(self) -> bool:
        """Check if agent is in healthy epistemic state."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.CONVERGING, HealthStatus.EXPLORING)


@dataclass
class EpiplexityState:
    """Tracks epistemic state over time."""

    # Current embedding
    current_embedding: list[float] | None = None
    previous_embedding: list[float] | None = None

    # History for windowed analysis
    epiplexity_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # Statistics
    total_measurements: int = 0
    stagnant_episodes: int = 0
    max_consecutive_stagnant: int = 0
    current_consecutive_stagnant: int = 0


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Get embedding for text."""
        ...


class MockEmbeddingProvider:
    """
    Mock embedding provider for testing.

    Uses hash-based pseudo-embeddings that are deterministic
    but provide reasonable similarity structure.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embedding from text hash."""
        # Create deterministic seed from text
        text_hash = hashlib.sha256(text.encode()).digest()

        # Generate embedding components from hash
        embedding = []
        for i in range(self.dim):
            # Use different parts of hash for each dimension
            byte_idx = i % len(text_hash)
            val = (text_hash[byte_idx] + i * 17) % 256
            # Normalize to [-1, 1]
            embedding.append((val / 127.5) - 1.0)

        # Normalize to unit vector
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


@dataclass
class EpiplexityMonitor:
    """
    Monitor for detecting epistemic stagnation.

    Implements the operational Epiplexity approximation:

        Ê_t = α·(1 - cos(e_t, e_{t-1})) + (1-α)·σ(H(m_t|m_{<t}))

    Where:
    - e_t is the embedding of message t
    - cos is cosine similarity
    - σ is sigmoid normalization to [0,1]
    - H is conditional perplexity (approximated)
    - α is the mixing parameter

    The Epiplexic Integral for windowed detection:

        E_w = (1/w) Σ Ê_t  for t in window

    An agent is flagged when E_w > δ for sustained periods.

    Example:
        >>> monitor = EpiplexityMonitor(
        ...     embedding_provider=MockEmbeddingProvider(),
        ...     alpha=0.5,
        ...     window_size=5,
        ...     threshold=0.7,
        ... )
        >>> result = monitor.measure("First message")
        >>> result.status
        HealthStatus.HEALTHY
        >>> # Repeating same message creates stagnation
        >>> for _ in range(10):
        ...     result = monitor.measure("Repeating message")
        >>> result.status  # Will eventually become STAGNANT
    """

    embedding_provider: EmbeddingProvider
    alpha: float = 0.5  # Mixing parameter (0 = perplexity only, 1 = embedding only)
    window_size: int = 10  # Window for integral calculation
    threshold: float = 0.7  # δ threshold for stagnation detection
    critical_duration: int = 5  # Consecutive stagnant measurements before CRITICAL

    # Internal state
    state: EpiplexityState = field(default_factory=EpiplexityState)

    # Perplexity approximation parameters
    perplexity_baseline: float = 2.0  # Baseline perplexity for normalization
    perplexity_scale: float = 0.5  # Scale for sigmoid

    def measure(
        self,
        message: str,
        perplexity: float | None = None,
    ) -> EpiplexityResult:
        """
        Measure epiplexity for a new message.

        Args:
            message: The agent's output message
            perplexity: Optional measured perplexity. If not provided,
                       approximated from embedding stability.

        Returns:
            EpiplexityResult with current health status
        """
        # Get embedding
        embedding = self.embedding_provider.embed(message)

        # Calculate embedding novelty: 1 - cos(e_t, e_{t-1})
        if self.state.previous_embedding is not None:
            cosine_sim = self._cosine_similarity(embedding, self.state.previous_embedding)
            # Normalize from [-1, 1] to [0, 2], then to [0, 1]
            embedding_novelty = (1.0 - cosine_sim) / 2.0
        else:
            # First message - maximum novelty
            embedding_novelty = 1.0

        # Calculate normalized perplexity
        if perplexity is not None:
            # Use provided perplexity with sigmoid normalization
            normalized_perplexity = self._sigmoid(
                (perplexity - self.perplexity_baseline) * self.perplexity_scale
            )
        else:
            # Approximate: low novelty + repetition suggests high uncertainty
            # This is a heuristic when actual perplexity isn't available
            normalized_perplexity = self._approximate_perplexity(embedding, embedding_novelty)

        # Calculate combined Epiplexity score
        # Ê_t = α·novelty + (1-α)·perplexity
        epiplexity = (
            self.alpha * embedding_novelty +
            (1 - self.alpha) * normalized_perplexity
        )

        # Update state
        self.state.previous_embedding = self.state.current_embedding
        self.state.current_embedding = embedding
        self.state.epiplexity_history.append(epiplexity)
        self.state.total_measurements += 1

        # Calculate Epiplexic Integral (windowed mean)
        window = list(self.state.epiplexity_history)[-self.window_size:]
        epiplexic_integral = sum(window) / len(window) if window else 0.0

        # Determine health status
        status = self._determine_status(
            embedding_novelty,
            normalized_perplexity,
            epiplexic_integral,
        )

        # Track stagnation episodes
        if status in (HealthStatus.STAGNANT, HealthStatus.CRITICAL):
            self.state.current_consecutive_stagnant += 1
            if self.state.current_consecutive_stagnant > self.state.max_consecutive_stagnant:
                self.state.max_consecutive_stagnant = self.state.current_consecutive_stagnant
        else:
            if self.state.current_consecutive_stagnant > 0:
                self.state.stagnant_episodes += 1
            self.state.current_consecutive_stagnant = 0

        return EpiplexityResult(
            embedding_novelty=embedding_novelty,
            normalized_perplexity=normalized_perplexity,
            epiplexity=epiplexity,
            epiplexic_integral=epiplexic_integral,
            status=status,
            window_size=len(window),
            threshold=self.threshold,
        )

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for normalization."""
        return 1.0 / (1.0 + math.exp(-x))

    def _approximate_perplexity(
        self,
        embedding: Sequence[float],
        novelty: float,
    ) -> float:
        """
        Approximate perplexity when not directly available.

        Heuristic: If output is repetitive (low novelty) but embedding
        is still varying (not converged), model is likely uncertain.
        """
        if self.state.previous_embedding is None:
            return 0.5  # Neutral starting point

        # Check embedding stability (variance of differences)
        history_len = len(self.state.epiplexity_history)
        if history_len < 3:
            return 0.5

        # Use embedding magnitude variance as additional uncertainty signal
        magnitude = math.sqrt(sum(x * x for x in embedding))
        prev_magnitude = math.sqrt(sum(x * x for x in self.state.previous_embedding))
        magnitude_drift = abs(magnitude - prev_magnitude)

        # Recent trend: increasing epiplexity with low novelty suggests stagnation
        recent = list(self.state.epiplexity_history)[-3:]
        trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0

        # Low novelty with increasing trend = high approximate perplexity
        # Add magnitude drift as additional uncertainty indicator
        if novelty < 0.3 and trend > 0:
            return min(1.0, 0.5 + trend * 2 + magnitude_drift * 0.5)

        return 0.3 + novelty * 0.4 + magnitude_drift * 0.2

    def _determine_status(
        self,
        novelty: float,
        perplexity: float,
        integral: float,
    ) -> HealthStatus:
        """
        Determine health status from components.

        Decision logic:
        - High novelty (>0.5): EXPLORING (making progress)
        - Low novelty + low perplexity (<0.3): CONVERGING (task complete)
        - Low novelty + high perplexity: STAGNANT (loop)
        - Integral > threshold for critical_duration: CRITICAL
        """
        # Check for critical (sustained stagnation)
        if (
            integral > self.threshold and
            self.state.current_consecutive_stagnant >= self.critical_duration
        ):
            return HealthStatus.CRITICAL

        # High novelty = exploring
        if novelty > 0.5:
            return HealthStatus.EXPLORING

        # Low novelty + low perplexity = converging to solution
        if novelty < 0.3 and perplexity < 0.3:
            return HealthStatus.CONVERGING

        # Low novelty + high perplexity = stagnant
        if novelty < 0.3 and perplexity > 0.5:
            return HealthStatus.STAGNANT

        # Integral above threshold = stagnant
        if integral > self.threshold:
            return HealthStatus.STAGNANT

        return HealthStatus.HEALTHY

    def reset(self) -> None:
        """Reset monitor state."""
        self.state = EpiplexityState()

    def stats(self) -> dict:
        """Return monitoring statistics."""
        history = list(self.state.epiplexity_history)
        return {
            "total_measurements": self.state.total_measurements,
            "stagnant_episodes": self.state.stagnant_episodes,
            "max_consecutive_stagnant": self.state.max_consecutive_stagnant,
            "current_consecutive_stagnant": self.state.current_consecutive_stagnant,
            "mean_epiplexity": sum(history) / len(history) if history else 0.0,
            "max_epiplexity": max(history) if history else 0.0,
            "window_size": self.window_size,
            "threshold": self.threshold,
        }
