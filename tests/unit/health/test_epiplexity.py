import pytest
import math

from operon_ai.health.epiplexity import (
    MockEmbeddingProvider,
    EpiplexityMonitor,
    HealthStatus,
    DistanceProvider,
)


class DummyDistanceProvider(DistanceProvider):
    def distance(self, a: str, b: str) -> float:
        # Simple distance: 0.0 if same, 1.0 if different
        return 0.0 if a == b else 1.0


class DummyLowDistanceProvider(DistanceProvider):
    def distance(self, a: str, b: str) -> float:
        return 0.1  # Low novelty


def test_mock_embedding_provider():
    dim = 64
    provider = MockEmbeddingProvider(dim=dim)
    emb1 = provider.embed("hello world")
    emb2 = provider.embed("hello world")
    emb3 = provider.embed("different text")

    # Deterministic
    assert emb1 == emb2
    assert emb1 != emb3

    # Dimensionality
    assert len(emb1) == dim

    # Normalization (magnitude is ~1.0)
    mag1 = math.sqrt(sum(x * x for x in emb1))
    assert pytest.approx(mag1, 0.0001) == 1.0


def test_cosine_similarity():
    monitor = EpiplexityMonitor()
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    v3 = [1.0, 0.0, 0.0]

    # Orthogonal
    assert pytest.approx(monitor._cosine_similarity(v1, v2), 0.0001) == 0.0
    # Identical
    assert pytest.approx(monitor._cosine_similarity(v1, v3), 0.0001) == 1.0
    # Opposite
    v4 = [-1.0, 0.0, 0.0]
    assert pytest.approx(monitor._cosine_similarity(v1, v4), 0.0001) == -1.0


def test_exponential_saturation():
    monitor = EpiplexityMonitor(perplexity_h0=2.0)
    assert monitor._exponential_saturation(0.0) == 0.0

    # As h approaches infinity, approaches 1.0
    assert monitor._exponential_saturation(100.0) > 0.99

    # Value at h0
    assert pytest.approx(monitor._exponential_saturation(2.0), 0.01) == 1.0 - math.exp(-1.0)


def test_monitor_initial_state():
    monitor = EpiplexityMonitor(embedding_provider=MockEmbeddingProvider())
    result = monitor.measure("Initial message")

    # First message should have max novelty and explore
    assert result.embedding_novelty == 1.0
    assert result.status == HealthStatus.EXPLORING
    assert result.is_healthy is True


def test_monitor_stagnation_and_critical():
    monitor = EpiplexityMonitor(
        embedding_provider=MockEmbeddingProvider(),
        window_size=3,
        threshold=0.8,
        critical_duration=3,
    )

    # First message: EXPLORING
    result = monitor.measure("Repeated message", perplexity=2.0)
    assert result.status == HealthStatus.EXPLORING

    # Repeating same message with high perplexity -> STAGNANT -> CRITICAL
    status_history = []
    for _ in range(10):
        result = monitor.measure("Repeated message", perplexity=2.0)
        status_history.append(result.status)

    # Should eventually reach STAGNANT, then CRITICAL
    assert HealthStatus.STAGNANT in status_history
    assert HealthStatus.CRITICAL in status_history
    assert result.is_healthy is False


def test_monitor_converging():
    monitor = EpiplexityMonitor(embedding_provider=MockEmbeddingProvider())

    # To get converging, we need low novelty and low perplexity.
    # Note: Using MockEmbeddingProvider with the exact same message triggers 0 novelty eventually.
    # We call measure 3 times so the history populates and the state stabilizes.

    monitor.measure("Message A", perplexity=0.1)
    monitor.measure("Message A", perplexity=0.1)
    result = monitor.measure("Message A", perplexity=0.1)

    assert result.status == HealthStatus.CONVERGING
    assert result.is_healthy is True


def test_monitor_distance_provider():
    provider = DummyDistanceProvider()
    monitor = EpiplexityMonitor(distance_provider=provider, threshold=0.8)

    # First item
    result = monitor.measure(item="Item A")
    assert result.status == HealthStatus.EXPLORING

    # Different item (distance 1.0) -> high novelty -> EXPLORING
    result = monitor.measure(item="Item B")
    assert result.status == HealthStatus.EXPLORING

    # Same item (distance 0.0) -> low novelty
    # Repeating leads to stagnation
    for _ in range(10):
        # We don't provide perplexity so it's approx 0.5+
        result = monitor.measure(item="Item B")

    assert result.status in (HealthStatus.STAGNANT, HealthStatus.CRITICAL)


def test_monitor_distance_provider_stagnant():
    provider = DummyLowDistanceProvider()
    # High threshold to force stagnation fast
    monitor = EpiplexityMonitor(distance_provider=provider, window_size=2, threshold=0.9, alpha=0.9)

    monitor.measure(item="Item A")
    result = monitor.measure(item="Item B")

    # Second item has distance 0.1. Alpha is 0.9. Epiplexity approx 0.09.
    # Below threshold 0.9.
    assert result.status == HealthStatus.STAGNANT


def test_monitor_stats():
    monitor = EpiplexityMonitor(embedding_provider=MockEmbeddingProvider())
    monitor.measure("Msg1")
    monitor.measure("Msg2")
    stats = monitor.stats()

    assert stats["total_measurements"] == 2
    assert "mean_epiplexity" in stats
    assert stats["window_size"] == monitor.window_size
