import pytest
from datetime import datetime, timezone

from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength
from operon_ai.organelles.lysosome import Lysosome, Waste, WasteType
from operon_ai.healing.autophagy_daemon import (
    AutophagyDaemon,
    ContextHealthStatus,
    ContextMetrics,
    PruneResult,
    create_simple_summarizer,
)

class MockHistoneStore(HistoneStore):
    def __init__(self):
        super().__init__()
        self.markers = []

    def add_marker(self, content: str, marker_type: MarkerType, strength: MarkerStrength, tags: list[str], context: str) -> str:
        self.markers.append({
            "content": content,
            "marker_type": marker_type,
            "strength": strength,
            "tags": tags,
            "context": context
        })
        return f"mock_hash_{len(self.markers)}"

class MockLysosome(Lysosome):
    def __init__(self):
        # Passing mock or omitting args if Lysosome doesn't require init args for this test
        self.wastes = []

    def ingest(self, waste: Waste) -> None:
        self.wastes.append(waste)

@pytest.fixture
def mock_histone_store():
    return MockHistoneStore()

@pytest.fixture
def mock_lysosome():
    return MockLysosome()

@pytest.fixture
def simple_summarizer():
    return create_simple_summarizer(max_summary_lines=4)

@pytest.fixture
def daemon(mock_histone_store, mock_lysosome, simple_summarizer):
    return AutophagyDaemon(
        histone_store=mock_histone_store,
        lysosome=mock_lysosome,
        summarizer=simple_summarizer,
        toxicity_threshold=0.8,
        warning_threshold=0.6,
        min_tokens_for_pruning=100,  # lowered for easier testing
        tokens_per_char=0.25,
        silent=True
    )

def test_estimate_tokens(daemon):
    text = "12345678"  # 8 chars
    assert daemon.estimate_tokens(text) == 2  # 8 * 0.25 = 2

def test_assess_health(daemon):
    max_tokens = 100
    # tokens_per_char = 0.25 -> 4 chars = 1 token

    # Healthy (< 0.6 fill, meaning < 60 tokens, meaning < 240 chars)
    healthy_context = "A" * 100  # 25 tokens -> 0.25 fill
    metrics = daemon.assess_health(healthy_context, max_tokens)
    assert metrics.status == ContextHealthStatus.HEALTHY
    assert metrics.fill_percentage == 0.25

    # Accumulating (>= 0.6 and < 0.8 fill, meaning >= 60 and < 80 tokens, meaning >= 240 and < 320 chars)
    accumulating_context = "A" * 260  # 65 tokens -> 0.65 fill
    metrics = daemon.assess_health(accumulating_context, max_tokens)
    assert metrics.status == ContextHealthStatus.ACCUMULATING
    assert metrics.fill_percentage == 0.65

    # Critical (>= 0.8 fill, meaning >= 80 tokens, meaning >= 320 chars)
    critical_context = "A" * 340  # 85 tokens -> 0.85 fill
    metrics = daemon.assess_health(critical_context, max_tokens)
    assert metrics.status == ContextHealthStatus.CRITICAL
    assert metrics.fill_percentage == 0.85

def test_assess_health_useful_ratio(daemon):
    context = "Line 1\nLine 2\nError: something failed\nLine 4"
    metrics = daemon.assess_health(context, 100)
    assert metrics.useful_content_ratio == 0.75  # 3/4 useful lines

def test_check_and_prune_no_action(daemon):
    # Context too small (< 100 tokens -> < 400 chars)
    small_context = "A" * 100 # 25 tokens
    # Should not prune because it's below min_tokens_for_pruning
    new_context, result = daemon.check_and_prune(small_context, max_tokens=50)
    assert result is None
    assert new_context == small_context

def test_check_and_prune_triggers(daemon):
    max_tokens = 200 # 800 chars

    # Trigger on critical status
    critical_context = "A" * 700 # 175 tokens -> 0.875 fill (Critical)
    _, result = daemon.check_and_prune(critical_context, max_tokens)
    assert result is not None
    assert result.pruned is True

    # Trigger on accumulating status with low useful ratio
    accumulating_context = "A\n" * 300 # 300 chars, but lots of newlines. Wait, tokens_per_char doesn't care about newlines. 600 chars total -> 150 tokens -> 0.75 fill.
    accumulating_context = "Error: 1\nError: 2\nError: 3\nError: 4\nGood: 1" # ratio 1/5 = 0.2 < 0.5.
    # Pad to make sure tokens >= 100
    padded = accumulating_context + "A" * 400 # ~110 tokens -> 0.55 fill, Wait, warning threshold is 0.6. Need more.
    accumulating_noise_context = "Error:\n" * 50 + "A" * 200 # 50 noise lines + 1 big line. ratio ~ 1/51. 350 + 200 = 550 chars -> 137 tokens -> ~0.68 fill (Accumulating).
    _, result = daemon.check_and_prune(accumulating_noise_context, max_tokens)
    assert result is not None
    assert result.pruned is True

    # Trigger on force
    healthy_context = "A" * 400 # 100 tokens -> 0.5 fill (Healthy)
    _, result = daemon.check_and_prune(healthy_context, max_tokens, force=True)
    assert result is not None
    assert result.pruned is True

def test_check_and_prune_execution(daemon, mock_histone_store, mock_lysosome):
    context = "Line 1\nLine 2\nLine 3\nLine 4\n" + "A\n" * 6000 # > 150 tokens
    max_tokens = 200

    new_context, result = daemon.check_and_prune(context, max_tokens, force=True)

    # Check result
    assert result is not None
    assert result.pruned is True
    assert result.tokens_freed > 0
    assert result.waste_items_flushed == 1

    # Check dependencies were called
    assert len(mock_histone_store.markers) == 1
    assert mock_histone_store.markers[0]["marker_type"] == MarkerType.ACETYLATION
    assert len(mock_lysosome.wastes) == 1
    assert mock_lysosome.wastes[0].waste_type == WasteType.EXPIRED_CACHE

    # Check new context structure
    assert "[Context consolidated via autophagy]" in new_context
    assert "[End of consolidated context]" in new_context

def test_stats(daemon):
    assert daemon.stats()["prune_count"] == 0

    context = "A\n" * 5000
    daemon.check_and_prune(context, 100, force=True)

    stats = daemon.stats()
    assert stats["prune_count"] == 1
    assert stats["total_tokens_freed"] > 0
    assert stats["last_check"] is not None

def test_simple_summarizer_extracts_useful_lines():
    summarizer = create_simple_summarizer(max_summary_lines=4)
    context = "Useful 1\nError: something\nUseful 2\nFailed: another\nUseful 3"
    summary = summarizer(context)
    assert "Useful 1" in summary
    assert "Useful 2" in summary
    assert "Useful 3" in summary
    assert "Error" not in summary
    assert "Failed" not in summary

def test_simple_summarizer_truncates_long_context():
    summarizer = create_simple_summarizer(max_summary_lines=4)
    context = "L1\nL2\nL3\nL4\nL5\nL6"
    summary = summarizer(context)
    lines = summary.split("\n")
    assert len(lines) == 4
    assert lines[0] == "L1"
    assert lines[1] == "L2"
    assert lines[2] == "L5"
    assert lines[3] == "L6"
