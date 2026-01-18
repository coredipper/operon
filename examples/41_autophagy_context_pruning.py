#!/usr/bin/env python3
"""
Example 41: Autophagy Context Pruning (Cognitive Self-Healing)
=============================================================

Demonstrates the biological Autophagy pattern where context windows are
proactively pruned to prevent pollution and maintain performance.

Key concepts:
- Context health monitoring: Track fill percentage and noise ratio
- Sleep/wake cycles: Pause to consolidate important information
- Long-term memory storage: Preserve summaries in HistoneStore
- Waste disposal: Flush raw context via Lysosome

Biological parallel:
- Autophagy: Cells digesting accumulated waste to recycle components
- Glymphatic system: Brain clearing metabolic waste during sleep
- Synaptic pruning: Eliminating unused connections to maintain efficiency

The key insight: Context pollution degrades performance gradually.
Proactive cleanup prevents catastrophic degradation.

Prerequisites:
- Example 14 for epigenetic memory patterns
- Example 11 for Lysosome cleanup patterns

See Also:
- operon_ai/healing/autophagy_daemon.py for the core implementation
- Article Section 5.5: Homeostasis - Cognitive Healing

Usage:
    python examples/41_autophagy_context_pruning.py
    python examples/41_autophagy_context_pruning.py --test
"""

import sys

from operon_ai import HistoneStore, Lysosome
from operon_ai.healing import (
    AutophagyDaemon,
    ContextHealthStatus,
    create_simple_summarizer,
)


# =============================================================================
# Context Generation Helpers
# =============================================================================

def generate_clean_context(size_chars: int) -> str:
    """Generate clean, useful context."""
    useful_content = [
        "Task: Analyze the quarterly sales data and identify trends.",
        "Step 1: Load data from database - COMPLETED",
        "Step 2: Clean and preprocess - COMPLETED",
        "Step 3: Calculate aggregates - IN PROGRESS",
        "Key finding: Q3 showed 15% growth over Q2",
        "Key finding: Product category A outperformed B by 23%",
        "Recommendation: Focus marketing on category A",
    ]
    result = []
    total = 0
    i = 0
    while total < size_chars:
        line = useful_content[i % len(useful_content)]
        result.append(line)
        total += len(line)
        i += 1
    return "\n".join(result)


def generate_polluted_context(size_chars: int, noise_ratio: float = 0.6) -> str:
    """
    Generate context with noise (errors, retries, verbose logs).

    Args:
        size_chars: Target size in characters
        noise_ratio: Fraction of content that is noise
    """
    useful = [
        "Task: Build the feature as specified.",
        "Step 1: Analyze requirements - DONE",
        "Key insight: Need to handle edge case X",
        "Implementation note: Use pattern Y for efficiency",
    ]

    noise = [
        "Error: Connection timeout. Retrying...",
        "Error: Connection timeout. Retrying...",
        "Error: Connection timeout. Retrying...",
        "Failed: Could not parse response",
        "Exception: ValueError in line 42",
        "Traceback (most recent call last):",
        "  File 'agent.py', line 100, in process",
        "    return self._handle_error(e)",
        "Retry attempt 1 of 3...",
        "Retry attempt 2 of 3...",
        "Retry attempt 3 of 3...",
        "I apologize, I cannot complete this request.",
        "Let me try a different approach...",
        "Actually, I need to reconsider...",
        "Debug: Entering function X",
        "Debug: Variable y = None",
        "Debug: Exiting function X",
    ]

    result = []
    total = 0
    i = 0

    while total < size_chars:
        if i % 10 < (noise_ratio * 10):
            line = noise[i % len(noise)]
        else:
            line = useful[i % len(useful)]
        result.append(line)
        total += len(line)
        i += 1

    return "\n".join(result)


# =============================================================================
# Demo: Health Assessment
# =============================================================================

def demo_health_assessment():
    """
    Demo showing context health assessment at different fill levels.
    """
    print("=" * 60)
    print("Demo 1: Context Health Assessment")
    print("=" * 60)

    daemon = AutophagyDaemon(
        histone_store=HistoneStore(silent=True),
        lysosome=Lysosome(silent=True),
        summarizer=create_simple_summarizer(),
        toxicity_threshold=0.8,
        warning_threshold=0.6,
        silent=True,
    )

    max_tokens = 8000  # Typical context window

    print("\nAssessing health at different context sizes:\n")

    for fill_pct in [0.3, 0.5, 0.65, 0.75, 0.85]:
        # Generate context of appropriate size
        target_chars = int((fill_pct * max_tokens) / daemon.tokens_per_char)
        context = generate_clean_context(target_chars)

        metrics = daemon.assess_health(context, max_tokens)

        status_emoji = {
            ContextHealthStatus.HEALTHY: "🟢",
            ContextHealthStatus.ACCUMULATING: "🟡",
            ContextHealthStatus.CRITICAL: "🔴",
            ContextHealthStatus.PRUNING: "⏳",
        }

        print(f"  {status_emoji[metrics.status]} {fill_pct:.0%} fill: "
              f"Status={metrics.status.value}, "
              f"Useful={metrics.useful_content_ratio:.0%}")

    return daemon


# =============================================================================
# Demo: Automatic Pruning
# =============================================================================

def demo_automatic_pruning():
    """
    Demo showing automatic pruning when context exceeds threshold.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Automatic Pruning (Sleep/Wake Cycle)")
    print("=" * 60)

    histone = HistoneStore(silent=True)
    lysosome = Lysosome(silent=True)

    daemon = AutophagyDaemon(
        histone_store=histone,
        lysosome=lysosome,
        summarizer=create_simple_summarizer(max_summary_lines=5),
        toxicity_threshold=0.8,
        warning_threshold=0.6,
    )

    max_tokens = 4000

    # Generate polluted context at 85% fill
    target_chars = int((0.85 * max_tokens) / daemon.tokens_per_char)
    polluted_context = generate_polluted_context(target_chars, noise_ratio=0.7)

    print(f"\nContext before pruning:")
    print(f"  Size: ~{daemon.estimate_tokens(polluted_context)} tokens")
    print(f"  Fill: {daemon.estimate_tokens(polluted_context) / max_tokens:.0%}")
    print(f"  First 200 chars: {polluted_context[:200]}...")

    # Run the autophagy check
    new_context, result = daemon.check_and_prune(polluted_context, max_tokens)

    if result:
        print(f"\nPrune Result:")
        print(f"  Tokens before: {result.tokens_before}")
        print(f"  Tokens after: {result.tokens_after}")
        print(f"  Tokens freed: {result.tokens_freed}")
        print(f"  Duration: {result.duration_ms:.1f}ms")

        print(f"\nNew context (first 300 chars):")
        print(f"  {new_context[:300]}...")

        print(f"\nSummary stored in long-term memory:")
        print(f"  {result.summary_stored[:200]}...")

        print(f"\nHistone markers added: {len(histone._markers)}")
        print(f"Lysosome waste items: {lysosome._total_ingested}")
    else:
        print("\nNo pruning needed")

    return daemon


# =============================================================================
# Demo: Threshold Comparison
# =============================================================================

def demo_threshold_comparison():
    """
    Demo comparing different toxicity thresholds.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Threshold Comparison")
    print("=" * 60)

    max_tokens = 4000
    target_chars = int((0.75 * max_tokens) / 0.25)  # 75% fill
    context = generate_polluted_context(target_chars, noise_ratio=0.5)

    print(f"\nTesting same context (75% fill) with different thresholds:\n")

    for threshold in [0.6, 0.7, 0.8, 0.9]:
        daemon = AutophagyDaemon(
            histone_store=HistoneStore(silent=True),
            lysosome=Lysosome(silent=True),
            summarizer=create_simple_summarizer(),
            toxicity_threshold=threshold,
            silent=True,
        )

        new_context, result = daemon.check_and_prune(context, max_tokens)
        pruned = "Yes" if result else "No"
        freed = result.tokens_freed if result else 0

        print(f"  Threshold {threshold:.0%}: Pruned={pruned}, Freed={freed} tokens")


# =============================================================================
# Demo: Accumulated Session
# =============================================================================

def demo_accumulated_session():
    """
    Demo simulating an agent session that accumulates context over time.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Accumulated Session with Periodic Pruning")
    print("=" * 60)

    histone = HistoneStore(silent=True)
    lysosome = Lysosome(silent=True)

    daemon = AutophagyDaemon(
        histone_store=histone,
        lysosome=lysosome,
        summarizer=create_simple_summarizer(max_summary_lines=3),
        toxicity_threshold=0.7,
        warning_threshold=0.5,
    )

    max_tokens = 2000
    context = ""

    print("\nSimulating 10 operations, each adding ~300 tokens:\n")

    for i in range(10):
        # Add new content (simulating agent operation)
        new_content = generate_polluted_context(1200, noise_ratio=0.4 + i * 0.05)
        context += f"\n--- Operation {i + 1} ---\n" + new_content

        # Check health
        metrics = daemon.assess_health(context, max_tokens)

        status_emoji = {
            ContextHealthStatus.HEALTHY: "🟢",
            ContextHealthStatus.ACCUMULATING: "🟡",
            ContextHealthStatus.CRITICAL: "🔴",
            ContextHealthStatus.PRUNING: "⏳",
        }

        print(f"  Op {i + 1}: {status_emoji[metrics.status]} "
              f"Fill={metrics.fill_percentage:.0%}, "
              f"Useful={metrics.useful_content_ratio:.0%}", end="")

        # Prune if needed
        context, result = daemon.check_and_prune(context, max_tokens)
        if result:
            print(f" → PRUNED (freed {result.tokens_freed} tokens)")
        else:
            print()

    print(f"\nSession Summary:")
    print(f"  Total prunes: {daemon._prune_count}")
    print(f"  Total tokens freed: {daemon._total_tokens_freed}")
    print(f"  Final context size: ~{daemon.estimate_tokens(context)} tokens")
    print(f"  Markers in long-term memory: {len(histone._markers)}")

    return daemon


# =============================================================================
# Demo: Noise vs Clean Content
# =============================================================================

def demo_noise_detection():
    """
    Demo showing how the daemon detects noisy vs clean context.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Noise Detection (Useful Content Ratio)")
    print("=" * 60)

    daemon = AutophagyDaemon(
        histone_store=HistoneStore(silent=True),
        lysosome=Lysosome(silent=True),
        summarizer=create_simple_summarizer(),
        toxicity_threshold=0.8,
        silent=True,
    )

    max_tokens = 4000
    target_chars = int((0.6 * max_tokens) / daemon.tokens_per_char)

    print("\nComparing contexts with same size but different noise levels:\n")

    for noise_ratio in [0.0, 0.3, 0.5, 0.7, 0.9]:
        context = generate_polluted_context(target_chars, noise_ratio=noise_ratio)
        metrics = daemon.assess_health(context, max_tokens)

        bar = "█" * int(metrics.useful_content_ratio * 20)
        bar_empty = "░" * (20 - len(bar))

        print(f"  Noise {noise_ratio:.0%}: Useful=[{bar}{bar_empty}] "
              f"{metrics.useful_content_ratio:.0%}")


# =============================================================================
# Smoke Test
# =============================================================================

def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke test...")

    # Test 1: Health assessment
    daemon = AutophagyDaemon(
        histone_store=HistoneStore(silent=True),
        lysosome=Lysosome(silent=True),
        summarizer=create_simple_summarizer(),
        toxicity_threshold=0.8,
        silent=True,
    )

    context = generate_clean_context(2000)
    metrics = daemon.assess_health(context, 8000)
    assert metrics.status == ContextHealthStatus.HEALTHY, f"Expected HEALTHY, got {metrics.status}"
    assert metrics.fill_percentage < 0.5, "Expected low fill"

    # Test 2: Automatic pruning
    large_context = generate_polluted_context(6000, noise_ratio=0.7)
    new_context, result = daemon.check_and_prune(large_context, 2000)
    assert result is not None, "Expected pruning to occur"
    assert result.tokens_freed > 0, "Expected tokens freed"
    assert len(new_context) < len(large_context), "Expected smaller context"

    # Test 3: No pruning when not needed
    small_context = generate_clean_context(500)
    new_context2, result2 = daemon.check_and_prune(small_context, 8000)
    assert result2 is None, "Expected no pruning"
    assert new_context2 == small_context, "Expected unchanged context"

    # Test 4: Stats tracking
    stats = daemon.stats()
    assert stats["prune_count"] >= 1, "Expected at least one prune recorded"
    assert stats["total_tokens_freed"] > 0, "Expected tokens freed recorded"

    print("Smoke test passed!")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 41: Autophagy Context Pruning")
    print("Cognitive Self-Healing through Sleep/Wake Cycles")
    print("=" * 60)

    demo_health_assessment()
    demo_automatic_pruning()
    demo_threshold_comparison()
    demo_accumulated_session()
    demo_noise_detection()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Autophagy Daemon demonstrates cognitive healing:

1. Health Monitoring: Track context fill and noise ratio
2. Sleep Cycles: Pause when toxicity threshold exceeded
3. Consolidation: Summarize useful state into long-term memory
4. Flush: Clear raw context via Lysosome
5. Wake: Resume with clean context + summary

Key biological parallel: During sleep, the brain's glymphatic system
clears metabolic waste that accumulates during waking hours. Without
this cleanup, cognitive function degrades.

Key software insight: Context windows accumulate "waste" (failed
attempts, verbose logs, error traces) that degrades LLM performance.
Proactive pruning maintains output quality over long sessions.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
