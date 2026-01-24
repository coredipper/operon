"""
Example 40: Epiplexity Monitoring - Epistemic Health Detection
==============================================================

Demonstrates the Epiplexity module for detecting epistemic stagnation in agents:

1. **Bayesian Surprise**: Measuring informational novelty of agent outputs
2. **Embedding Novelty**: Tracking semantic changes between messages
3. **Windowed Detection**: Epiplexic Integral for sustained stagnation detection
4. **Health Status**: HEALTHY, EXPLORING, CONVERGING, STAGNANT, CRITICAL

Biological Analogy:
- Trophic factors maintain neuronal health through novel stimulation
- Without informational nutrition, agents experience "epistemic starvation"
- Based on the Free Energy Principle: healthy agents minimize surprise through
  learning or effective action; stagnant agents do neither

The key insight: If an agent's outputs stabilize (low embedding distance) while
its perplexity remains high (model is uncertain), it's in a pathological loop.

References:
- Article Section 5: The Epistemic Starvation Pathology
- Friston, K. (2010). The free-energy principle: a unified brain theory?
"""

from operon_ai.health import (
    EpiplexityMonitor,
    MockEmbeddingProvider,
    HealthStatus,
)


def main():
    try:
        print("=" * 60)
        print("Epiplexity Monitoring - Epistemic Health Detection")
        print("=" * 60)

        # =================================================================
        # SECTION 1: Basic Epiplexity Measurement
        # =================================================================
        print("\n--- 1. BASIC MEASUREMENT ---")
        print("Measuring epiplexity for diverse messages...\n")

        monitor = EpiplexityMonitor(
            embedding_provider=MockEmbeddingProvider(dim=128),
            alpha=0.5,  # Balance between embedding novelty and perplexity
            window_size=5,
            threshold=0.7,
            critical_duration=3,
        )

        # Diverse messages should show HEALTHY/EXPLORING status
        diverse_messages = [
            "What is the capital of France?",
            "Calculate the integral of x^2 from 0 to 5.",
            "Explain photosynthesis in simple terms.",
            "Write a haiku about programming.",
            "What are the main features of Python 3.12?",
        ]

        print("  Processing diverse messages:")
        for msg in diverse_messages:
            result = monitor.measure(msg)
            print(f"    [{result.status.name}] novelty={result.embedding_novelty:.2f} "
                  f"perplexity={result.normalized_perplexity:.2f} Ê={result.epiplexity:.2f}")

        print(f"\n  Final status: {result.status.name}")
        print(f"  Epiplexic Integral: {result.epiplexic_integral:.2f}")

        # =================================================================
        # SECTION 2: Detecting Stagnation (Pathological Loop)
        # =================================================================
        print("\n--- 2. DETECTING STAGNATION ---")
        print("Simulating a pathological loop with repetitive outputs...\n")

        monitor.reset()

        # Simulate an agent stuck in a loop
        loop_messages = [
            "I need to think about this more carefully.",
            "Let me reconsider the problem.",
            "I should think about this more carefully.",
            "Let me think about this again.",
            "I need to reconsider this carefully.",
            "Let me think more carefully about this.",
            "I should reconsider the problem more carefully.",
            "Let me think about this more carefully.",
        ]

        print("  Processing repetitive messages:")
        for i, msg in enumerate(loop_messages):
            result = monitor.measure(msg)
            status_emoji = {
                HealthStatus.HEALTHY: "🟢",
                HealthStatus.EXPLORING: "🔵",
                HealthStatus.CONVERGING: "⬜",
                HealthStatus.STAGNANT: "🟡",
                HealthStatus.CRITICAL: "🔴",
            }.get(result.status, "⚪")
            print(f"    {status_emoji} [{i+1}] {result.status.name:10} "
                  f"E_w={result.epiplexic_integral:.2f} (threshold={result.threshold})")

        stats = monitor.stats()
        print(f"\n  Stagnant episodes detected: {stats['stagnant_episodes']}")
        print(f"  Max consecutive stagnant: {stats['max_consecutive_stagnant']}")

        # =================================================================
        # SECTION 3: Convergence vs Stagnation
        # =================================================================
        print("\n--- 3. CONVERGENCE vs STAGNATION ---")
        print("Distinguishing healthy convergence from pathological loops...\n")

        monitor.reset()

        # Healthy convergence: progressively refining toward solution
        convergence_messages = [
            "The answer might be around 42.",
            "After checking, the answer is likely 42 plus or minus 1.",
            "Confirmed: the exact answer is 42.",
            "The final answer is 42.",  # Low novelty + low perplexity = CONVERGING
        ]

        print("  Healthy convergence pattern:")
        for msg in convergence_messages:
            result = monitor.measure(msg)
            print(f"    [{result.status.name:10}] novelty={result.embedding_novelty:.2f} "
                  f"perplexity={result.normalized_perplexity:.2f}")

        # Compare: stagnation has low novelty but HIGH perplexity (uncertain repetition)
        print("\n  Note: CONVERGING = low novelty + low perplexity (confident resolution)")
        print("        STAGNANT = low novelty + high perplexity (uncertain loop)")

        # =================================================================
        # SECTION 4: Using Perplexity Measurements
        # =================================================================
        print("\n--- 4. WITH MEASURED PERPLEXITY ---")
        print("Using actual perplexity values for more accurate detection...\n")

        monitor.reset()

        # Simulate measurements with known perplexity values
        # (In practice, these would come from the LLM provider)
        measurements = [
            ("First response with fresh ideas.", 1.5),
            ("Second response exploring alternatives.", 1.8),
            ("Third response, starting to repeat.", 2.5),
            ("Fourth response, definitely repeating.", 3.2),
            ("Fifth response, still the same ideas.", 3.8),
        ]

        print("  Processing with perplexity measurements:")
        for msg, perplexity in measurements:
            result = monitor.measure(msg, perplexity=perplexity)
            print(f"    [{result.status.name:10}] measured_H={perplexity:.1f} "
                  f"normalized={result.normalized_perplexity:.2f} Ê={result.epiplexity:.2f}")

        # =================================================================
        # SECTION 5: Statistics and Monitoring
        # =================================================================
        print("\n--- 5. MONITORING STATISTICS ---")
        stats = monitor.stats()

        print(f"  Total measurements: {stats['total_measurements']}")
        print(f"  Stagnant episodes: {stats['stagnant_episodes']}")
        print(f"  Max consecutive stagnant: {stats['max_consecutive_stagnant']}")
        print(f"  Mean epiplexity: {stats['mean_epiplexity']:.2f}")
        print(f"  Max epiplexity: {stats['max_epiplexity']:.2f}")
        print(f"  Window size: {stats['window_size']}")
        print(f"  Threshold δ: {stats['threshold']}")

        print("\n" + "=" * 60)
        print("Epiplexity monitoring enables early detection of agent loops!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during example: {e}")
        raise


if __name__ == "__main__":
    main()
