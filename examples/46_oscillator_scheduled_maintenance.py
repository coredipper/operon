#!/usr/bin/env python3
"""
Example 46: Oscillator-Scheduled Maintenance
=============================================

Demonstrates how to combine Oscillator + Autophagy + NegativeFeedbackLoop
to build a long-running agent service that periodically prunes context,
runs health checks, and self-corrects via feedback loop.

Architecture:

```
 [HeartbeatOscillator] ──fast──> ContextHealthMonitor
       |                            ├── context fill %
       |                            └── noise ratio
       |
 [SlowOscillator]  ──slow──> AutophagyDaemon
       |                       ├── prune stale context
       |                       └── Lysosome disposes waste
       |
 [NegativeFeedbackLoop]
       ├── setpoint: noise_ratio = 0.15
       ├── measure: actual noise ratio
       └── adjust: autophagy toxicity threshold
```

Key concepts:
- Fast heartbeat cycle: rapid health checks (context fill %, noise ratio)
- Slow maintenance cycle: autophagy prunes context, lysosome disposes waste
- Negative feedback loop: maintains noise ratio around target setpoint
  by adjusting autophagy toxicity threshold

Note: We do NOT call start()/stop() on oscillators (they spawn threads).
Instead, we compute phases mathematically and simulate cycles in a loop.

Prerequisites:
- Example 31 for Oscillator concepts
- Example 39 for Autophagy concepts
- Example 02 for Negative Feedback Loop

Usage:
    python examples/46_oscillator_scheduled_maintenance.py
    python examples/46_oscillator_scheduled_maintenance.py --test
"""

import sys
import math
from dataclasses import dataclass, field

from operon_ai import (
    ATP_Store,
    HistoneStore,
    Lysosome,
    Waste,
    WasteType,
)
from operon_ai.healing import (
    AutophagyDaemon,
    create_simple_summarizer,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class HealthSnapshot:
    """Result of a single health check."""
    cycle: int
    phase: str
    context_fill_pct: float
    noise_ratio: float
    amplitude: float
    is_healthy: bool


@dataclass
class MaintenanceResult:
    """Result of a maintenance cycle."""
    cycle: int
    bytes_pruned: int
    waste_disposed: int
    noise_before: float
    noise_after: float
    threshold_adjusted: float


# =============================================================================
# Context Health Monitor (fast heartbeat)
# =============================================================================


class ContextHealthMonitor:
    """
    Monitors agent context health at heartbeat frequency.

    Checks:
    - Context fill %: how much of the token budget is used
    - Noise ratio: fraction of context that is low-value noise
    """

    def __init__(self, max_tokens: int = 8000, silent: bool = False):
        self.max_tokens = max_tokens
        self.silent = silent
        self._history: list[HealthSnapshot] = []

    def check(
        self,
        context: str,
        cycle: int,
        phase: str,
        amplitude: float,
    ) -> HealthSnapshot:
        """Run a health check on the current context."""
        # Estimate token count (rough: 1 token ~ 4 chars)
        estimated_tokens = len(context) // 4
        fill_pct = min(1.0, estimated_tokens / self.max_tokens)

        # Estimate noise ratio: lines that look like filler/noise
        lines = context.split("\n")
        if lines:
            noise_lines = sum(
                1 for line in lines
                if self._is_noise(line)
            )
            noise_ratio = noise_lines / len(lines)
        else:
            noise_ratio = 0.0

        is_healthy = fill_pct < 0.8 and noise_ratio < 0.3

        snapshot = HealthSnapshot(
            cycle=cycle,
            phase=phase,
            context_fill_pct=fill_pct,
            noise_ratio=noise_ratio,
            amplitude=amplitude,
            is_healthy=is_healthy,
        )
        self._history.append(snapshot)

        if not self.silent:
            status = "OK" if is_healthy else "WARN"
            print(
                f"  [Heartbeat {cycle}] phase={phase} "
                f"fill={fill_pct:.0%} noise={noise_ratio:.0%} [{status}]"
            )

        return snapshot

    @staticmethod
    def _is_noise(line: str) -> bool:
        """Heuristic: is this line noise?"""
        stripped = line.strip()
        if not stripped:
            return False
        # Repeated chars
        if len(set(stripped)) <= 2 and len(stripped) > 5:
            return True
        # Filler tokens
        if stripped.lower() in ("...", "---", "***", "===", "thinking..."):
            return True
        # Repeated words
        words = stripped.split()
        if len(words) > 3 and len(set(words)) == 1:
            return True
        return False

    def get_latest_noise_ratio(self) -> float:
        """Get most recent noise ratio."""
        if self._history:
            return self._history[-1].noise_ratio
        return 0.0


# =============================================================================
# Maintenance Scheduler (slow oscillator)
# =============================================================================


class MaintenanceScheduler:
    """
    Runs autophagy and cleanup on a slow oscillator cycle.

    The maintenance intensity is modulated by the oscillator amplitude:
    - High amplitude: aggressive pruning
    - Low amplitude: gentle pruning
    """

    def __init__(
        self,
        histone_store: HistoneStore,
        lysosome: Lysosome,
        toxicity_threshold: float = 0.3,
        silent: bool = False,
    ):
        self.histone_store = histone_store
        self.lysosome = lysosome
        self.toxicity_threshold = toxicity_threshold
        self.silent = silent

        self.autophagy = AutophagyDaemon(
            histone_store=histone_store,
            lysosome=lysosome,
            summarizer=create_simple_summarizer(),
            toxicity_threshold=0.8,
            silent=silent,
        )
        self._history: list[MaintenanceResult] = []

    def run_maintenance(
        self,
        context: str,
        cycle: int,
        amplitude: float,
        noise_ratio: float,
    ) -> tuple[str, MaintenanceResult]:
        """
        Run a maintenance cycle.

        Args:
            context: Current agent context
            cycle: Cycle number
            amplitude: Oscillator amplitude (modulates pruning aggressiveness)
            noise_ratio: Current noise ratio from health monitor

        Returns:
            (pruned_context, MaintenanceResult)
        """
        # Adjust max_token_ratio based on amplitude
        # Higher amplitude = more aggressive pruning (lower ratio threshold)
        adjusted_ratio = max(0.3, 0.8 - amplitude * 0.3)

        if not self.silent:
            print(
                f"  [Maintenance {cycle}] "
                f"amplitude={amplitude:.2f} "
                f"threshold={adjusted_ratio:.2f} "
                f"noise={noise_ratio:.0%}"
            )

        # Run autophagy
        pruned_context, prune_result = self.autophagy.check_and_prune(
            context,
            max_tokens=8000,
        )

        # Dispose waste via lysosome
        if noise_ratio > self.toxicity_threshold:
            self.lysosome.ingest(Waste(
                waste_type=WasteType.FAILED_OPERATION,
                content=f"Noise ratio {noise_ratio:.0%} exceeded threshold {self.toxicity_threshold:.0%}",
                source="maintenance_scheduler",
            ))

        digest_result = self.lysosome.digest()

        # Calculate post-maintenance noise ratio
        lines = pruned_context.split("\n")
        post_noise = 0.0
        if lines:
            noise_lines = sum(
                1 for line in lines
                if self._is_noise_line(line)
            )
            post_noise = noise_lines / len(lines)

        tokens_freed = prune_result.tokens_freed if prune_result else 0

        result = MaintenanceResult(
            cycle=cycle,
            bytes_pruned=tokens_freed,
            waste_disposed=digest_result.disposed,
            noise_before=noise_ratio,
            noise_after=post_noise,
            threshold_adjusted=adjusted_ratio,
        )
        self._history.append(result)

        if not self.silent:
            print(
                f"  [Maintenance {cycle}] "
                f"freed={tokens_freed} tokens "
                f"waste={digest_result.disposed} "
                f"noise: {noise_ratio:.0%} -> {post_noise:.0%}"
            )

        return pruned_context, result

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        """Check if line is noise (same heuristic as ContextHealthMonitor)."""
        stripped = line.strip()
        if not stripped:
            return False
        # Repeated chars
        if len(set(stripped)) <= 2 and len(stripped) > 5:
            return True
        # Filler tokens
        if stripped.lower() in ("...", "---", "***", "===", "thinking..."):
            return True
        # Repeated words
        words = stripped.split()
        if len(words) > 3 and len(set(words)) == 1:
            return True
        return False


# =============================================================================
# Feedback Controller
# =============================================================================


class NoiseRatioController:
    """
    Negative feedback loop that maintains noise ratio around a setpoint.

    Adjusts the autophagy toxicity threshold based on the error between
    actual noise ratio and target setpoint.
    """

    def __init__(
        self,
        setpoint: float = 0.15,
        proportional_gain: float = 0.5,
        silent: bool = False,
    ):
        self.setpoint = setpoint
        self.gain = proportional_gain
        self.silent = silent
        self._adjustments: list[dict] = []

    def step(
        self,
        actual_noise_ratio: float,
        current_threshold: float,
    ) -> float:
        """
        Compute one feedback step.

        If noise > setpoint: lower the toxicity threshold (prune more)
        If noise < setpoint: raise the toxicity threshold (prune less)

        Returns:
            New toxicity threshold
        """
        error = actual_noise_ratio - self.setpoint
        adjustment = -error * self.gain  # Negative feedback
        new_threshold = max(0.05, min(0.8, current_threshold + adjustment))

        self._adjustments.append({
            "noise": actual_noise_ratio,
            "setpoint": self.setpoint,
            "error": error,
            "adjustment": adjustment,
            "new_threshold": new_threshold,
        })

        if not self.silent:
            direction = "tighter" if adjustment < 0 else "looser"
            print(
                f"  [Feedback] noise={actual_noise_ratio:.2f} "
                f"setpoint={self.setpoint:.2f} "
                f"error={error:+.2f} -> {direction} "
                f"threshold={new_threshold:.2f}"
            )

        return new_threshold


# =============================================================================
# Oscillator Simulation (no threads)
# =============================================================================


def compute_oscillator_phase(cycle: int, period: int) -> tuple[str, float]:
    """
    Compute oscillator phase mathematically.

    Returns (phase_name, amplitude) for the given cycle.
    """
    # Use sine wave: 0..1..0..-1..0
    t = (cycle % period) / period
    amplitude = abs(math.sin(2 * math.pi * t))

    # Phase names based on position in cycle
    if t < 0.25:
        phase = "rising"
    elif t < 0.5:
        phase = "peak"
    elif t < 0.75:
        phase = "falling"
    else:
        phase = "trough"

    return phase, amplitude


def run_maintenance_cycle(
    context: str,
    total_cycles: int,
    heartbeat_period: int,
    maintenance_period: int,
    health_monitor: ContextHealthMonitor,
    scheduler: MaintenanceScheduler,
    controller: NoiseRatioController,
    silent: bool = False,
) -> tuple[str, list[HealthSnapshot], list[MaintenanceResult]]:
    """
    Run a full maintenance simulation.

    Args:
        context: Initial context
        total_cycles: Number of cycles to simulate
        heartbeat_period: Heartbeat oscillator period
        maintenance_period: Maintenance oscillator period
        health_monitor: Health monitor instance
        scheduler: Maintenance scheduler instance
        controller: Feedback controller instance
        silent: Suppress output

    Returns:
        (final_context, health_snapshots, maintenance_results)
    """
    health_snapshots: list[HealthSnapshot] = []
    maintenance_results: list[MaintenanceResult] = []

    for cycle in range(total_cycles):
        # Fast heartbeat: health check every cycle
        hb_phase, hb_amplitude = compute_oscillator_phase(cycle, heartbeat_period)
        snapshot = health_monitor.check(context, cycle, hb_phase, hb_amplitude)
        health_snapshots.append(snapshot)

        # Slow maintenance: run every maintenance_period cycles
        if cycle > 0 and cycle % maintenance_period == 0:
            _mt_phase, mt_amplitude = compute_oscillator_phase(
                cycle // maintenance_period, maintenance_period
            )

            noise_ratio = health_monitor.get_latest_noise_ratio()

            # Run maintenance
            context, result = scheduler.run_maintenance(
                context, cycle, mt_amplitude, noise_ratio,
            )
            maintenance_results.append(result)

            # Feedback: adjust toxicity threshold
            new_threshold = controller.step(
                result.noise_after,
                scheduler.toxicity_threshold,
            )
            scheduler.toxicity_threshold = new_threshold

    return context, health_snapshots, maintenance_results


# =============================================================================
# Demo: Gradual Context Pollution
# =============================================================================


def demo_gradual_pollution():
    """
    Demo where context gradually accumulates noise.

    The maintenance scheduler detects rising noise and prunes it.
    The feedback loop adjusts the toxicity threshold to maintain homeostasis.
    """
    print("=" * 60)
    print("Demo 1: Gradual Context Pollution")
    print("=" * 60)

    # Build components
    histone_store = HistoneStore()
    lysosome = Lysosome(silent=True)
    health_monitor = ContextHealthMonitor(max_tokens=8000, silent=False)
    scheduler = MaintenanceScheduler(
        histone_store=histone_store,
        lysosome=lysosome,
        toxicity_threshold=0.3,
        silent=False,
    )
    controller = NoiseRatioController(setpoint=0.15, silent=False)

    # Build initial context with useful content
    useful_lines = [
        "User asked about Python best practices.",
        "Agent explained PEP 8 style guidelines.",
        "User asked about type hints in Python 3.12.",
        "Agent provided examples of TypeVar and ParamSpec.",
        "User asked about async/await patterns.",
        "Agent explained asyncio event loop.",
    ]

    # Gradually add noise over cycles
    noise_templates = [
        "thinking... " * 5,
        "=== " * 10,
        "processing processing processing processing",
        "... ... ... ... ...",
        "analyzing analyzing analyzing analyzing",
        "*** *** *** ***",
    ]

    context_lines = list(useful_lines)

    print("\nSimulating 20 cycles (heartbeat=4, maintenance=5)...")
    print("-" * 60)

    total_cycles = 20
    for cycle in range(total_cycles):
        # Every 2 cycles, add some noise
        if cycle % 2 == 0 and cycle > 0:
            noise_idx = (cycle // 2) % len(noise_templates)
            context_lines.append(noise_templates[noise_idx])

    context = "\n".join(context_lines)
    print(f"\nInitial context: {len(context_lines)} lines, {len(context)} chars")

    final_context, snapshots, maintenances = run_maintenance_cycle(
        context=context,
        total_cycles=total_cycles,
        heartbeat_period=4,
        maintenance_period=5,
        health_monitor=health_monitor,
        scheduler=scheduler,
        controller=controller,
    )

    print(f"\nFinal context: {len(final_context)} chars")
    print(f"Health checks: {len(snapshots)}")
    print(f"Maintenance runs: {len(maintenances)}")

    if snapshots:
        initial_noise = snapshots[0].noise_ratio
        final_noise = snapshots[-1].noise_ratio
        print(f"Noise ratio: {initial_noise:.0%} -> {final_noise:.0%}")

    return snapshots, maintenances


# =============================================================================
# Demo: Feedback Correction
# =============================================================================


def demo_feedback_correction():
    """
    Demo showing the feedback loop adjusting toxicity threshold.

    Starts with a high noise ratio and shows how the controller
    ratchets down the threshold to compensate.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Feedback Correction Loop")
    print("=" * 60)

    histone_store = HistoneStore()
    lysosome = Lysosome(silent=True)
    health_monitor = ContextHealthMonitor(max_tokens=8000, silent=False)
    scheduler = MaintenanceScheduler(
        histone_store=histone_store,
        lysosome=lysosome,
        toxicity_threshold=0.5,  # Start with loose threshold
        silent=False,
    )
    controller = NoiseRatioController(
        setpoint=0.10,  # Strict target
        proportional_gain=0.8,  # Aggressive correction
        silent=False,
    )

    # Build very noisy context
    context_lines = [
        "User asked about machine learning models.",
        "Agent explained gradient descent.",
    ]
    # Add lots of noise
    for i in range(15):
        context_lines.append(f"thinking... step {i} " * 3)
        context_lines.append("=== " * 8)

    context = "\n".join(context_lines)
    print(f"\nStarting with very noisy context: {len(context_lines)} lines")
    print(f"Initial toxicity threshold: {scheduler.toxicity_threshold}")
    print(f"Target noise ratio (setpoint): {controller.setpoint}")
    print("-" * 60)

    final_context, snapshots, maintenances = run_maintenance_cycle(
        context=context,
        total_cycles=15,
        heartbeat_period=3,
        maintenance_period=3,
        health_monitor=health_monitor,
        scheduler=scheduler,
        controller=controller,
    )

    print(f"\nFinal toxicity threshold: {scheduler.toxicity_threshold:.2f}")
    print(f"Threshold adjustments: {len(controller._adjustments)}")

    if controller._adjustments:
        print("\nThreshold progression:")
        for adj in controller._adjustments:
            print(
                f"  noise={adj['noise']:.2f} "
                f"error={adj['error']:+.2f} "
                f"threshold={adj['new_threshold']:.2f}"
            )

    return snapshots, maintenances


# =============================================================================
# Smoke Test
# =============================================================================


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Oscillator phase computation
    phase, amplitude = compute_oscillator_phase(0, 8)
    assert phase == "rising", f"Expected 'rising', got '{phase}'"
    assert amplitude >= 0.0, "Amplitude must be non-negative"
    print("  Test 1: Oscillator phase computation - PASSED")

    # Test 2: Health monitor
    monitor = ContextHealthMonitor(max_tokens=8000, silent=True)
    snapshot = monitor.check("Hello world", 0, "peak", 1.0)
    assert snapshot.context_fill_pct >= 0.0
    assert snapshot.noise_ratio >= 0.0
    assert snapshot.is_healthy
    print("  Test 2: Health monitor basic check - PASSED")

    # Test 3: Noise detection
    noisy_context = "\n".join([
        "Useful line 1",
        "thinking... " * 5,
        "=== " * 10,
        "Useful line 2",
        "processing processing processing processing",
    ])
    snapshot = monitor.check(noisy_context, 1, "peak", 1.0)
    assert snapshot.noise_ratio > 0, "Should detect some noise"
    print("  Test 3: Noise detection - PASSED")

    # Test 4: Maintenance scheduler
    histone_store = HistoneStore()
    lysosome = Lysosome(silent=True)
    scheduler = MaintenanceScheduler(
        histone_store=histone_store,
        lysosome=lysosome,
        toxicity_threshold=0.3,
        silent=True,
    )
    pruned, result = scheduler.run_maintenance(noisy_context, 1, 0.5, 0.4)
    assert isinstance(result, MaintenanceResult)
    assert result.cycle == 1
    print("  Test 4: Maintenance scheduler - PASSED")

    # Test 5: Feedback controller
    controller = NoiseRatioController(setpoint=0.15, silent=True)
    new_threshold = controller.step(0.4, 0.3)  # Noise too high
    assert new_threshold < 0.3, "Should tighten threshold when noise is high"
    new_threshold = controller.step(0.05, 0.3)  # Noise too low
    assert new_threshold > 0.3, "Should loosen threshold when noise is low"
    print("  Test 5: Feedback controller - PASSED")

    # Test 6: Full maintenance cycle
    context = "\n".join(["Useful content"] * 5 + ["thinking... " * 5] * 3)
    histone2 = HistoneStore()
    lyso2 = Lysosome(silent=True)
    monitor2 = ContextHealthMonitor(max_tokens=8000, silent=True)
    sched2 = MaintenanceScheduler(
        histone_store=histone2,
        lysosome=lyso2,
        toxicity_threshold=0.3,
        silent=True,
    )
    ctrl2 = NoiseRatioController(setpoint=0.15, silent=True)

    final, snapshots, maintenances = run_maintenance_cycle(
        context=context,
        total_cycles=12,
        heartbeat_period=3,
        maintenance_period=4,
        health_monitor=monitor2,
        scheduler=sched2,
        controller=ctrl2,
        silent=True,
    )
    assert len(snapshots) == 12, f"Expected 12 snapshots, got {len(snapshots)}"
    assert len(maintenances) >= 2, f"Expected >= 2 maintenance runs, got {len(maintenances)}"
    print("  Test 6: Full maintenance cycle - PASSED")

    print("\nSmoke tests passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 46: Oscillator-Scheduled Maintenance")
    print("Oscillator + Autophagy + NegativeFeedbackLoop")
    print("=" * 60)

    demo_gradual_pollution()
    demo_feedback_correction()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Oscillator-Scheduled Maintenance pattern combines three mechanisms:

1. HeartbeatOscillator (fast): Rapid health checks - measure context
   fill %, noise ratio at every cycle.

2. Slow Oscillator (maintenance): Periodic autophagy prunes stale
   context, lysosome disposes waste.

3. NegativeFeedbackLoop: Maintains noise ratio around target setpoint
   by adjusting the autophagy toxicity threshold.

Key insight: The feedback loop creates homeostasis - the system
automatically adapts its pruning aggressiveness based on how noisy
the context actually is, rather than using a fixed threshold.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
