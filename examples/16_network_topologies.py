"""
Example 16: Network Topologies
==============================

This example demonstrates:
- Cascade: Signal amplification and multi-stage processing
- Oscillator: Periodic tasks and rhythm management
- Enhanced QuorumSensing: Weighted voting strategies

Biological Analogy:
- Cascade: MAPK signaling pathway with signal amplification
- Oscillator: Circadian rhythms and cell cycle timing
- Quorum: Bacterial quorum sensing for collective decisions
"""

import time
from operon_ai.state import ATP_Store
from operon_ai.topology import (
    Cascade,
    CascadeStage,
    HeartbeatOscillator,
    MAPKCascade,
    Oscillator,
    OscillatorPhase,
    QuorumSensing,
    VoteType,
    VotingStrategy,
    WaveformType,
)


def demonstrate_cascade_basics():
    """Demonstrate basic cascade operations."""
    print("\n" + "="*60)
    print("1. SIGNAL CASCADE: MULTI-STAGE PROCESSING")
    print("="*60)

    cascade = Cascade(
        name="DataProcessor",
        halt_on_failure=True,
        silent=True,
    )

    # Add processing stages
    cascade.add_stage(CascadeStage(
        name="validate",
        processor=lambda x: x if isinstance(x, str) and len(x) > 0 else None,
        checkpoint=lambda x: x is not None,
    ))

    cascade.add_stage(CascadeStage(
        name="normalize",
        processor=lambda x: x.strip().lower(),
        amplification=1.0,
    ))

    cascade.add_stage(CascadeStage(
        name="tokenize",
        processor=lambda x: x.split(),
        amplification=2.0,  # Tokens expand the signal
    ))

    cascade.add_stage(CascadeStage(
        name="filter",
        processor=lambda tokens: [t for t in tokens if len(t) > 2],
        amplification=0.5,  # Filtering reduces signal
    ))

    print("\n--- Running Cascade ---")
    result = cascade.run("  Hello World From Operon  ")

    print(f"\n--- Results ---")
    print(f"  Success: {result.success}")
    print(f"  Final Output: {result.final_output}")
    print(f"  Stages Completed: {result.stages_completed}/{result.stages_total}")
    print(f"  Total Amplification: {result.total_amplification:.2f}x")
    print(f"  Processing Time: {result.total_time_ms:.2f}ms")

    # Show stage details
    print("\n--- Stage Details ---")
    for stage in result.stage_results:
        print(f"  {stage.stage_name}: {stage.status.value} "
              f"(amp: {stage.amplification_factor:.1f}x, {stage.processing_time_ms:.1f}ms)")

    return cascade


def demonstrate_cascade_with_checkpoints():
    """Demonstrate cascade with checkpoint gates."""
    print("\n" + "="*60)
    print("2. CASCADE WITH CHECKPOINT GATES")
    print("="*60)

    cascade = Cascade(name="SecurityPipeline", silent=True)

    # Security validation stages
    cascade.add_stage(CascadeStage(
        name="input_sanitize",
        processor=lambda x: x.replace("<script>", ""),
        checkpoint=lambda x: "<script>" not in x,  # Block if script tag found
    ))

    cascade.add_stage(CascadeStage(
        name="length_check",
        processor=lambda x: x,
        checkpoint=lambda x: len(x) < 1000,  # Block if too long
    ))

    cascade.add_stage(CascadeStage(
        name="content_filter",
        processor=lambda x: x.upper(),  # Transform to uppercase
    ))

    # Test with safe input
    print("\n--- Safe Input ---")
    result = cascade.run("Hello, this is safe input")
    print(f"  Result: {result.final_output}")
    print(f"  Success: {result.success}")

    # Test with malicious input
    print("\n--- Malicious Input (contains script tag) ---")
    cascade_result = cascade.run("Hello <script>alert('xss')</script>")
    print(f"  Blocked: {cascade_result.blocked_at is not None}")
    print(f"  Blocked At: {cascade_result.blocked_at}")

    return cascade


def demonstrate_mapk_cascade():
    """Demonstrate MAPK-like signaling cascade."""
    print("\n" + "="*60)
    print("3. MAPK SIGNALING CASCADE")
    print("="*60)

    print("\n  Simulating: Growth Factor -> MAPKKK -> MAPKK -> MAPK -> Response")

    mapk = MAPKCascade(
        name="GrowthSignal",
        tier1_amplification=5.0,   # MAPKKK
        tier2_amplification=10.0,  # MAPKK
        tier3_amplification=2.0,   # MAPK
        silent=True,
    )

    result = mapk.run("growth_factor")

    print(f"\n--- Cascade Results ---")
    print(f"  Success: {result.success}")
    print(f"  Total Amplification: {result.total_amplification:.0f}x")

    if result.final_output:
        print(f"  Response: {result.final_output.get('response', 'N/A')}")
        print(f"  Signal Path: Tier {result.final_output.get('tier', 'N/A')}")

    return mapk


def demonstrate_oscillator_basics():
    """Demonstrate basic oscillator operations."""
    print("\n" + "="*60)
    print("4. OSCILLATOR: PERIODIC EXECUTION")
    print("="*60)

    tick_count = [0]

    def on_tick(value):
        tick_count[0] += 1

    cycle_count = [0]

    def on_cycle(result):
        cycle_count[0] += 1
        print(f"  Cycle {result.cycle_number + 1} completed "
              f"({result.duration_seconds:.2f}s)")

    # Create oscillator with 2Hz frequency (0.5s period)
    osc = Oscillator(
        frequency_hz=2.0,
        amplitude=1.0,
        waveform=WaveformType.SINE,
        max_cycles=3,
        on_cycle_complete=on_cycle,
        on_tick=on_tick,
        silent=True,
    )

    print(f"\n--- Oscillator Config ---")
    print(f"  Frequency: {osc.frequency_hz} Hz")
    print(f"  Period: {osc.period_seconds:.2f}s")
    print(f"  Waveform: {osc.waveform.value}")
    print(f"  Max Cycles: {osc.max_cycles}")

    print(f"\n--- Running (3 cycles) ---")
    osc.start()
    time.sleep(2.0)  # Let it run for ~3 cycles
    osc.stop()

    stats = osc.get_statistics()
    print(f"\n--- Statistics ---")
    print(f"  Cycles Completed: {stats['cycle_count']}")
    print(f"  Total Runtime: {stats['total_runtime_seconds']:.2f}s")
    print(f"  Tick Count: {tick_count[0]}")

    return osc


def demonstrate_phased_oscillator():
    """Demonstrate oscillator with named phases."""
    print("\n" + "="*60)
    print("5. PHASED OSCILLATOR (WORK/REST CYCLE)")
    print("="*60)

    work_count = [0]
    rest_count = [0]

    def do_work():
        work_count[0] += 1
        return "working"

    def do_rest():
        rest_count[0] += 1
        return "resting"

    osc = Oscillator(
        frequency_hz=1.0,  # 1 second total period
        max_cycles=2,
        silent=True,
    )

    # Add phases (must sum to period)
    osc.add_phase(OscillatorPhase(
        name="work",
        duration_seconds=0.3,
        action=do_work,
        on_enter=lambda: print("  -> Entering WORK phase"),
        on_exit=lambda: print("  <- Exiting WORK phase"),
    ))

    osc.add_phase(OscillatorPhase(
        name="rest",
        duration_seconds=0.7,
        action=do_rest,
        on_enter=lambda: print("  -> Entering REST phase"),
        on_exit=lambda: print("  <- Exiting REST phase"),
    ))

    print("\n--- Running 2 Work/Rest Cycles ---")
    osc.start()
    time.sleep(2.5)  # Allow 2 cycles
    osc.stop()

    print(f"\n--- Summary ---")
    print(f"  Work actions: {work_count[0]}")
    print(f"  Rest actions: {rest_count[0]}")

    return osc


def demonstrate_heartbeat():
    """Demonstrate heartbeat oscillator."""
    print("\n" + "="*60)
    print("6. HEARTBEAT OSCILLATOR")
    print("="*60)

    beats = [0]

    def heartbeat():
        beats[0] += 1
        return f"beat-{beats[0]}"

    hb = HeartbeatOscillator(
        beats_per_minute=120,  # 2 beats per second
        on_beat=heartbeat,
        max_cycles=5,
        silent=True,
    )

    print(f"\n--- Config ---")
    print(f"  BPM: 120 (2 Hz)")
    print(f"  Max Beats: 5")

    print(f"\n--- Running ---")
    hb.start()
    time.sleep(3.0)  # ~5 beats
    hb.stop()

    print(f"\n--- Results ---")
    print(f"  Total Beats: {beats[0]}")

    return hb


def demonstrate_quorum_strategies():
    """Demonstrate different quorum voting strategies."""
    print("\n" + "="*60)
    print("7. QUORUM SENSING: VOTING STRATEGIES")
    print("="*60)

    budget = ATP_Store(budget=1000, silent=True)

    # Test different strategies
    strategies = [
        (VotingStrategy.MAJORITY, "Simple Majority (>50%)"),
        (VotingStrategy.SUPERMAJORITY, "Supermajority (>66%)"),
        (VotingStrategy.WEIGHTED, "Weighted Voting"),
        (VotingStrategy.CONFIDENCE, "Confidence-Weighted"),
    ]

    for strategy, description in strategies:
        print(f"\n--- {description} ---")

        quorum = QuorumSensing(
            n_agents=5,
            budget=budget,
            strategy=strategy,
            silent=True,
        )

        # Set different weights for weighted voting
        if strategy == VotingStrategy.WEIGHTED:
            quorum.set_agent_weight("Bacterium_0", 2.0)  # Expert
            quorum.set_agent_weight("Bacterium_1", 1.5)  # Senior
            # Others remain at 1.0

        result = quorum.run_vote("Should we deploy to production?")

        print(f"  Permits: {result.permit_votes}, Blocks: {result.block_votes}")
        print(f"  Weighted Score: {result.weighted_score:.1%}")
        print(f"  Decision: {'APPROVED' if result.reached else 'REJECTED'}")

    return quorum


def demonstrate_quorum_reliability():
    """Demonstrate agent reliability tracking."""
    print("\n" + "="*60)
    print("8. QUORUM: RELIABILITY TRACKING")
    print("="*60)

    budget = ATP_Store(budget=1000, silent=True)

    quorum = QuorumSensing(
        n_agents=5,
        budget=budget,
        strategy=VotingStrategy.WEIGHTED,
        enable_reliability_tracking=True,
        silent=True,
    )

    print("\n--- Initial Rankings ---")
    for rank in quorum.get_agent_rankings()[:3]:
        print(f"  {rank['name']}: reliability={rank['reliability']:.2f}, "
              f"weight={rank['effective_weight']:.2f}")

    # Run some votes and update reliability
    print("\n--- Running Votes and Updating Reliability ---")
    for i in range(3):
        result = quorum.run_vote(f"Decision {i+1}")
        # Simulate: assume the final decision was correct
        quorum.update_all_reliability(result.decision)

    print("\n--- Updated Rankings ---")
    for rank in quorum.get_agent_rankings()[:3]:
        print(f"  {rank['name']}: reliability={rank['reliability']:.2f}, "
              f"votes={rank['votes_cast']}, weight={rank['effective_weight']:.2f}")

    # Statistics
    stats = quorum.get_statistics()
    print(f"\n--- Statistics ---")
    print(f"  Total Votes: {stats['total_votes']}")
    print(f"  Quorums Reached: {stats['quorums_reached']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")

    return quorum


def main():
    """Run all topology demonstrations."""
    print("="*60)
    print("NETWORK TOPOLOGIES DEMONSTRATION")
    print("Cascade, Oscillator, and Quorum Sensing")
    print("="*60)

    demonstrate_cascade_basics()
    demonstrate_cascade_with_checkpoints()
    demonstrate_mapk_cascade()
    demonstrate_oscillator_basics()
    demonstrate_phased_oscillator()
    demonstrate_heartbeat()
    demonstrate_quorum_strategies()
    demonstrate_quorum_reliability()

    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
