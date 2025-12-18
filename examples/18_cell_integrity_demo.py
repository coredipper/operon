"""
Cell Integrity Systems Demo

This example demonstrates the three integrated systems for robust agentic operations:
1. Quality System (Ubiquitin-Proteasome) - Provenance tracking and cascade prevention
2. Surveillance System (Immune) - Byzantine agent detection
3. Coordination System (Cell Cycle) - Deadlock prevention

Each system is biologically inspired, providing intuitive mental models for
complex reliability patterns.
"""

from datetime import timedelta
from operon_ai.cell import IntegratedCell, CellHealth


def main():
    print("=" * 60)
    print("Cell Integrity Systems Demo")
    print("=" * 60)

    # Create an integrated cell with all three systems
    cell = IntegratedCell(
        pool_capacity=100,           # Ubiquitin pool capacity
        degradation_threshold=0.3,    # Quality threshold
        max_operation_time=timedelta(seconds=30),  # Watchdog timeout
    )

    # Register agents and resources
    cell.register_agent("summarizer")
    cell.register_agent("translator")
    cell.register_resource("api_rate_limit")
    cell.register_resource("shared_context")

    print("\n1. QUALITY SYSTEM (Ubiquitin-Proteasome)")
    print("-" * 40)

    # Execute operations with provenance tracking
    result1 = cell.execute(
        agent_id="summarizer",
        operation_id="summarize_doc_1",
        work_fn=lambda: "This is a summary of the document.",
    )

    print(f"   Execution success: {result1.success}")
    if result1.tagged_output:
        tag = result1.tagged_output.tag
        print(f"   Provenance tag:")
        print(f"     - Origin: {tag.origin}")
        print(f"     - Confidence: {tag.confidence}")
        print(f"     - Generation: {tag.generation}")

    print("\n2. SURVEILLANCE SYSTEM (Immune)")
    print("-" * 40)

    # Record multiple observations to build behavioral profile
    for i in range(5):
        cell.execute(
            agent_id="translator",
            operation_id=f"translate_{i}",
            work_fn=lambda: f"Translated text {i}",
        )

    # Check surveillance display
    display = cell.surveillance.displays.get("translator")
    if display:
        print(f"   Observations recorded: {len(display.observations)}")
        peptide = display.generate_peptide()
        if peptide:
            print(f"   Behavioral fingerprint (MHCPeptide):")
            print(f"     - Output length mean: {peptide.output_length_mean:.1f}")
            print(f"     - Response time mean: {peptide.response_time_mean:.4f}s")
            print(f"     - Confidence mean: {peptide.confidence_mean:.2f}")

    print("\n3. COORDINATION SYSTEM (Cell Cycle)")
    print("-" * 40)

    # Execute with resource coordination
    result2 = cell.execute(
        agent_id="summarizer",
        operation_id="coordinated_op",
        work_fn=lambda: "Coordinated output",
        resources=["api_rate_limit"],
    )

    print(f"   Coordinated execution success: {result2.success}")
    if result2.coordination_result:
        print(f"   Phase reached: {result2.coordination_result.phase_reached}")
        print(f"   Duration: {result2.coordination_result.duration_ms:.2f}ms")

    # Check that resource was released
    lock = cell.coordination.controller.resources.get("api_rate_limit")
    if lock:
        print(f"   Resource released: {lock.is_available}")

    print("\n4. INTEGRATED HEALTH CHECK")
    print("-" * 40)

    health = cell.health()
    print(f"   Cell health: {health}")
    print(f"   Pool status: {health.pool_status}")
    print(f"   Active operations: {health.coordination_stats.get('active_operations', 0)}")

    print("\n5. MAINTENANCE CYCLE")
    print("-" * 40)

    events = cell.run_maintenance()
    print(f"   Maintenance events: {events}")

    print("\n6. GRACEFUL SHUTDOWN")
    print("-" * 40)

    cell.shutdown()
    print("   Cell shutdown complete")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def demonstrate_quality_decay():
    """Demonstrate confidence decay through agent chain."""
    from operon_ai.quality import UbiquitinTag, TaggedData

    print("\n--- Quality System: Confidence Decay ---")

    # Create initial high-confidence data
    tag = UbiquitinTag(confidence=1.0, origin="source_agent", generation=0)
    data = TaggedData(data="Original data", tag=tag)
    print(f"Gen 0: confidence={data.tag.confidence:.2f}")

    # Simulate passing through multiple agents with decay
    for i in range(5):
        new_tag = data.tag.reduce_confidence(0.9).increment_generation()
        data = TaggedData(data=f"Processed by agent_{i+1}", tag=new_tag)
        print(f"Gen {i+1}: confidence={data.tag.confidence:.2f}")

    print("→ Confidence decays through chain, preventing cascade failures")


def demonstrate_two_signal():
    """Demonstrate two-signal immune activation."""
    from operon_ai.surveillance.types import Signal1, Signal2
    from operon_ai.surveillance.thymus import BaselineProfile
    from operon_ai.surveillance.tcell import TCell

    print("\n--- Surveillance System: Two-Signal Activation ---")

    # Create a baseline profile
    profile = BaselineProfile(
        agent_id="test_agent",
        output_length_bounds=(50, 150),
        response_time_bounds=(0.1, 1.0),
        confidence_bounds=(0.8, 1.0),
        error_rate_max=0.1,
        valid_vocabulary_hashes={"hash1", "hash2"},
        valid_structure_hashes={"struct1"},
        canary_accuracy_min=0.9,
    )

    tcell = TCell(profile=profile)

    print("T-cell requires TWO signals to activate:")
    print("  Signal 1: MHC recognition (anomaly detected)")
    print("  Signal 2: Co-stimulation (canary failure, repeated anomaly, etc.)")
    print("→ This prevents false positives (attacking healthy agents)")


def demonstrate_deadlock_detection():
    """Demonstrate deadlock detection and resolution."""
    from operon_ai.coordination import CoordinationSystem
    from operon_ai.coordination.types import ResourceLock

    print("\n--- Coordination System: Deadlock Detection ---")

    system = CoordinationSystem()
    system.register_resource("resource_a")
    system.register_resource("resource_b")

    # Create potential deadlock scenario
    ctx1 = system.start_operation("op1", "agent1", priority=1)
    ctx2 = system.start_operation("op2", "agent2", priority=2)

    system.controller.advance(ctx1)
    system.controller.advance(ctx2)

    # op1 holds A, wants B
    system.controller.acquire_resource(ctx1, "resource_a")
    # op2 holds B, wants A
    system.controller.acquire_resource(ctx2, "resource_b")

    # Check for potential deadlock
    system.controller.acquire_resource(ctx1, "resource_b")  # Blocked
    system.controller.acquire_resource(ctx2, "resource_a")  # Blocked - deadlock!

    deadlock = system.controller.check_deadlock()
    if deadlock:
        print(f"  Deadlock detected!")
        print(f"  Agents involved: {deadlock.agents}")
        print(f"  Resources: {deadlock.resources}")

    print("→ Cell cycle checkpoints prevent and detect deadlocks")

    # Cleanup
    system.shutdown()


if __name__ == "__main__":
    main()

    print("\n" + "=" * 60)
    print("Additional Demonstrations")
    print("=" * 60)

    demonstrate_quality_decay()
    demonstrate_two_signal()
    demonstrate_deadlock_detection()
