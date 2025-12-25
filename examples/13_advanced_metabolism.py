"""
Example 13: Advanced Metabolism - Multi-Currency Energy Management
==================================================================

This example demonstrates the enhanced ATP_Store with:
- Multiple energy currencies (ATP, GTP, NADH)
- Metabolic states (Normal, Conserving, Starving, Feasting)
- Energy debt and regeneration
- Energy transfer between agents

Biological Analogy:
Like cellular metabolism where ATP is the primary energy currency,
GTP powers specialized processes, and NADH serves as an energy reserve
that can be converted to ATP through oxidative phosphorylation.
"""

import time
from operon_ai.state import (
    ATP_Store,
    EnergyType,
    MetabolicState,
)


def demonstrate_multi_currency():
    """Demonstrate multiple energy currencies."""
    print("\n" + "="*60)
    print("1. MULTI-CURRENCY ENERGY SYSTEM")
    print("="*60)

    # Create a metabolism with all three currencies
    metabolism = ATP_Store(
        budget=100,          # ATP for general operations
        gtp_budget=50,       # GTP for specialized operations
        nadh_reserve=30,     # NADH reserve for emergencies
        regeneration_rate=0,  # Disable auto-regen for demo
    )

    print(f"\nInitial state:")
    print(f"  ATP: {metabolism.atp}/{metabolism.max_atp}")
    print(f"  GTP: {metabolism.gtp}/{metabolism.max_gtp}")
    print(f"  NADH: {metabolism.nadh}/{metabolism.max_nadh}")
    print(f"  State: {metabolism.get_state().value}")

    # Consume ATP for regular operations
    print("\n--- Regular Operations (ATP) ---")
    for i in range(3):
        success = metabolism.consume(20, f"operation_{i}", EnergyType.ATP)
        print(f"  Operation {i}: {'✓' if success else '✗'} (ATP: {metabolism.atp})")

    # Use GTP for specialized operation
    print("\n--- Specialized Operation (GTP) ---")
    success = metabolism.consume(25, "tool_call", EnergyType.GTP)
    print(f"  Tool call: {'✓' if success else '✗'} (GTP: {metabolism.gtp})")

    # Convert NADH to ATP
    print("\n--- Oxidative Phosphorylation ---")
    converted = metabolism.convert_nadh_to_atp(20)
    print(f"  Converted {converted} NADH → ATP")
    print(f"  ATP: {metabolism.atp}, NADH: {metabolism.nadh}")

    return metabolism


def demonstrate_metabolic_states():
    """Demonstrate metabolic state transitions."""
    print("\n" + "="*60)
    print("2. METABOLIC STATE TRANSITIONS")
    print("="*60)

    def on_state_change(new_state: MetabolicState):
        states = {
            MetabolicState.NORMAL: "🟢 Normal",
            MetabolicState.CONSERVING: "🟡 Conserving",
            MetabolicState.STARVING: "🔴 Starving",
            MetabolicState.FEASTING: "🟣 Feasting",
        }
        print(f"  State change: {states.get(new_state, new_state.value)}")

    metabolism = ATP_Store(
        budget=100,
        on_state_change=on_state_change,
        silent=True,  # We'll handle our own output
    )

    print("\nDraining energy to trigger state changes:")
    for i in range(12):
        metabolism.consume(10, f"drain_{i}")
        print(f"  After drain {i+1}: ATP={metabolism.atp}, State={metabolism.get_state().value}")
        if metabolism.get_state() == MetabolicState.STARVING:
            break

    print("\nRegenerating energy:")
    for i in range(5):
        metabolism.regenerate(20)
        print(f"  After regen {i+1}: ATP={metabolism.atp}, State={metabolism.get_state().value}")

    return metabolism


def demonstrate_energy_debt():
    """Demonstrate energy debt system."""
    print("\n" + "="*60)
    print("3. ENERGY DEBT SYSTEM")
    print("="*60)

    metabolism = ATP_Store(
        budget=50,
        max_debt=30,         # Allow up to 30 units of debt
        debt_interest=0.1,   # 10% interest per operation
    )

    print(f"\nInitial: ATP={metabolism.atp}, Max Debt={metabolism.max_debt}")

    # Consume more than available (using debt)
    print("\n--- Operations with Debt ---")
    for i in range(4):
        success = metabolism.consume(20, f"expensive_op_{i}", allow_debt=True)
        print(f"  Op {i}: {'✓' if success else '✗'} ATP={metabolism.atp}, Debt={metabolism.get_debt()}")

    # Apply interest
    print("\n--- Applying Interest ---")
    metabolism.apply_debt_interest()
    print(f"  After interest: Debt={metabolism.get_debt()}")

    # Regenerate to pay off debt
    print("\n--- Regenerating (pays debt first) ---")
    for i in range(3):
        metabolism.regenerate(20)
        print(f"  After regen: ATP={metabolism.atp}, Debt={metabolism.get_debt()}")

    return metabolism


def demonstrate_energy_transfer():
    """Demonstrate energy sharing between agents."""
    print("\n" + "="*60)
    print("4. ENERGY TRANSFER BETWEEN AGENTS")
    print("="*60)

    # Create two agent metabolisms
    donor = ATP_Store(budget=100, silent=True)
    recipient = ATP_Store(budget=20, silent=True)

    print(f"\nBefore transfer:")
    print(f"  Donor ATP: {donor.atp}")
    print(f"  Recipient ATP: {recipient.atp}")

    # Transfer energy
    print("\n--- Transferring 40 ATP ---")
    success = donor.transfer_to(recipient, 40, EnergyType.ATP)
    print(f"  Transfer: {'✓ Success' if success else '✗ Failed'}")

    print(f"\nAfter transfer:")
    print(f"  Donor ATP: {donor.atp}")
    print(f"  Recipient ATP: {recipient.atp}")

    return donor, recipient


def demonstrate_metabolic_report():
    """Demonstrate comprehensive metabolic reporting."""
    print("\n" + "="*60)
    print("5. METABOLIC HEALTH REPORT")
    print("="*60)

    metabolism = ATP_Store(
        budget=100,
        gtp_budget=50,
        nadh_reserve=30,
        max_debt=20,
        silent=True,
    )

    # Simulate some activity
    for i in range(5):
        metabolism.consume(10, f"task_{i}")
    metabolism.consume(20, "tool", EnergyType.GTP)
    metabolism.consume(100, "big_op", allow_debt=True)  # This will fail or use debt

    # Get comprehensive report
    report = metabolism.get_report()

    print(f"\n📊 Metabolic Report:")
    print(f"  State: {report.state.value}")
    print(f"  ATP: {report.atp} / GTP: {report.gtp} / NADH: {report.nadh}")
    print(f"  Total Capacity: {report.total_capacity}")
    print(f"  Utilization: {report.utilization:.1%}")
    print(f"  Regeneration Rate: {report.regeneration_rate}/s")
    print(f"  Debt: {report.debt}")
    print(f"  Health Score: {report.health_score:.1%}")
    print(f"  Transactions: {report.transactions_count}")

    # Get statistics
    stats = metabolism.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"  Total Consumed: {stats['total_consumed']}")
    print(f"  Total Regenerated: {stats['total_regenerated']}")
    print(f"  Operations: {stats['operations_count']}")
    print(f"  Failed Operations: {stats['failed_operations']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")

    return metabolism


def main():
    """Run all metabolism demonstrations."""
    print("="*60)
    print("ADVANCED METABOLISM DEMONSTRATION")
    print("Multi-Currency Energy Management System")
    print("="*60)

    demonstrate_multi_currency()
    demonstrate_metabolic_states()
    demonstrate_energy_debt()
    demonstrate_energy_transfer()
    demonstrate_metabolic_report()

    print("\n" + "="*60)
    print("✅ All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
