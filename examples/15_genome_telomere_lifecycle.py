"""
Example 15: Agent Lifecycle with Genome and Telomere
====================================================

This example demonstrates:
- Genome: Immutable configuration with gene expression control
- Telomere: Lifecycle management and senescence
- Integration: How genome and telomere work together

Biological Analogy:
- Genome: DNA that defines the fundamental traits of an organism
- Genes: Individual traits that can be expressed or silenced
- Telomere: Chromosome caps that shorten with each division
- Senescence: Cellular aging leading to reduced function
"""

from operon_ai.state import (
    ExpressionLevel,
    Gene,
    GeneType,
    Genome,
    LifecyclePhase,
    SenescenceReason,
    Telomere,
    TelomereStatus,
)


def demonstrate_genome_basics():
    """Demonstrate basic genome operations."""
    print("\n" + "="*60)
    print("1. GENOME: IMMUTABLE CONFIGURATION")
    print("="*60)

    # Create a genome with initial genes
    genome = Genome(
        genes=[
            Gene(name="model", value="gpt-4", gene_type=GeneType.STRUCTURAL, required=True),
            Gene(name="temperature", value=0.7, gene_type=GeneType.REGULATORY),
            Gene(name="max_tokens", value=4096, gene_type=GeneType.HOUSEKEEPING),
        ],
        allow_mutations=False,  # Immutable by default
        silent=True,
    )

    print("\n--- Initial Genome ---")
    for gene_info in genome.list_genes():
        print(f"  {gene_info['name']}: {gene_info['value']} "
              f"({gene_info['type']}, {gene_info['expression']})")

    # Try to add a duplicate (will fail - immutable)
    print("\n--- Attempting Duplicate Gene (should fail) ---")
    success = genome.add_gene(Gene(name="model", value="gpt-3.5"))
    print(f"  Add duplicate: {'Success' if success else 'Blocked (expected)'}")

    # Get expressed configuration
    print("\n--- Expressed Configuration ---")
    config = genome.express()
    for key, value in config.items():
        print(f"  {key}: {value}")

    return genome


def demonstrate_gene_expression():
    """Demonstrate gene expression control."""
    print("\n" + "="*60)
    print("2. GENE EXPRESSION CONTROL")
    print("="*60)

    genome = Genome(
        genes=[
            Gene(name="verbose_mode", value=True, gene_type=GeneType.REGULATORY),
            Gene(name="debug_mode", value=True, gene_type=GeneType.CONDITIONAL),
            Gene(name="safety_checks", value=True, gene_type=GeneType.HOUSEKEEPING, required=True),
            Gene(name="experimental_feature", value=True, gene_type=GeneType.DORMANT),
        ],
        silent=True,
    )

    print("\n--- Initial Expression ---")
    config = genome.express()
    print(f"  Active genes: {list(config.keys())}")

    # Silence verbose mode
    print("\n--- Silencing verbose_mode ---")
    genome.silence_gene("verbose_mode", reason="user preference")
    config = genome.express()
    print(f"  Active genes: {list(config.keys())}")

    # Express conditional gene with context
    print("\n--- Express with Context (debug_mode) ---")
    config_with_context = genome.express(context={"debug_mode": True})
    print(f"  Active genes: {list(config_with_context.keys())}")

    # Note: dormant genes are never expressed
    print("\n--- Dormant Gene Status ---")
    dormant = genome.get_gene("experimental_feature")
    print(f"  experimental_feature exists: {dormant is not None}")
    print(f"  experimental_feature in config: {'experimental_feature' in config}")

    return genome


def demonstrate_genome_inheritance():
    """Demonstrate genome replication and inheritance."""
    print("\n" + "="*60)
    print("3. GENOME REPLICATION (INHERITANCE)")
    print("="*60)

    # Parent genome
    parent = Genome(
        genes=[
            Gene(name="base_model", value="gpt-4"),
            Gene(name="temperature", value=0.7),
            Gene(name="creativity", value=0.8),
        ],
        allow_mutations=True,  # Allow mutations in children
        mutation_rate=0.0,     # No random mutations
        silent=True,
    )

    print("\n--- Parent Genome ---")
    print(f"  Hash: {parent.get_hash()}")
    print(f"  Generation: {parent._generation}")
    for gene_info in parent.list_genes():
        print(f"  {gene_info['name']}: {gene_info['value']}")

    # Create child with specific mutations
    print("\n--- Creating Child with Mutations ---")
    child = parent.replicate(
        mutations={"temperature": 0.9, "creativity": 0.5},
        inherit_expression=True
    )

    print(f"  Child Hash: {child.get_hash()}")
    print(f"  Child Generation: {child._generation}")
    for gene_info in child.list_genes():
        print(f"  {gene_info['name']}: {gene_info['value']}")

    # Show differences
    print("\n--- Genome Diff (Parent vs Child) ---")
    diff = parent.diff(child)
    for gene_name, (parent_val, child_val) in diff.items():
        print(f"  {gene_name}: {parent_val} -> {child_val}")

    return parent, child


def demonstrate_telomere_lifecycle():
    """Demonstrate telomere-based lifecycle management."""
    print("\n" + "="*60)
    print("4. TELOMERE: LIFECYCLE MANAGEMENT")
    print("="*60)

    def on_phase_change(old_phase, new_phase):
        print(f"  Phase: {old_phase.value} -> {new_phase.value}")

    def on_senescence(reason):
        print(f"  Senescence triggered: {reason.value}")

    telomere = Telomere(
        max_operations=100,
        error_threshold=10,
        allow_renewal=True,
        on_phase_change=on_phase_change,
        on_senescence=on_senescence,
        silent=True,
    )

    print("\n--- Initial State ---")
    status = telomere.get_status()
    print(f"  Phase: {status.phase.value}")
    print(f"  Telomere: {status.telomere_length}/{status.max_telomere_length}")
    print(f"  Health: {status.health_score:.1%}")

    # Perform operations (shortens telomeres)
    print("\n--- Performing Operations ---")
    for i in range(80):
        can_continue = telomere.tick()
        if i % 20 == 19:  # Report every 20 ops
            status = telomere.get_status()
            print(f"  After {i+1} ops: "
                  f"Telomere={status.telomere_length}, "
                  f"Phase={status.phase.value}")
        if not can_continue:
            print(f"  Stopped at operation {i+1}")
            break

    return telomere


def demonstrate_telomere_renewal():
    """Demonstrate telomere renewal (telomerase)."""
    print("\n" + "="*60)
    print("5. TELOMERE RENEWAL (TELOMERASE)")
    print("="*60)

    telomere = Telomere(
        max_operations=50,
        allow_renewal=True,
        silent=True,
    )

    # Deplete telomeres
    print("\n--- Depleting Telomeres ---")
    for i in range(45):
        telomere.tick()

    status = telomere.get_status()
    print(f"  After 45 ops: Telomere={status.telomere_length}, Phase={status.phase.value}")

    # Renew telomeres
    print("\n--- Applying Telomerase ---")
    success = telomere.renew(amount=30, reset_errors=True)
    print(f"  Renewal: {'Success' if success else 'Failed'}")

    status = telomere.get_status()
    print(f"  After renewal: Telomere={status.telomere_length}, Phase={status.phase.value}")

    # Continue operations
    print("\n--- Continuing Operations ---")
    for i in range(20):
        can_continue = telomere.tick()
        if not can_continue:
            print(f"  Stopped after {i+1} more ops")
            break

    status = telomere.get_status()
    print(f"  Final: Telomere={status.telomere_length}, Phase={status.phase.value}")

    return telomere


def demonstrate_error_senescence():
    """Demonstrate error-based senescence."""
    print("\n" + "="*60)
    print("6. ERROR-BASED SENESCENCE")
    print("="*60)

    def on_senescence(reason):
        print(f"  SENESCENCE: {reason.value}")

    telomere = Telomere(
        max_operations=1000,
        error_threshold=5,  # Low threshold for demo
        on_senescence=on_senescence,
        silent=True,
    )

    print("\n--- Simulating Errors ---")
    for i in range(6):
        can_continue = telomere.record_error()
        status = telomere.get_status()
        print(f"  Error {i+1}: Can continue={can_continue}, Phase={status.phase.value}")
        if not can_continue:
            break

    return telomere


def demonstrate_integrated_lifecycle():
    """Demonstrate genome and telomere working together."""
    print("\n" + "="*60)
    print("7. INTEGRATED LIFECYCLE")
    print("="*60)

    # Create an agent with genome and telomere
    genome = Genome(
        genes=[
            Gene(name="agent_type", value="worker", gene_type=GeneType.STRUCTURAL),
            Gene(name="max_ops", value=100),
            Gene(name="allow_renewal", value=True),
        ],
        silent=True,
    )

    # Configure telomere from genome
    config = genome.express()
    telomere = Telomere(
        max_operations=config["max_ops"],
        allow_renewal=config["allow_renewal"],
        silent=True,
    )

    print("\n--- Agent Configuration ---")
    print(f"  Type: {config['agent_type']}")
    print(f"  Max Operations: {config['max_ops']}")
    print(f"  Renewal Allowed: {config['allow_renewal']}")

    # Simulate agent lifecycle
    print("\n--- Agent Lifecycle ---")
    operations = 0
    while telomere.is_operational():
        can_continue = telomere.tick(cost=5)  # Each op costs 5 telomere units
        operations += 1

        # Check for senescence
        if telomere.get_phase() == LifecyclePhase.SENESCENT:
            print(f"  Senescence at operation {operations}")
            # Decision: renew or terminate
            if config["allow_renewal"]:
                print("  Attempting renewal...")
                telomere.renew()
                break

        if operations >= 50:  # Safety limit for demo
            break

    status = telomere.get_status()
    stats = telomere.get_statistics()

    print(f"\n--- Final Status ---")
    print(f"  Operations: {stats['operations_count']}")
    print(f"  Phase: {status.phase.value}")
    print(f"  Health: {status.health_score:.1%}")
    print(f"  Genome Hash: {genome.get_hash()}")

    return genome, telomere


def main():
    """Run all genome and telomere demonstrations."""
    print("="*60)
    print("GENOME AND TELOMERE LIFECYCLE DEMONSTRATION")
    print("Immutable Configuration + Lifecycle Management")
    print("="*60)

    demonstrate_genome_basics()
    demonstrate_gene_expression()
    demonstrate_genome_inheritance()
    demonstrate_telomere_lifecycle()
    demonstrate_telomere_renewal()
    demonstrate_error_senescence()
    demonstrate_integrated_lifecycle()

    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
