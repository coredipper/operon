"""
Example 111: DNA Repair — State Integrity Checking
===================================================

Demonstrates the DNA repair motif for proactive state integrity:

1. Create a Genome with 4 genes + HistoneStore for repair memory
2. Take a StateCheckpoint of the healthy genome
3. Inject corruption: mutate a gene + drift expression
4. Scan to detect damage reports
5. Repair each damage site
6. Re-scan to confirm clean state
7. Issue a state_integrity_verified certificate
8. Inspect repair lessons stored in epigenetic memory

DNA repair fills the gap between surveillance (external threats) and
healing (output/agent recovery).  It detects INTERNAL state corruption
before it propagates downstream.

Usage: python examples/111_dna_repair_integrity.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import (
    Gene,
    GeneType,
    Genome,
    ExpressionLevel,
    HistoneStore,
    DNARepair,
    CorruptionType,
)


def main():
    print("=" * 60)
    print("DNA Repair — State Integrity Checking")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Setup: genome, histones, repair engine
    # -----------------------------------------------------------------------
    genome = Genome(
        genes=[
            Gene("model", "gpt-4", gene_type=GeneType.STRUCTURAL, required=True),
            Gene("temperature", 0.7),
            Gene("max_tokens", 4096),
            Gene("retry_count", 3),
        ],
        allow_mutations=True,
        silent=True,
    )

    histones = HistoneStore(silent=True)
    repair = DNARepair(histone_store=histones, silent=False)

    # -----------------------------------------------------------------------
    # 2. Checkpoint healthy state
    # -----------------------------------------------------------------------
    print("\n--- Checkpoint ---")
    checkpoint = repair.checkpoint(genome)
    print(f"  hash:  {checkpoint.genome_hash}")
    print(f"  genes: {checkpoint.gene_count}")

    # -----------------------------------------------------------------------
    # 3. Inject corruption
    # -----------------------------------------------------------------------
    print("\n--- Injecting Corruption ---")

    # Genome drift: mutate temperature
    genome.mutate("temperature", 0.9, reason="unauthorized drift")
    print("  mutated temperature: 0.7 → 0.9")

    # Expression drift: silence max_tokens without regulatory signal
    genome._expression["max_tokens"].level = ExpressionLevel.SILENCED
    genome._expression["max_tokens"].modifier = ""
    print("  silenced max_tokens (no regulatory signal)")

    # -----------------------------------------------------------------------
    # 4. Scan for damage
    # -----------------------------------------------------------------------
    print("\n--- Scan ---")
    damage = repair.scan(genome, checkpoint)
    print(f"\n  Total damage sites: {len(damage)}")
    for d in damage:
        print(f"  [{d.severity.name}] {d.corruption_type.value} @ {d.location}")
        print(f"    expected: {d.expected}")
        print(f"    actual:   {d.actual}")
        print(f"    fix:      {d.recommended_strategy.value}")

    # -----------------------------------------------------------------------
    # 5. Repair each damage
    # -----------------------------------------------------------------------
    print("\n--- Repair ---")
    # Apply the highest-severity repair (CHECKPOINT_RESTORE) which fixes
    # all state at once, then re-scan to confirm clean.
    repair_results = []
    remaining = damage
    for _ in range(len(damage) + 2):
        if not remaining:
            break
        result = repair.repair(genome, remaining[0], checkpoint=checkpoint)
        repair_results.append(result)
        if not result.success:
            break
        prev = len(remaining)
        remaining = repair.scan(genome, checkpoint)
        if len(remaining) >= prev:
            break

    # -----------------------------------------------------------------------
    # 6. Re-scan (should be clean)
    # -----------------------------------------------------------------------
    print("\n--- Re-scan ---")
    print(f"  remaining damage: {len(remaining)}")

    # -----------------------------------------------------------------------
    # 7. Certify state integrity
    # -----------------------------------------------------------------------
    print("\n--- Certify ---")
    cert = repair.certify(genome, checkpoint)
    verification = cert.verify()
    print(f"  theorem:    {cert.theorem}")
    print(f"  holds:      {verification.holds}")
    print(f"  evidence:   {dict(verification.evidence)}")

    # -----------------------------------------------------------------------
    # 8. Inspect repair memory
    # -----------------------------------------------------------------------
    print("\n--- Repair Memory (Histone Markers) ---")
    context = histones.retrieve_context(tags=["repair"])
    print(f"  repair markers: {context.active_markers}")
    for marker in context.markers:
        print(f"  [{', '.join(marker.tags)}] {marker.content[:60]}...")

    # -----------------------------------------------------------------------
    # 9. Statistics
    # -----------------------------------------------------------------------
    print("\n--- Statistics ---")
    stats = repair.get_statistics()
    print(f"  checkpoints:       {stats['checkpoints']}")
    print(f"  scans:             {stats['scans_performed']}")
    print(f"  repairs attempted: {stats['repairs_attempted']}")
    print(f"  repairs succeeded: {stats['repairs_successful']}")

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------
    print("\n--- Assertions ---")

    assert len(damage) >= 2, "should detect at least 2 damage sites"
    assert any(d.corruption_type == CorruptionType.CHECKSUM_FAILURE for d in damage)
    assert any(d.corruption_type == CorruptionType.EXPRESSION_DRIFT for d in damage)

    # Re-scan should be clean after repair
    assert len(remaining) == 0, f"post-repair scan should be clean, got {len(remaining)} damage"

    # Certificate should hold after repair
    assert verification.holds, "repaired genome should verify clean"

    # Histone markers should exist
    assert context.active_markers >= 1, "repair markers should be stored"

    # Statistics should reflect operations
    assert stats["checkpoints"] == 1
    assert stats["scans_performed"] >= 2
    assert stats["repairs_attempted"] >= 1
    assert stats["repairs_successful"] >= 1

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
