"""
Example 56: Metabolic-Epigenetic Coupling — Cost-Gated Retrieval
================================================================

Demonstrates how metabolic state gates access to epigenetic memory,
implementing the paper's Equation 15 (Section 6.1.1).

Core idea: Memory retrieval isn't free. When the cell is starving,
only deeply embedded (permanent) memories remain accessible, while
transient memories are effectively "silenced." This creates emergent
behavior where resource pressure narrows cognitive focus.

    Access(d) = Open      if R > threshold for marker strength
              = Silenced  if R <= threshold

Biological Analogy:
- Chromatin remodeling requires ATP in real cells
- Under metabolic stress, cells silence non-essential genes
- Only constitutively active (housekeeping) genes remain expressed
- This is how biology achieves "graceful degradation" under stress

References:
- Article Section 6.1.1: RAG as Digital Methylation
- Article Section 6.6: Bioenergetic Intelligence
"""

from operon_ai.state.metabolism import ATP_Store, MetabolicState, MetabolicAccessPolicy
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength


def main():
    try:
        print("=" * 60)
        print("Metabolic-Epigenetic Coupling — Cost-Gated Retrieval")
        print("=" * 60)

        # =================================================================
        # SECTION 1: Set Up Coupled Systems
        # =================================================================
        print("\n--- Section 1: Coupling ATP_Store with HistoneStore ---")

        atp = ATP_Store(budget=100, silent=True)
        policy = MetabolicAccessPolicy(retrieval_cost=5)
        histones = HistoneStore(energy_gate=(atp, policy), silent=True)

        # Store memories at different strengths
        histones.add_marker(
            "User prefers concise responses",
            marker_type=MarkerType.PHOSPHORYLATION,
            strength=MarkerStrength.WEAK,
            tags=["preference"],
        )
        histones.add_marker(
            "Always validate SQL before execution",
            marker_type=MarkerType.ACETYLATION,
            strength=MarkerStrength.MODERATE,
            tags=["safety"],
        )
        histones.add_marker(
            "Production DB credentials are read-only",
            marker_type=MarkerType.METHYLATION,
            strength=MarkerStrength.STRONG,
            tags=["safety", "critical"],
        )
        histones.add_marker(
            "NEVER run DELETE without WHERE clause",
            marker_type=MarkerType.METHYLATION,
            strength=MarkerStrength.PERMANENT,
            tags=["safety", "critical"],
        )

        print(f"  Stored 4 markers (WEAK, MODERATE, STRONG, PERMANENT)")
        print(f"  ATP: {atp.atp}/{atp.max_atp}")
        print(f"  Metabolic state: {atp.get_state().value}")

        # =================================================================
        # SECTION 2: Normal State — Full Access
        # =================================================================
        print("\n--- Section 2: NORMAL state — all memories accessible ---")

        result = histones.retrieve_context(tags=["safety"])
        print(f"  Retrieved {len(result.markers)} markers:")
        for m in result.markers:
            print(f"    [{m.strength.name:10s}] {m.content}")
        print(f"  ATP after retrieval: {atp.atp}")

        # =================================================================
        # SECTION 3: Drain to CONSERVING — Weak markers silenced
        # =================================================================
        print("\n--- Section 3: CONSERVING state — weak context silenced ---")

        atp.consume(70, "expensive_operation")
        print(f"  ATP: {atp.atp}/{atp.max_atp} → {atp.get_state().value}")

        result = histones.retrieve_context(tags=["safety"])
        print(f"  Retrieved {len(result.markers)} markers (STRONG+ only):")
        for m in result.markers:
            print(f"    [{m.strength.name:10s}] {m.content}")
        print(f"  Silenced: 'User prefers concise responses' (WEAK)")
        print(f"  Silenced: 'Always validate SQL' (MODERATE)")

        # =================================================================
        # SECTION 4: Drain to STARVING — Only permanent memories
        # =================================================================
        print("\n--- Section 4: STARVING state — only permanent memories ---")

        # Need to consume more. At this point state may block low-priority ops.
        # The energy gate consumes directly, so it still works.
        atp.consume(15, "critical_op", priority=5)
        print(f"  ATP: {atp.atp}/{atp.max_atp} → {atp.get_state().value}")

        result = histones.retrieve_context(tags=["safety"])
        print(f"  Retrieved {len(result.markers)} markers (PERMANENT only):")
        for m in result.markers:
            print(f"    [{m.strength.name:10s}] {m.content}")
        print(f"  The cell retains only its most critical constraint:")
        print(f"  'NEVER run DELETE without WHERE clause'")

        # =================================================================
        # SECTION 5: Recovery — Regenerate ATP, memories return
        # =================================================================
        print("\n--- Section 5: Recovery — regenerate ATP ---")

        atp.regenerate(80)
        print(f"  ATP: {atp.atp}/{atp.max_atp} → {atp.get_state().value}")

        result = histones.retrieve_context(tags=["safety"])
        print(f"  Retrieved {len(result.markers)} markers (full access restored):")
        for m in result.markers:
            print(f"    [{m.strength.name:10s}] {m.content}")

        # =================================================================
        # SECTION 6: Statistics
        # =================================================================
        print("\n--- Section 6: Statistics ---")

        stats = histones.get_statistics()
        print(f"  Total retrievals: {stats['total_retrievals']}")
        print(f"  Gated retrievals: {stats['gated_retrievals']}")
        print(f"  Markers by strength: {stats['by_strength']}")

        # =================================================================
        # SECTION 7: Without coupling — backward compatible
        # =================================================================
        print("\n--- Section 7: Uncoupled HistoneStore — backward compatible ---")

        uncoupled = HistoneStore(silent=True)
        uncoupled.add_marker("test", strength=MarkerStrength.WEAK, tags=["test"])
        result = uncoupled.retrieve_context(tags=["test"])
        print(f"  Uncoupled store: {len(result.markers)} markers (no ATP check)")

        print("\n" + "=" * 60)
        print("DONE — Cost-Gated Retrieval demonstrated successfully")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
