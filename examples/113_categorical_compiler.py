"""
Example 113: Categorical Compiler — Functorial Property Preservation
=====================================================================

Demonstrates convergence compilers formalized as functors in the
ArchAgents category (de los Riscos et al., arXiv:2603.28906).

1. Extract an Architecture from a SkillOrganism
2. Apply the Swarms functor: compile with property verification
3. Show certificate preservation (Prop 5.1)
4. Compare all four compiler functors
5. Demonstrate a failing certificate traveling through compilation

The key insight: compilers are not just serializers. They are
structure-preserving maps (functors) that carry certificates from
source to target. The preservation is verifiable, not accidental.

Usage: python examples/113_categorical_compiler.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.categorical import (
    extract_architecture,
    swarms_functor,
    deerflow_functor,
    ralph_functor,
    scion_functor,
)


def build_organism(budget: int = 1000):
    fast = Nucleus(provider=MockProvider(responses={"classify": "ROUTE: a"}))
    deep = Nucleus(provider=MockProvider(responses={"a": "done"}))
    return skill_organism(
        stages=[
            SkillStage(name="intake", role="Normalizer",
                       instructions="Normalize input.", mode="fixed"),
            SkillStage(name="router", role="Classifier",
                       instructions="Classify request.", mode="fixed"),
            SkillStage(name="executor", role="Analyst",
                       instructions="Analyze and respond.", mode="fuzzy"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=budget, silent=True),
    )


FUNCTORS = [swarms_functor, deerflow_functor, ralph_functor, scion_functor]


def main():
    print("=" * 60)
    print("Categorical Compiler — Functorial Property Preservation")
    print("=" * 60)

    org = build_organism()

    # -----------------------------------------------------------------------
    # 1. Extract Architecture from organism
    # -----------------------------------------------------------------------
    print("\n--- Source Architecture ---")
    arch = extract_architecture(org)
    print(f"  stages:      {arch.stage_names}")
    print(f"  edges:       {arch.edges}")
    print(f"  sequential:  {arch.is_sequential}")
    print(f"  certificates: {sorted(arch.certificate_theorems)}")
    print(f"  interface:   {dict(arch.interface)}")

    # -----------------------------------------------------------------------
    # 2. Apply all four functors
    # -----------------------------------------------------------------------
    print("\n--- Functor Applications ---")
    results = {}
    for functor in FUNCTORS:
        result = functor.compile(org)
        results[functor.name] = result
        p = result.preservation

        print(f"\n  F = {functor.name}:")
        print(f"    target stages:  {result.target_architecture.stage_names}")
        print(f"    graph:          {'preserved' if p.graph_preserved else 'enriched'}")
        print(f"    certificates:   {'preserved' if p.certificate_preserved else 'LOST'}")
        print(f"    interface:      {'preserved' if p.interface_preserved else 'remapped'}")

        for v in p.certificate_verifications:
            status = "HOLDS" if v.holds else "FAILS"
            print(f"    {v.certificate.theorem}: {status}")

    # -----------------------------------------------------------------------
    # 3. Certificate preservation (Prop 5.1)
    # -----------------------------------------------------------------------
    print("\n--- Prop 5.1: Certificate Preservation ---")
    print("  Source theorems:", sorted(arch.certificate_theorems))
    for name, result in results.items():
        target_theorems = result.target_architecture.certificate_theorems
        preserved = arch.certificate_theorems <= target_theorems
        print(f"  {name}: {sorted(target_theorems)} — {'preserved' if preserved else 'LOST'}")

    # -----------------------------------------------------------------------
    # 4. Failing certificate propagation
    # -----------------------------------------------------------------------
    print("\n--- Failing Certificate Propagation ---")
    empty_org = build_organism(budget=0)
    result = swarms_functor.compile(empty_org)
    for v in result.preservation.certificate_verifications:
        print(f"  {v.certificate.theorem}: holds={v.holds}")
    print(f"  Certificate exists in target but honestly reports failure")

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------
    print("\n--- Assertions ---")

    # Source architecture is well-formed
    assert arch.stage_count == 3
    assert arch.is_sequential
    assert "priority_gating" in arch.certificate_theorems

    # All functors preserve certificates (Prop 5.1)
    for name, result in results.items():
        assert arch.certificate_theorems <= result.target_architecture.certificate_theorems, (
            f"{name} lost certificates"
        )

    # Swarms functor preserves everything (1:1 compiler)
    swarms_result = results["swarms"]
    assert swarms_result.preservation.all_preserved

    # Failing certificate propagates honestly
    empty_result = swarms_functor.compile(build_organism(budget=0))
    assert not all(v.holds for v in empty_result.preservation.certificate_verifications)

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
