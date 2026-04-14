"""
Example 118: Harness-Architecture Mapping
===========================================

Demonstrates the central claim of Paper 5: a SkillOrganism's
harness maps to a categorical Architecture triple (G, Know, Φ).

1. Build an organism with budget and components
2. Extract the Architecture triple via extract_architecture()
3. Display G (syntactic wiring), Know (certificates), Φ (interface)
4. Map to the four-pillar framework (Memory/Skills/Protocols/Harness)
5. Apply a compiler functor and show property preservation

The Architecture is an object in the ArchAgents category
(de los Riscos et al., arXiv:2603.28906). Compilers are functors
that preserve the Architecture's structural properties.

See also:
  Liu (arXiv:2604.11767) — alternative formalization via typed λ-calculus
  NLAH (arXiv:2603.25723) — validates harness-as-portable-artifact

Usage: python examples/118_harness_architecture_mapping.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.watcher import WatcherComponent, WatcherConfig
from operon_ai.patterns.verifier import VerifierComponent
from operon_ai.convergence.categorical import (
    extract_architecture,
    swarms_functor,
    deerflow_functor,
)


def main():
    print("=" * 60)
    print("Harness-Architecture Mapping")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build organism with budget + components
    # ------------------------------------------------------------------
    fast = Nucleus(provider=MockProvider(responses={"classify": "ROUTE: a"}))
    deep = Nucleus(provider=MockProvider(responses={"a": "done"}))

    org = skill_organism(
        stages=[
            SkillStage(name="intake", role="Normalizer",
                       instructions="Normalize input.", mode="fixed"),
            SkillStage(name="router", role="Classifier",
                       instructions="Classify request.", mode="fixed"),
            SkillStage(name="executor", role="Engineer",
                       instructions="Execute task.", mode="fuzzy"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=1000, silent=True),
        components=[
            WatcherComponent(config=WatcherConfig()),
            VerifierComponent(),
        ],
    )

    # ------------------------------------------------------------------
    # 2. Extract Architecture triple
    # ------------------------------------------------------------------
    arch = extract_architecture(org)

    print("\n--- Architecture Triple: (G, Know, Φ) ---")

    # G — Syntactic wiring (graph structure)
    print("\n  G (Graph — syntactic wiring):")
    print(f"    stages:     {list(arch.stage_names)}")
    print(f"    edges:      {list(arch.edges)}")
    print(f"    sequential: {arch.is_sequential}")
    print(f"    |V|={arch.stage_count}, |E|={len(arch.edges)}")

    # Know — Knowledge structure (certificates)
    print("\n  Know (Knowledge — structural guarantees):")
    print(f"    theorems:   {sorted(arch.certificate_theorems)}")
    for cert_dict in arch.certificates:
        print(f"    {cert_dict['theorem']}: {cert_dict['conclusion']}")

    # Φ — Profunctor interface (mode → model mapping)
    print("\n  Φ (Interface — mode-to-model mapping):")
    for stage_name, mode in arch.interface:
        tier = {"fixed": "fast", "fuzzy": "fast→deep", "deep": "deep"}.get(mode, mode)
        print(f"    {stage_name:12s} mode={mode:6s} → tier={tier}")

    # ------------------------------------------------------------------
    # 3. Four-pillar mapping
    # ------------------------------------------------------------------
    print("\n--- Four-Pillar Mapping ---")
    print("  (de los Riscos et al. §3 ↔ NLAH §2)")
    print()
    print("  Pillar          Architecture Component   Operon Realization")
    print("  ─────────────── ────────────────────────  ──────────────────────")
    print("  Harness         G (graph)                 SkillOrganism.stages + edges")
    print("  Protocols       Know (certificates)       ATP_Store.certify → Certificate")
    print("  Skills          Know (certificates)       SkillStage.role + handler")
    print("  Memory          Φ (interface)             mode → nucleus mapping")

    # ------------------------------------------------------------------
    # 4. Apply compiler functors
    # ------------------------------------------------------------------
    print("\n--- Functor Applications ---")
    for functor in [swarms_functor, deerflow_functor]:
        result = functor.compile(org)
        p = result.preservation

        print(f"\n  F = {functor.name}:")
        print(f"    target stages:  {list(result.target_architecture.stage_names)}")
        print(f"    graph:          {'preserved' if p.graph_preserved else 'enriched'}")
        print(f"    certificates:   {'preserved' if p.certificate_preserved else 'LOST'}")
        print(f"    interface:      {'preserved' if p.interface_preserved else 'remapped'}")
        print(f"    all_preserved:  {p.all_preserved}")

    # ------------------------------------------------------------------
    # 5. Show the formal relationship
    # ------------------------------------------------------------------
    print("\n--- Formal Summary ---")
    print(f"  Organism → Architecture = ({arch.stage_count} stages, "
          f"{len(arch.certificates)} certs, "
          f"{len(arch.interface)} interface mappings)")
    print(f"  Functor F: Arch(Operon) → Arch(Target)")
    print(f"  Prop 5.1: ∀ cert ∈ Know(source), cert ∈ Know(F(source))")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    # Architecture is well-formed
    assert arch.stage_count == 3
    assert arch.is_sequential
    assert len(arch.edges) == 2
    assert ("intake", "router") in arch.edges
    assert ("router", "executor") in arch.edges

    # Certificates exist
    assert "priority_gating" in arch.certificate_theorems

    # Interface maps all stages
    interface_stages = {name for name, _ in arch.interface}
    assert interface_stages == set(arch.stage_names)

    # Functors preserve
    for functor in [swarms_functor, deerflow_functor]:
        result = functor.compile(org)
        assert result.preservation.certificate_preserved

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
