"""
Example 117: Certificate Round-Trip
=====================================

Demonstrates that structural certificates survive compilation
*and* decompilation — the full round-trip.

1. Build a 3-stage organism with an ATP budget (→ priority_gating cert)
2. Compile to all 4 dict-based targets (Swarms, DeerFlow, Ralph, Scion)
3. Verify certificates hold in each compiled output
4. Decompile Swarms and DeerFlow back to ExternalTopology
5. Verify certificates survive the round-trip
6. Show certificate serialization (dict ↔ Certificate round-trip)

This validates Prop 5.1: structural guarantees are functorially
stable under compilation — the compiler preserves, not creates.

Usage: python examples/117_certificate_round_trip.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence import (
    organism_to_swarms,
    organism_to_deerflow,
    organism_to_ralph,
    organism_to_scion,
)
from operon_ai.convergence.swarms_compiler import swarms_to_topology
from operon_ai.convergence.deerflow_compiler import deerflow_to_topology
from operon_ai.core.certificate import (
    certificate_to_dict,
    certificate_from_dict,
    verify_compiled,
)


def build_organism(budget: int = 1000):
    fast = Nucleus(provider=MockProvider(responses={"classify": "ROUTE: a"}))
    deep = Nucleus(provider=MockProvider(responses={"a": "done"}))
    return skill_organism(
        stages=[
            SkillStage(name="intake", role="Normalizer",
                       instructions="Normalize.", mode="fixed"),
            SkillStage(name="router", role="Classifier",
                       instructions="Route.", mode="fixed"),
            SkillStage(name="executor", role="Worker",
                       instructions="Execute.", mode="fuzzy"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=budget, silent=True),
    )


COMPILERS = {
    "Swarms": organism_to_swarms,
    "DeerFlow": organism_to_deerflow,
    "Ralph": organism_to_ralph,
    "Scion": organism_to_scion,
}


def main():
    print("=" * 60)
    print("Certificate Round-Trip")
    print("=" * 60)

    org = build_organism()
    source_certs = org.collect_certificates()
    source_theorems = {c.theorem for c in source_certs}

    # ------------------------------------------------------------------
    # 1. Source certificates
    # ------------------------------------------------------------------
    print(f"\n--- Source Certificates ({len(source_certs)}) ---")
    for cert in source_certs:
        v = cert.verify()
        print(f"  {cert.theorem}: holds={v.holds}  source={cert.source}")

    # ------------------------------------------------------------------
    # 2. Compile to all 4 targets → verify
    # ------------------------------------------------------------------
    print("\n--- Forward: Compile & Verify ---")
    compiled_outputs = {}
    for name, compiler in COMPILERS.items():
        compiled = compiler(org)
        compiled_outputs[name] = compiled

        verifications = verify_compiled(compiled)
        all_hold = all(v.holds for v in verifications)
        compiled_theorems = {c["theorem"] for c in compiled.get("certificates", [])}

        print(f"  {name:10s}  certs={len(verifications)}  "
              f"all_hold={all_hold}  "
              f"theorems_preserved={source_theorems <= compiled_theorems}")

    # ------------------------------------------------------------------
    # 3. Decompile: Swarms → ExternalTopology
    # ------------------------------------------------------------------
    print("\n--- Round-Trip: Swarms ---")
    swarms_compiled = compiled_outputs["Swarms"]
    swarms_topo = swarms_to_topology(swarms_compiled)

    print(f"  source:       {swarms_topo.source}")
    print(f"  agents:       {[a['name'] for a in swarms_topo.agents]}")
    print(f"  edges:        {list(swarms_topo.edges)}")

    swarms_rt_certs = swarms_topo.metadata.get("certificates", [])
    swarms_rt_theorems = {c["theorem"] for c in swarms_rt_certs}
    print(f"  certificates: {sorted(swarms_rt_theorems)}")
    print(f"  preserved:    {source_theorems <= swarms_rt_theorems}")

    # ------------------------------------------------------------------
    # 4. Decompile: DeerFlow → ExternalTopology
    # ------------------------------------------------------------------
    print("\n--- Round-Trip: DeerFlow ---")
    deerflow_compiled = compiled_outputs["DeerFlow"]
    deerflow_topo = deerflow_to_topology(deerflow_compiled)

    print(f"  source:       {deerflow_topo.source}")
    print(f"  agents:       {[a['name'] for a in deerflow_topo.agents]}")
    print(f"  edges:        {list(deerflow_topo.edges)}")

    deerflow_rt_certs = deerflow_topo.metadata.get("certificates", [])
    deerflow_rt_theorems = {c["theorem"] for c in deerflow_rt_certs}
    print(f"  certificates: {sorted(deerflow_rt_theorems)}")
    print(f"  preserved:    {source_theorems <= deerflow_rt_theorems}")

    # ------------------------------------------------------------------
    # 5. Certificate serialization round-trip
    # ------------------------------------------------------------------
    print("\n--- Serialization Round-Trip ---")
    for cert in source_certs:
        d = certificate_to_dict(cert)
        restored = certificate_from_dict(d)
        v_orig = cert.verify()
        v_restored = restored.verify()
        match = v_orig.holds == v_restored.holds
        print(f"  {cert.theorem}: dict→Certificate→verify  match={match}")

    # ------------------------------------------------------------------
    # 6. Failing budget: certificate travels but fails
    # ------------------------------------------------------------------
    print("\n--- Failing Certificate Propagation ---")
    empty_org = build_organism(budget=0)
    empty_compiled = organism_to_swarms(empty_org)
    empty_verifications = verify_compiled(empty_compiled)
    for v in empty_verifications:
        print(f"  {v.certificate.theorem}: holds={v.holds} (budget=0 → honest failure)")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    # Source has certificates
    assert len(source_certs) >= 1

    # All 4 compilers preserve certificates
    for name, compiled in compiled_outputs.items():
        compiled_theorems = {c["theorem"] for c in compiled.get("certificates", [])}
        assert source_theorems <= compiled_theorems, f"{name} lost certificates"
        for v in verify_compiled(compiled):
            assert v.holds, f"{name}/{v.certificate.theorem} should hold"

    # Swarms round-trip preserves
    assert source_theorems <= swarms_rt_theorems, "Swarms round-trip lost certs"

    # DeerFlow round-trip preserves
    assert source_theorems <= deerflow_rt_theorems, "DeerFlow round-trip lost certs"

    # Serialization round-trip
    for cert in source_certs:
        d = certificate_to_dict(cert)
        restored = certificate_from_dict(d)
        assert cert.verify().holds == restored.verify().holds

    # Failing budget propagates honestly
    for v in empty_verifications:
        assert not v.holds, "budget=0 certificate should fail"

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
