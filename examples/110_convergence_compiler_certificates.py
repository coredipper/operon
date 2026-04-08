"""
Example 110: Convergence Compiler Certificate Preservation
==========================================================

Demonstrates that structural certificates survive compilation to
external frameworks (Swarms, DeerFlow, Ralph, Scion).

1. Build a 3-stage SkillOrganism with an ATP budget
2. Collect certificates from the organism
3. Compile to all four targets
4. Verify certificates are preserved in the compiled output
5. Show a failing scenario: budget=0 certificate travels but fails

This validates Prop 5.1: structural guarantees are functorially
stable under compilation — the compiler preserves, not creates.

Usage: python examples/110_convergence_compiler_certificates.py
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
from operon_ai.core.certificate import certificate_to_dict, verify_compiled


def build_organism(budget: int = 1000):
    """Build a 3-stage organism with ATP budget."""
    fast = Nucleus(provider=MockProvider(responses={
        "classify": "ROUTE: analysis",
    }))
    deep = Nucleus(provider=MockProvider(responses={
        "analysis": "Detailed analysis of the request...",
    }))

    return skill_organism(
        stages=[
            SkillStage(
                name="intake",
                role="Normalizer",
                handler=lambda task, state, outputs, stage: {
                    "request": task, "channel": "api",
                },
            ),
            SkillStage(
                name="router",
                role="Classifier",
                instructions="Classify the request.",
                mode="fixed",
            ),
            SkillStage(
                name="executor",
                role="Analyst",
                instructions="Analyze and respond.",
                mode="fuzzy",
            ),
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
    print("Convergence Compiler Certificate Preservation")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Build organism and collect certificates
    # -----------------------------------------------------------------------
    org = build_organism(budget=1000)
    certs = org.collect_certificates()

    print(f"\n--- Organism Certificates ({len(certs)}) ---")
    for cert in certs:
        v = cert.verify()
        print(f"  {cert.theorem}: holds={v.holds}")

    # -----------------------------------------------------------------------
    # 2. Compile to all four targets
    # -----------------------------------------------------------------------
    print("\n--- Compile & Verify ---")

    compiled_outputs = {}
    for name, compiler in COMPILERS.items():
        compiled = compiler(org)
        compiled_outputs[name] = compiled

        cert_dicts = compiled.get("certificates", [])
        print(f"\n  {name}:")
        print(f"    agents:       {len(compiled.get('agents', []))}")
        print(f"    certificates: {len(cert_dicts)}")

        # Verify certificates in compiled output
        verifications = verify_compiled(compiled)
        for v in verifications:
            status = "HOLDS" if v.holds else "FAILS"
            print(f"    {v.certificate.theorem}: {status}")

    # -----------------------------------------------------------------------
    # 3. Failing scenario: budget=0
    # -----------------------------------------------------------------------
    print("\n--- Failing Budget (budget=0) ---")

    empty_org = build_organism(budget=0)
    empty_certs = empty_org.collect_certificates()

    for cert in empty_certs:
        v = cert.verify()
        print(f"  pre-compile:  {cert.theorem} holds={v.holds}")

    for name, compiler in COMPILERS.items():
        compiled = compiler(empty_org)
        verifications = verify_compiled(compiled)
        for v in verifications:
            print(f"  {name}: {v.certificate.theorem} holds={v.holds}")

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------
    print("\n--- Assertions ---")

    # Budget=1000 certificates hold pre and post compilation
    assert len(certs) >= 1, "organism should have at least 1 certificate"
    for cert in certs:
        assert cert.verify().holds

    for name, compiled in compiled_outputs.items():
        assert "certificates" in compiled, f"{name} must have certificates key"
        compiled_certs = compiled["certificates"]
        assert len(compiled_certs) == len(certs), (
            f"{name} should preserve all {len(certs)} certificates, got {len(compiled_certs)}"
        )
        original_theorems = {c.theorem for c in certs}
        compiled_theorems = {c["theorem"] for c in compiled_certs}
        assert original_theorems == compiled_theorems, (
            f"{name} theorem mismatch: {original_theorems} vs {compiled_theorems}"
        )
        verifications = verify_compiled(compiled)
        for v in verifications:
            assert v.holds, f"{name}/{v.certificate.theorem} should hold"

    # Budget=0 certificates exist but fail
    for cert in empty_certs:
        assert not cert.verify().holds

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
