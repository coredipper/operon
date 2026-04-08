"""
Example 109: Certificate Verification Framework
================================================

Demonstrates the categorical certificate framework (v0.28.0):

1. Issue certificates from ATP_Store, QuorumSensingBio, and MTORScaler
2. Verify each certificate — derivation replay
3. Serialize and deserialize via certificate_to_dict / certificate_from_dict
4. Prove immutability — parameters resist tampering
5. Show a failing certificate — budget=0 means priority_gating fails

Certificates are self-verifiable: verify() re-derives the guarantee from
parameters rather than trusting a boolean flag.  If conditions change,
the certificate honestly reports that the guarantee no longer holds.

Usage: python examples/109_certificate_verification.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, QuorumSensingBio
from operon_ai.state.mtor import MTORScaler
from operon_ai.core.certificate import (
    certificate_to_dict,
    certificate_from_dict,
)


def main():
    # -----------------------------------------------------------------------
    # 1. Create certifiable components
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Certificate Verification Framework")
    print("=" * 60)

    # ATP_Store: priority gating
    atp = ATP_Store(budget=1000, silent=True)
    atp.consume(100, "warmup", priority=5)

    # QuorumSensing: no false activation
    qs = QuorumSensingBio(population_size=10)
    qs.calibrate()

    # MTOR: no oscillation via hysteresis
    mtor = MTORScaler(atp_store=atp)

    # -----------------------------------------------------------------------
    # 2. Issue and verify certificates
    # -----------------------------------------------------------------------
    print("\n--- Issue & Verify ---")

    atp_cert = atp.certify()
    qs_cert = qs.certify()
    mtor_cert = mtor.certify()

    certs = [
        ("ATP priority_gating", atp_cert),
        ("QS no_false_activation", qs_cert),
        ("MTOR no_oscillation", mtor_cert),
    ]

    for label, cert in certs:
        v = cert.verify()
        status = "HOLDS" if v.holds else "FAILS"
        print(f"\n  {label}:")
        print(f"    theorem:    {cert.theorem}")
        print(f"    conclusion: {cert.conclusion}")
        print(f"    status:     {status}")
        print(f"    evidence:   {dict(v.evidence)}")

    # -----------------------------------------------------------------------
    # 3. Serialization round-trip
    # -----------------------------------------------------------------------
    print("\n--- Serialization Round-Trip ---")

    for label, cert in certs:
        d = certificate_to_dict(cert)
        cert2 = certificate_from_dict(d)
        v1 = cert.verify()
        v2 = cert2.verify()
        match = "✓" if v1.holds == v2.holds else "✗"
        print(f"  {match} {label}: original={v1.holds}, deserialized={v2.holds}")

    # -----------------------------------------------------------------------
    # 4. Immutability proof
    # -----------------------------------------------------------------------
    print("\n--- Immutability Proof ---")

    try:
        atp_cert.parameters["budget"] = 0  # type: ignore[index]
        print("  ✗ parameters were mutated (should not happen)")
    except TypeError:
        print("  ✓ parameters are frozen (TypeError on mutation attempt)")

    try:
        atp_cert.theorem = "fake"  # type: ignore[misc]
        print("  ✗ theorem was mutated (should not happen)")
    except AttributeError:
        print("  ✓ certificate is frozen (AttributeError on field assignment)")

    # -----------------------------------------------------------------------
    # 5. Failing certificate: budget=0
    # -----------------------------------------------------------------------
    print("\n--- Failing Certificate ---")

    empty_atp = ATP_Store(budget=0, silent=True)
    empty_cert = empty_atp.certify()
    v = empty_cert.verify()
    print(f"  budget=0 → holds={v.holds}")
    print(f"  evidence: {dict(v.evidence)}")

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------
    print("\n--- Assertions ---")

    # All three valid certificates hold
    for label, cert in certs:
        assert cert.verify().holds, f"{label} should hold"

    # Serialization round-trips preserve holds
    for label, cert in certs:
        d = certificate_to_dict(cert)
        cert2 = certificate_from_dict(d)
        assert cert.verify().holds == cert2.verify().holds

    # Immutability
    immutable = False
    try:
        atp_cert.parameters["budget"] = 0  # type: ignore[index]
    except TypeError:
        immutable = True
    assert immutable, "parameters must be frozen"

    # Failing certificate
    assert not empty_cert.verify().holds, "budget=0 should fail"

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
