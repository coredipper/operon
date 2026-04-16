"""
Example 122: Behavioral Certificates
=======================================

Demonstrates behavioral certificates -- runtime quality guarantees
that extend the structural certificate framework. While structural
certificates verify configuration invariants (priority gating,
no oscillation), behavioral certificates verify output quality
from evidence collected during execution.

Three behavioral certificate types:
1. behavioral_quality   -- mean rubric score >= threshold
2. behavioral_stability -- mean signal severity < threshold
3. behavioral_no_anomaly -- no confirmed threats (standalone)

Usage: python examples/122_behavioral_certificates.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, Nucleus, SkillStage, skill_organism
from operon_ai.core.certificate import certificate_to_dict
from operon_ai.patterns.verifier import VerifierComponent
from operon_ai.patterns.watcher import WatcherComponent, WatcherConfig
from operon_ai.providers.mock import MockProvider


def main():
    # -- Setup ---------------------------------------------------------------

    provider = MockProvider()
    fast = Nucleus(provider=provider, base_energy_cost=10)
    deep = Nucleus(provider=provider, base_energy_cost=30)
    budget = ATP_Store(budget=500, silent=True)

    # Rubric: keyword-based quality scorer
    def rubric(output: str, stage_name: str) -> float:
        if "error" in output.lower():
            return 0.3
        return 0.9

    verifier = VerifierComponent(rubric=rubric)
    watcher = WatcherComponent(config=WatcherConfig(), budget=budget)

    organism = skill_organism(
        stages=[
            SkillStage(name="plan", role="planner",
                       instructions="Plan the approach.", mode="fast"),
            SkillStage(name="execute", role="executor",
                       instructions="Execute the plan.", mode="fast"),
            SkillStage(name="review", role="reviewer",
                       instructions="Review the output.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[watcher, verifier],
        budget=budget,
    )

    # -- Run -----------------------------------------------------------------

    print("Running organism...")
    result = organism.run("Implement a fibonacci function")
    print(f"Final output: {result.final_output[:80]}...")

    # -- Structural certificates (always available) --------------------------

    print("\n--- Structural Certificates ---")
    structural_cert = budget.certify()
    sv = structural_cert.verify()
    print(f"  {structural_cert.theorem}: holds={sv.holds}")
    print(f"  conclusion: {structural_cert.conclusion}")

    # -- Behavioral certificates (from run evidence) -------------------------

    print("\n--- Behavioral Certificates ---")

    # 1. Quality certificate from verifier
    quality_cert = verifier.certify_behavior(threshold=0.8)
    if quality_cert:
        qv = quality_cert.verify()
        print(f"\n  behavioral_quality:")
        print(f"    holds={qv.holds}")
        print(f"    evidence: mean={qv.evidence['mean']}, "
              f"min={qv.evidence['min']}, n={qv.evidence['n']}")
        print(f"    conclusion: {quality_cert.conclusion}")

    # 2. Stability certificate from watcher
    stability_cert = watcher.certify_behavior(category="epistemic", threshold=0.5)
    if stability_cert:
        stv = stability_cert.verify()
        print(f"\n  behavioral_stability:")
        print(f"    holds={stv.holds}")
        print(f"    evidence: mean={stv.evidence['mean']}, "
              f"max={stv.evidence['max']}, n={stv.evidence['n']}")
        print(f"    conclusion: {stability_cert.conclusion}")
    else:
        print("\n  behavioral_stability: no epistemic signals collected")

    # -- All certificates via collect_certificates() -------------------------

    print("\n--- All Certificates (collect_certificates) ---")
    all_certs = organism.collect_certificates()
    for cert in all_certs:
        v = cert.verify()
        print(f"  {cert.theorem}: holds={v.holds} [{cert.source}]")

    # -- Serialization -------------------------------------------------------

    print("\n--- Serialized ---")
    for cert in all_certs:
        d = certificate_to_dict(cert)
        print(f"  {d['theorem']}: {d['conclusion']}")

    # -- Verify all hold -----------------------------------------------------

    all_hold = all(cert.verify().holds for cert in all_certs)
    print(f"\nAll certificates hold: {all_hold}")


if __name__ == "__main__":
    main()
