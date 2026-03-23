"""
Example 78 — Social Learning
==============================

Demonstrates cross-organism template sharing with trust-weighted adoption.
Organism A exports successful templates; Organism B imports them with
epistemic vigilance (TrustRegistry).

Usage:
    python examples/78_social_learning.py
"""

from operon_ai import (
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    TaskFingerprint,
    SocialLearning,
    TrustRegistry,
)

# ---------------------------------------------------------------------------
# 1. Organism A builds a successful template library
# ---------------------------------------------------------------------------

lib_a = PatternLibrary()
lib_a.register_template(PatternTemplate(
    template_id="customer_support",
    name="Customer Support Pipeline",
    topology="skill_organism",
    stage_specs=(
        {"name": "classify", "role": "Router", "mode": "fast"},
        {"name": "resolve", "role": "Agent", "mode": "deep"},
        {"name": "verify", "role": "QA", "mode": "fast"},
    ),
    intervention_policy={"max_retries": 2},
    fingerprint=TaskFingerprint("sequential", 3, 3, ("router", "agent", "qa")),
    tags=("support", "customer"),
))

# Record 4 successes and 1 failure (80% success rate)
for i in range(4):
    lib_a.record_run(PatternRunRecord(
        record_id=lib_a.make_id(), template_id="customer_support",
        fingerprint=TaskFingerprint("sequential", 3, 3, ("router", "agent", "qa")),
        success=True, latency_ms=1000.0, tokens_used=2000,
    ))
lib_a.record_run(PatternRunRecord(
    record_id=lib_a.make_id(), template_id="customer_support",
    fingerprint=TaskFingerprint("sequential", 3, 3, ("router", "agent", "qa")),
    success=False, latency_ms=3000.0, tokens_used=5000,
))

sl_a = SocialLearning(organism_id="organism-A", library=lib_a)

print("=== Organism A ===")
print(f"  Templates: {lib_a.summary()['template_count']}")
print(f"  Success rate: {lib_a.success_rate('customer_support'):.0%}")
print()

# ---------------------------------------------------------------------------
# 2. Organism A exports successful templates
# ---------------------------------------------------------------------------

exchange = sl_a.export_templates(min_success_rate=0.6)

print("=== Export ===")
print(f"  Peer: {exchange.peer_id}")
print(f"  Templates: {len(exchange.templates)}")
print(f"  Records: {len(exchange.records)}")
print()

# ---------------------------------------------------------------------------
# 3. Organism B imports with trust-weighted adoption
# ---------------------------------------------------------------------------

lib_b = PatternLibrary()
sl_b = SocialLearning(organism_id="organism-B", library=lib_b)

result = sl_b.import_from_peer(exchange)

print("=== Import by Organism B ===")
print(f"  Adopted: {result.adopted_template_ids}")
print(f"  Rejected: {result.rejected_template_ids}")
print(f"  Trust used: {result.trust_score_used:.2f}")
print(f"  Provenance: {sl_b.get_provenance('customer_support')}")
print()

# ---------------------------------------------------------------------------
# 4. Record adoption outcomes → trust updates
# ---------------------------------------------------------------------------

# Template works for B → trust increases
score = sl_b.record_adoption_outcome("customer_support", success=True)
print(f"After success: trust for A = {score:.3f}")

score = sl_b.record_adoption_outcome("customer_support", success=True)
print(f"After 2nd success: trust for A = {score:.3f}")

score = sl_b.record_adoption_outcome("customer_support", success=False)
print(f"After failure: trust for A = {score:.3f}")
print()

# ---------------------------------------------------------------------------
# 5. Demonstrate rejection of low-trust peer
# ---------------------------------------------------------------------------

# Create a low-trust peer
tr = TrustRegistry(default_trust=0.1, min_trust_to_adopt=0.3)
sl_c = SocialLearning(organism_id="organism-C", library=PatternLibrary(), trust=tr)
result_c = sl_c.import_from_peer(exchange)

print("=== Low-Trust Import (Organism C, trust=0.1) ===")
print(f"  Adopted: {result_c.adopted_template_ids}")
print(f"  Rejected: {result_c.rejected_template_ids}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert "customer_support" in result.adopted_template_ids
assert lib_b.get_template("customer_support") is not None
assert sl_b.get_provenance("customer_support") == "organism-A"
assert len(result_c.adopted_template_ids) == 0
print("\n--- all assertions passed ---")
