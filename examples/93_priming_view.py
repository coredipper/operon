"""
Example 93 — PrimingView
==========================

Demonstrates the multi-channel PrimingView that extends SubstrateView
with developmental status, trust context, and experience channels.

Usage:
    python examples/93_priming_view.py
"""

from datetime import datetime, UTC
from types import MappingProxyType
from operon_ai.patterns.priming import PrimingView, build_priming_view
from operon_ai.patterns.types import SubstrateView

# 1. Create a basic SubstrateView
base = SubstrateView(facts=(), query=None, record_time=datetime.now(UTC))

# 2. Promote to PrimingView with additional channels
primed = build_priming_view(
    base,
    recent_outputs=({"stage": "research", "output": "Found 3 papers"},),
    trust_context={"peer_A": 0.85, "peer_B": 0.42},
    developmental_status=None,  # would be DevelopmentStatus in practice
)

print("=== PrimingView ===")
print(f"  isinstance(SubstrateView): {isinstance(primed, SubstrateView)}")
print(f"  Facts: {len(primed.facts)}")
print(f"  Recent outputs: {len(primed.recent_outputs)}")
print(f"  Trust context: {primed.trust_context}")
print(f"  Developmental status: {primed.developmental_status}")

# 3. Direct construction
direct = PrimingView(
    facts=(),
    query="test",
    record_time=datetime.now(UTC),
    recent_outputs=(MappingProxyType({"stage": "plan", "output": "Step 1: ..."}),),
    telemetry=({"event": "stage_complete", "latency_ms": 150},),
    experience=({"action": "RETRY", "success": True},),
    trust_context=MappingProxyType({"org_1": 0.9}),
)
print(f"\nDirect PrimingView: {len(direct.telemetry)} telemetry, {len(direct.experience)} experience")

# --test
assert isinstance(primed, SubstrateView)
assert isinstance(primed, PrimingView)
assert primed.trust_context["peer_A"] == 0.85
assert len(direct.telemetry) == 1
print("\n--- all assertions passed ---")
