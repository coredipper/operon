"""
Example 70: Bi-Temporal Compliance Audit
========================================

Demonstrates how bi-temporal memory supports regulatory auditing.

Story — Financial product approval:

A compliance team must approve a financial product.  The decision
depends on three facts (risk score, liquidity class, regulatory
category) that arrive at different times and may be corrected
after the approval decision has already been made.

The audit reconstructs the belief state at decision time to answer:
"Given what the system knew when the approval was made, was the
decision justified?"

This exercises multiple subjects/predicates, belief-state
reconstruction, ``history``, ``diff_between``, and ``timeline_for``.

Biological Analogy:
In an immune response, a cell commits to an action (e.g. producing
antibodies) based on partial, time-delayed signals.  If a later
signal contradicts an earlier one, the cell does not pretend it
always knew — it records the correction alongside the original.
Bi-temporal memory gives AI agents the same ability.

References:
- Roadmap Phase 1: docs/plans/roadmap.md
- Spec: docs/plans/2026-03-16-bitemporal-memory-implementation.md
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from operon_ai import BiTemporalMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day(n: int) -> datetime:
    """Return a deterministic date: 2025-01-01 + n days."""
    return datetime(2025, 1, 1) + timedelta(days=n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mem = BiTemporalMemory()
    product = "product:BOND-7Y"

    # --- Fact ingestion timeline ---
    # Day 1: risk score assessed in real world
    # Day 2: liquidity class assessed
    # Day 3: regulatory category assigned
    # Day 5: system ingests risk score
    # Day 6: system ingests liquidity class
    # Day 7: system ingests regulatory category
    # Day 8: approval decision made
    # Day 10: post-approval correction — risk score was actually higher

    risk = mem.record_fact(
        subject=product,
        predicate="risk_score",
        value=42,
        valid_from=_day(1),
        recorded_from=_day(5),
        source="risk_engine",
        tags=("quantitative",),
    )
    liquidity = mem.record_fact(
        subject=product,
        predicate="liquidity_class",
        value="A",
        valid_from=_day(2),
        recorded_from=_day(6),
        source="treasury",
    )
    reg = mem.record_fact(
        subject=product,
        predicate="regulatory_category",
        value="standard",
        valid_from=_day(3),
        recorded_from=_day(7),
        source="compliance_db",
    )

    t_decision = _day(8)
    t_correction = _day(10)

    print("=" * 60)
    print("Bi-Temporal Compliance Audit")
    print("=" * 60)

    # --- Belief state at decision time ---
    print("\n--- Belief state at decision time (day 8) ---")
    belief = mem.retrieve_belief_state(at_valid=_day(4), at_record=t_decision)
    for f in belief:
        print(f"  {f.predicate:25s} = {f.value!r:<12}  (source: {f.source})")

    # --- Post-approval correction ---
    print(f"\nCorrection on day 10: risk_score was actually 78 (not 42)")
    correction = mem.correct_fact(
        old_fact_id=risk.fact_id,
        value=78,
        valid_from=_day(1),
        recorded_from=t_correction,
        source="manual_review",
        confidence=0.95,
        tags=("audit", "correction"),
    )

    # --- Reconstructed belief at decision time ---
    print("\n--- Reconstructed belief at decision time (day 8) ---")
    print("   (what the system knew when the approval was made)")
    belief_at_decision = mem.retrieve_belief_state(
        at_valid=_day(4), at_record=t_decision,
    )
    for f in belief_at_decision:
        print(f"  {f.predicate:25s} = {f.value!r:<12}  (source: {f.source})")

    # --- Current belief ---
    print("\n--- Current belief (after correction) ---")
    belief_now = mem.retrieve_belief_state(
        at_valid=_day(4), at_record=_day(11),
    )
    for f in belief_now:
        print(f"  {f.predicate:25s} = {f.value!r:<12}  (source: {f.source})")

    # --- History of risk_score ---
    print("\n--- History of risk_score ---")
    for h in mem.history(product, predicate="risk_score"):
        status = "closed" if h.recorded_to else "active"
        print(f"  [{status}] value={h.value!r:<6}  "
              f"recorded day {h.recorded_from.day}  "
              f"(supersedes: {h.supersedes})")

    # --- Diff: what changed between day 8 and day 11 on record axis? ---
    print("\n--- Record-time diff (day 8 -> day 11) ---")
    diff = mem.diff_between(_day(8), _day(11), axis="record")
    for d in diff:
        print(f"  NEW: {d.predicate} = {d.value!r}  (source: {d.source})")

    # --- Timeline for the product ---
    print("\n--- World-time timeline ---")
    for t in mem.timeline_for(product):
        status = "closed" if t.recorded_to else "active"
        print(f"  [{status}] {t.predicate:25s} = {t.value!r:<12}  "
              f"valid from day {t.valid_from.day}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Automated smoke test for CI."""
    mem = BiTemporalMemory()
    product = "product:BOND-7Y"

    risk = mem.record_fact(product, "risk_score", 42, _day(1), _day(5), "risk_engine")
    mem.record_fact(product, "liquidity_class", "A", _day(2), _day(6), "treasury")
    mem.record_fact(product, "regulatory_category", "standard", _day(3), _day(7), "compliance_db")

    t_decision = _day(8)

    # At decision time, all three facts are known and valid
    belief = mem.retrieve_belief_state(at_valid=_day(4), at_record=t_decision)
    assert len(belief) == 3, f"Expected 3 facts at decision time, got {len(belief)}"

    # Correct risk score
    correction = mem.correct_fact(
        old_fact_id=risk.fact_id,
        value=78,
        valid_from=_day(1),
        recorded_from=_day(10),
        source="manual_review",
    )

    # Reconstructed belief at decision time still shows old risk score
    belief_at_decision = mem.retrieve_belief_state(at_valid=_day(4), at_record=t_decision)
    risk_facts = [f for f in belief_at_decision if f.predicate == "risk_score"]
    assert len(risk_facts) == 1 and risk_facts[0].value == 42, (
        f"Decision-time risk should be 42, got {risk_facts}"
    )

    # Current belief shows corrected risk score
    belief_now = mem.retrieve_belief_state(at_valid=_day(4), at_record=_day(11))
    risk_now = [f for f in belief_now if f.predicate == "risk_score"]
    assert len(risk_now) == 1 and risk_now[0].value == 78, (
        f"Current risk should be 78, got {risk_now}"
    )

    # History has both versions
    hist = mem.history(product, predicate="risk_score")
    assert len(hist) == 2, f"Expected 2 history entries, got {len(hist)}"

    # Record-time diff catches the correction
    diff = mem.diff_between(_day(8), _day(11), axis="record")
    assert any(d.value == 78 for d in diff), "Diff should include corrected value"

    # Timeline includes all facts
    tl = mem.timeline_for(product)
    assert len(tl) == 4, f"Expected 4 timeline entries, got {len(tl)}"

    print("\n[OK] Bi-temporal compliance audit example completed.")


if __name__ == "__main__":
    try:
        main()
        if "--test" in sys.argv:
            run_smoke_test()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
