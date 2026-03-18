"""
Example 69: Bi-Temporal Memory
==============================

Demonstrates the difference between *valid time* (when a fact is true
in the world) and *record time* (when the system learned about it).

Story — Client risk-tier compliance:

1. Day 1: client risk tier becomes "medium" in the real world
2. Day 3: system ingests the fact
3. Day 5: manual review corrects the tier to "high", retroactive to day 1

The example prints queries that show how the two time axes diverge and
how belief-state reconstruction returns different answers depending on
when you "look" along each axis.

Biological Analogy:
Just as a cell does not erase old synaptic weights when new evidence
arrives — it forms new connections while the old ones decay — bi-temporal
memory keeps every version of every fact.  The relative "active" status
of each record determines which version the system acts on at any given
point in time.

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

    day1, day2, day3, day4, day5, day6 = (_day(d) for d in range(1, 7))

    # Step 1 — record initial fact: risk tier is "medium" from day 1,
    #          but the system only learns this on day 3.
    fact = mem.record_fact(
        subject="client:42",
        predicate="risk_tier",
        value="medium",
        valid_from=day1,
        recorded_from=day3,
        source="crm",
    )

    print("=" * 60)
    print("Bi-Temporal Memory — Risk-Tier Compliance")
    print("=" * 60)
    print(f"\nRecorded: risk_tier = 'medium' (valid from day 1, ingested day 3)")
    print(f"  fact_id: {fact.fact_id}")

    # Step 2 — query *before* correction

    print("\n--- Before correction ---")

    valid_day2 = mem.retrieve_valid_at(at=day2, subject="client:42")
    print(f"  retrieve_valid_at(day 2):  {[f.value for f in valid_day2]}")

    known_day2 = mem.retrieve_known_at(at=day2, subject="client:42")
    print(f"  retrieve_known_at(day 2):  {[f.value for f in known_day2]}")

    known_day4 = mem.retrieve_known_at(at=day4, subject="client:42")
    print(f"  retrieve_known_at(day 4):  {[f.value for f in known_day4]}")

    belief_day2_day4 = mem.retrieve_belief_state(at_valid=day2, at_record=day4)
    print(f"  belief_state(v=2, r=4):    {[f.value for f in belief_day2_day4]}")

    # Step 3 — correction: manual review on day 5 changes tier to "high",
    #          retroactive to day 1.
    correction = mem.correct_fact(
        old_fact_id=fact.fact_id,
        value="high",
        valid_from=day1,
        recorded_from=day5,
        source="manual_review",
    )

    print(f"\nCorrected: risk_tier = 'high' (retroactive to day 1, recorded day 5)")
    print(f"  new fact_id: {correction.new_fact.fact_id}")
    print(f"  supersedes:  {correction.new_fact.supersedes}")

    # Step 4 — query *after* correction

    print("\n--- After correction ---")

    valid_day2_after = mem.retrieve_valid_at(at=day2, subject="client:42")
    print(f"  retrieve_valid_at(day 2):  {[f.value for f in valid_day2_after]}")

    belief_day2_day4_after = mem.retrieve_belief_state(
        at_valid=day2, at_record=day4,
    )
    print(f"  belief_state(v=2, r=4):    {[f.value for f in belief_day2_day4_after]}")

    belief_day2_day6 = mem.retrieve_belief_state(
        at_valid=day2, at_record=day6,
    )
    print(f"  belief_state(v=2, r=6):    {[f.value for f in belief_day2_day6]}")

    # Step 5 — history and timeline

    print("\n--- History ---")
    for h in mem.history("client:42"):
        status = "closed" if h.recorded_to else "active"
        print(f"  [{status}] {h.predicate} = {h.value!r}  "
              f"(valid {h.valid_from.day}, recorded {h.recorded_from.day})")

    print("\n--- Timeline ---")
    for t in mem.timeline_for("client:42"):
        print(f"  {t.predicate} = {t.value!r}  "
              f"(valid_from day {t.valid_from.day}, source={t.source})")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Automated smoke test for CI."""
    mem = BiTemporalMemory()
    day1, day2, day3, day4, day5, day6 = (_day(d) for d in range(1, 7))

    fact = mem.record_fact(
        subject="client:42",
        predicate="risk_tier",
        value="medium",
        valid_from=day1,
        recorded_from=day3,
        source="crm",
    )

    mem.correct_fact(
        old_fact_id=fact.fact_id,
        value="high",
        valid_from=day1,
        recorded_from=day5,
        source="manual_review",
    )

    # After correction, current valid-time view returns corrected value
    valid = mem.retrieve_valid_at(at=day2, subject="client:42")
    assert len(valid) == 1 and valid[0].value == "high", (
        f"Expected ['high'], got {[f.value for f in valid]}"
    )

    # Before ingestion, system knew nothing
    known = mem.retrieve_known_at(at=day2, subject="client:42")
    assert len(known) == 0, f"Expected [], got {[f.value for f in known]}"

    # Belief at (valid=day2, record=day4) = "medium" (before correction)
    belief_before = mem.retrieve_belief_state(at_valid=day2, at_record=day4)
    assert len(belief_before) == 1 and belief_before[0].value == "medium", (
        f"Expected ['medium'], got {[f.value for f in belief_before]}"
    )

    # Belief at (valid=day2, record=day6) = "high" (after correction)
    belief_after = mem.retrieve_belief_state(at_valid=day2, at_record=day6)
    assert len(belief_after) == 1 and belief_after[0].value == "high", (
        f"Expected ['high'], got {[f.value for f in belief_after]}"
    )

    print("\n[OK] Bi-temporal memory example completed.")


if __name__ == "__main__":
    try:
        main()
        if "--test" in sys.argv:
            run_smoke_test()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
