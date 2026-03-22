"""
Example 71: Bi-Temporal Substrate in a SkillOrganism
====================================================

Demonstrates the Phase 2 integration: a multi-stage organism backed by a
bi-temporal memory substrate.  Four handler stages simulate an enterprise
account-review workflow:

1. **Research** — records facts about the account into the substrate
2. **Strategist** — reads the substrate to produce a recommendation
3. **Evaluator** — records rubric-based concerns as new facts
4. **Adversary** — corrects a research assumption (the risk level)

After the run the example shows:

- Current recommendation
- What the strategist knew when it decided (belief-state reconstruction)
- What changed after evaluator and adversary critique (diff on record axis)
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from operon_ai import BiTemporalMemory, SkillStage, skill_organism


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 3, 22, 9, 0, 0)


def _t(minutes: int) -> datetime:
    """Return a deterministic time offset from _BASE."""
    return _BASE + timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def research_handler(task):
    """Simulate research by returning structured account data."""
    return {
        "account": "acct:42",
        "revenue": 2_500_000,
        "risk_level": "low",
        "segment": "enterprise",
    }


def research_extractor(task, shared_state, stage_outputs, stage, result):
    """Record research findings as individual facts in the substrate."""
    data = result.output
    now = _t(1)
    return [
        {
            "subject": data["account"],
            "predicate": "revenue",
            "value": data["revenue"],
            "valid_from": now,
            "recorded_from": now,
        },
        {
            "subject": data["account"],
            "predicate": "risk_level",
            "value": data["risk_level"],
            "valid_from": now,
            "recorded_from": now,
        },
        {
            "subject": data["account"],
            "predicate": "segment",
            "value": data["segment"],
            "valid_from": now,
            "recorded_from": now,
        },
    ]


def strategist_handler(task, shared_state, stage_outputs, stage, view):
    """Produce a recommendation based on substrate facts."""
    if view is None:
        return {"recommendation": "insufficient data"}
    facts_by_pred = {f.predicate: f.value for f in view.facts}
    risk = facts_by_pred.get("risk_level", "unknown")
    revenue = facts_by_pred.get("revenue", 0)
    if risk == "low" and revenue > 1_000_000:
        rec = "approve premium partnership"
    elif risk == "high":
        rec = "escalate to risk committee"
    else:
        rec = "standard onboarding"
    return {"recommendation": rec, "based_on": dict(facts_by_pred)}


def evaluator_handler(task, shared_state, stage_outputs):
    """Record evaluator concerns about the recommendation."""
    return {"concerns": ["revenue figure not verified", "single-source risk assessment"]}


def evaluator_extractor(task, shared_state, stage_outputs, stage, result):
    """Record each concern as a fact."""
    return [
        {
            "subject": "acct:42",
            "predicate": f"concern:{i}",
            "value": concern,
            "valid_from": _t(3),
            "recorded_from": _t(3),
        }
        for i, concern in enumerate(result.output["concerns"])
    ]


def adversary_handler(task, shared_state, stage_outputs):
    """Challenge the risk assessment — adversary believes risk is high."""
    return {"challenge": "risk_level should be high based on sector volatility"}


def adversary_extractor(task, shared_state, stage_outputs, stage, result):
    """Correct the original risk_level fact."""
    # Find the original risk_level fact to correct
    substrate = shared_state.get("_substrate_ref")
    if substrate is None:
        return None
    risk_facts = substrate.retrieve_valid_at(
        at=datetime.now(), subject="acct:42", predicate="risk_level",
    )
    if not risk_facts:
        return None
    return {
        "op": "correct",
        "old_fact_id": risk_facts[0].fact_id,
        "value": "high",
        "valid_from": _t(1),
        "recorded_from": _t(4),
        "tags": ("adversary", "correction"),
    }


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def build_organism():
    mem = BiTemporalMemory()

    organism = skill_organism(
        stages=[
            SkillStage(
                name="research",
                role="Researcher",
                handler=research_handler,
                fact_extractor=research_extractor,
            ),
            SkillStage(
                name="strategist",
                role="Strategist",
                handler=strategist_handler,
                read_query="acct:42",
                emit_output_fact=True,
                fact_tags=("strategy",),
            ),
            SkillStage(
                name="evaluator",
                role="Evaluator",
                handler=evaluator_handler,
                fact_extractor=evaluator_extractor,
            ),
            SkillStage(
                name="adversary",
                role="Adversary",
                handler=adversary_handler,
                fact_extractor=adversary_extractor,
            ),
        ],
        substrate=mem,
    )
    return organism, mem


def main() -> None:
    organism, mem = build_organism()

    # Stash substrate ref so adversary_extractor can find it
    result = organism.run(
        "Review account acct:42 for partnership eligibility",
        shared_state={"_substrate_ref": mem},
    )

    print("=" * 72)
    print("Bi-Temporal Substrate — SkillOrganism Integration")
    print("=" * 72)

    # -- Stage outputs --
    for sr in result.stage_results:
        print(f"\n[{sr.stage_name}] → {sr.output}")

    # -- Current substrate state --
    print("\n--- Current facts (active) ---")
    now = datetime.now()
    for f in mem.retrieve_valid_at(at=now):
        print(f"  {f.subject}.{f.predicate} = {f.value}  (source: {f.source})")

    # -- What strategist knew --
    print("\n--- What strategist knew when it decided ---")
    # Strategist ran at ~_t(2); record-time horizon = its read moment
    belief = mem.retrieve_belief_state(at_valid=_t(2), at_record=_t(2))
    for f in belief:
        print(f"  {f.subject}.{f.predicate} = {f.value}")

    # -- What changed after critique --
    print("\n--- Changes after critique (record-axis diff) ---")
    diff = mem.diff_between(_t(2), now, axis="record")
    for f in diff:
        tag = f" [corrects {f.supersedes}]" if f.supersedes else ""
        print(f"  {f.subject}.{f.predicate} = {f.value}{tag}")

    # -- Full audit trail for risk_level --
    print("\n--- Audit trail: acct:42.risk_level ---")
    for f in mem.history("acct:42", "risk_level"):
        status = "CLOSED" if f.recorded_to else "ACTIVE"
        print(f"  [{status}] value={f.value}  source={f.source}  "
              f"recorded={f.recorded_from.isoformat()}")


if __name__ == "__main__":
    try:
        main()
        if "--test" in sys.argv:
            organism, mem = build_organism()
            result = organism.run(
                "Review account acct:42",
                shared_state={"_substrate_ref": mem},
            )
            # Strategist saw low risk → approved premium partnership
            assert result.stage_results[1].output["recommendation"] == "approve premium partnership"
            # Adversary corrected risk to high
            risk_history = mem.history("acct:42", "risk_level")
            assert len(risk_history) == 2
            assert risk_history[0].value == "low"
            assert risk_history[0].recorded_to is not None
            assert risk_history[1].value == "high"
            assert risk_history[1].supersedes == risk_history[0].fact_id
            # Substrate has evaluator concerns
            concerns = [f for f in mem.retrieve_valid_at(at=datetime.now())
                        if f.predicate.startswith("concern:")]
            assert len(concerns) == 2
            print("\n[OK] Bi-temporal skill organism example completed.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
