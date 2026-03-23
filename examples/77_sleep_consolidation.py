"""
Example 77 — Sleep Consolidation
==================================

Demonstrates the SleepConsolidation cycle: prune stale context, replay
successful patterns into episodic memory, compress recurring patterns into
templates, run counterfactual replay over corrected bi-temporal facts,
and promote histone marks from temporary to permanent.

Usage:
    python examples/77_sleep_consolidation.py
"""

from datetime import datetime, timedelta

from operon_ai import (
    BiTemporalMemory,
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    TaskFingerprint,
    SleepConsolidation,
)
from operon_ai.healing.autophagy_daemon import AutophagyDaemon
from operon_ai.memory.episodic import EpisodicMemory
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength

# ---------------------------------------------------------------------------
# 1. Build operational history
# ---------------------------------------------------------------------------

lib = PatternLibrary()

tmpl = PatternTemplate(
    template_id="enterprise",
    name="Enterprise Analysis",
    topology="skill_organism",
    stage_specs=(
        {"name": "research", "role": "Researcher"},
        {"name": "strategy", "role": "Strategist"},
    ),
    intervention_policy={},
    fingerprint=TaskFingerprint("sequential", 3, 3, ("researcher", "strategist")),
)
lib.register_template(tmpl)

# Simulate 5 successful runs and 1 failed
for i in range(5):
    lib.record_run(PatternRunRecord(
        record_id=lib.make_id(), template_id="enterprise",
        fingerprint=tmpl.fingerprint, success=True,
        latency_ms=1200.0 + i * 100, tokens_used=3000 + i * 200,
    ))
lib.record_run(PatternRunRecord(
    record_id=lib.make_id(), template_id="enterprise",
    fingerprint=tmpl.fingerprint, success=False,
    latency_ms=2500.0, tokens_used=5000,
))

print(f"Library: {lib.summary()['record_count']} records, "
      f"success rate: {lib.success_rate('enterprise'):.0%}")

# ---------------------------------------------------------------------------
# 2. Build memory systems
# ---------------------------------------------------------------------------

episodic = EpisodicMemory()
histone = HistoneStore()

# Add some acetylation marks that have been accessed frequently
h1 = histone.add_marker(
    "Enterprise templates work well for sequential tasks",
    MarkerType.ACETYLATION, MarkerStrength.MODERATE,
)
# Simulate high access count
histone._markers[h1].access_count = 10

h2 = histone.add_marker(
    "Research stage benefits from fuzzy mode",
    MarkerType.ACETYLATION, MarkerStrength.MODERATE,
)
histone._markers[h2].access_count = 2  # Below threshold

# ---------------------------------------------------------------------------
# 3. Build bi-temporal memory with a correction
# ---------------------------------------------------------------------------

mem = BiTemporalMemory()
day1 = datetime(2026, 3, 1)
day3 = day1 + timedelta(days=2)
day5 = day1 + timedelta(days=4)

fact = mem.record_fact(
    "researcher", "methodology", "quantitative",
    valid_from=day1, recorded_from=day1, source="initial_assessment",
)
mem.correct_fact(
    fact.fact_id, "qualitative",
    valid_from=day1, recorded_from=day5, source="review",
)

# ---------------------------------------------------------------------------
# 4. Run sleep consolidation
# ---------------------------------------------------------------------------

daemon = AutophagyDaemon(
    histone_store=histone,
    lysosome=None,
    summarizer=lambda text: "Consolidated summary of operational history",
)

consolidation = SleepConsolidation(
    daemon=daemon,
    pattern_library=lib,
    episodic_memory=episodic,
    histone_store=histone,
    bitemporal_memory=mem,
    min_access_count_for_promotion=2,
    acetylation_promotion_threshold=5,
)

result = consolidation.consolidate()

print(f"\n=== Consolidation Result ===")
print(f"  Templates created: {result.templates_created}")
print(f"  Memories promoted: {result.memories_promoted}")
print(f"  Histone promotions: {result.histone_promotions}")
print(f"  Counterfactual results: {len(result.counterfactual_results)}")
print(f"  Duration: {result.duration_ms:.1f}ms")

if result.counterfactual_results:
    print(f"\n=== Counterfactual Analysis ===")
    for cr in result.counterfactual_results:
        print(f"  Corrections: {len(cr.corrections_found)}")
        print(f"  Affected stages: {cr.affected_stages}")
        print(f"  Outcome would change: {cr.outcome_would_change}")
        print(f"  Reasoning: {cr.reasoning}")

# ---------------------------------------------------------------------------
# 5. Check results
# ---------------------------------------------------------------------------

print(f"\n=== Post-Consolidation ===")
print(f"  Episodic memories: {len(episodic.retrieve('run', limit=100))}")
consolidated = lib.get_template("enterprise_c")
print(f"  Consolidated template: {'yes' if consolidated else 'no'}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert result.templates_created >= 1
assert result.memories_promoted >= 1
assert result.histone_promotions >= 1  # h1 had access_count=10
assert consolidated is not None
print("\n--- all assertions passed ---")
