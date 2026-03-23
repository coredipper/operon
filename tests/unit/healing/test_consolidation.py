"""Tests for SleepConsolidation and counterfactual_replay."""

from datetime import datetime, timedelta

from operon_ai import (
    BiTemporalMemory,
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    TaskFingerprint,
)
from operon_ai.healing.consolidation import (
    SleepConsolidation,
    ConsolidationResult,
    CounterfactualResult,
    counterfactual_replay,
)
from operon_ai.healing.autophagy_daemon import AutophagyDaemon
from operon_ai.memory.episodic import EpisodicMemory, MemoryTier
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp(**kw):
    defaults = dict(task_shape="sequential", tool_count=2, subtask_count=2, required_roles=("worker",))
    defaults.update(kw)
    return TaskFingerprint(**defaults)


def _tmpl(tid="t1"):
    return PatternTemplate(
        template_id=tid, name="test", topology="skill_organism",
        stage_specs=({"name": "s1", "role": "W"},),
        intervention_policy={}, fingerprint=_fp(),
    )


def _record(tid="t1", success=True, **kw):
    defaults = dict(
        record_id=PatternLibrary.make_id(), template_id=tid,
        fingerprint=_fp(), success=success, latency_ms=100.0, tokens_used=500,
    )
    defaults.update(kw)
    return PatternRunRecord(**defaults)


def _build_consolidation(
    records=None, templates=None, bitemporal=None,
    histone_access_count=0,
):
    lib = PatternLibrary()
    for t in (templates or [_tmpl()]):
        lib.register_template(t)
    for r in (records or []):
        lib.record_run(r)

    episodic = EpisodicMemory()
    histone = HistoneStore()

    # Optionally add an acetylation mark with high access count
    if histone_access_count > 0:
        h = histone.add_marker("important lesson", MarkerType.ACETYLATION, MarkerStrength.MODERATE)
        marker = histone._markers[h]
        marker.access_count = histone_access_count

    daemon = AutophagyDaemon(
        histone_store=histone,
        lysosome=None,
        summarizer=lambda text: "Summary",
    )

    return SleepConsolidation(
        daemon=daemon,
        pattern_library=lib,
        episodic_memory=episodic,
        histone_store=histone,
        bitemporal_memory=bitemporal,
        min_access_count_for_promotion=2,
        acetylation_promotion_threshold=3,
    )


# ---------------------------------------------------------------------------
# SleepConsolidation
# ---------------------------------------------------------------------------


def test_consolidation_returns_result():
    sc = _build_consolidation(records=[_record(success=True)])
    result = sc.consolidate()
    assert isinstance(result, ConsolidationResult)
    assert result.duration_ms >= 0


def test_consolidation_replays_successful_patterns():
    sc = _build_consolidation(records=[
        _record(success=True),
        _record(success=True),
        _record(success=False),
    ])
    result = sc.consolidate()
    # 2 successful records stored in episodic memory
    entries = sc.episodic_memory.retrieve("Successful run", limit=10)
    assert len(entries) >= 2


def test_consolidation_promotes_working_to_episodic():
    # 3 successful records for same template → should promote
    sc = _build_consolidation(records=[
        _record(tid="t1", success=True),
        _record(tid="t1", success=True),
        _record(tid="t1", success=True),
    ])
    result = sc.consolidate()
    assert result.memories_promoted >= 1


def test_consolidation_compresses_to_templates():
    sc = _build_consolidation(
        templates=[_tmpl(tid="t1")],
        records=[
            _record(tid="t1", success=True),
            _record(tid="t1", success=True),
            _record(tid="t1", success=True),
        ],
    )
    result = sc.consolidate()
    assert result.templates_created >= 1
    # A consolidated template should exist
    consolidated = sc.pattern_library.get_template("t1_c")
    assert consolidated is not None
    assert "consolidated" in consolidated.tags


def test_consolidation_promotes_histone_marks():
    sc = _build_consolidation(histone_access_count=5)
    result = sc.consolidate()
    assert result.histone_promotions >= 1


def test_consolidation_backward_compat_no_bitemporal():
    sc = _build_consolidation(records=[_record(success=True)])
    result = sc.consolidate()
    assert result.counterfactual_results == ()


# ---------------------------------------------------------------------------
# counterfactual_replay
# ---------------------------------------------------------------------------


def test_counterfactual_replay_detects_correction():
    mem = BiTemporalMemory()
    day1 = datetime(2026, 1, 1)
    day3 = day1 + timedelta(days=2)
    day5 = day1 + timedelta(days=4)

    fact = mem.record_fact("worker", "status", "active",
                           valid_from=day1, recorded_from=day1, source="sys")
    mem.correct_fact(fact.fact_id, "inactive",
                     valid_from=day1, recorded_from=day5, source="review")

    record = _record(recorded_at=day3)
    cr = counterfactual_replay(record, mem, run_time=day3, now=day5 + timedelta(days=1))
    assert isinstance(cr, CounterfactualResult)
    assert len(cr.corrections_found) >= 1


def test_counterfactual_replay_no_corrections():
    mem = BiTemporalMemory()
    day1 = datetime(2026, 1, 1)
    mem.record_fact("x", "y", "z", valid_from=day1, recorded_from=day1, source="sys")

    record = _record(recorded_at=day1 + timedelta(days=1))
    cr = counterfactual_replay(record, mem, run_time=day1, now=day1 + timedelta(days=2))
    assert cr.outcome_would_change is False
    assert cr.corrections_found == ()


def test_consolidation_result_structure():
    sc = _build_consolidation(records=[_record(success=True)])
    result = sc.consolidate()
    assert hasattr(result, "templates_created")
    assert hasattr(result, "memories_promoted")
    assert hasattr(result, "histone_promotions")
    assert hasattr(result, "counterfactual_results")
    assert hasattr(result, "prune_result")
    assert hasattr(result, "duration_ms")
