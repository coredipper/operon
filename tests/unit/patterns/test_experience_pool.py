"""Tests for WatcherComponent experience pool."""

from datetime import datetime, timedelta
from typing import Any

from operon_ai import (
    InterventionKind,
    SkillStage,
    TaskFingerprint,
    WatcherComponent,
    WatcherConfig,
    WatcherIntervention,
)
from operon_ai.patterns.watcher import ExperienceRecord, WATCHER_STATE_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp(**kw):
    defaults = dict(task_shape="sequential", tool_count=2, subtask_count=2, required_roles=("worker",))
    defaults.update(kw)
    return TaskFingerprint(**defaults)


def _stage(name="s1"):
    return SkillStage(name=name, role="Worker")


class _FakeResult:
    def __init__(self, output="ok", model_alias="fast", action_type="EXECUTE"):
        self.output = output
        self.model_alias = model_alias
        self.action_type = action_type
        self.metadata = {}


# ---------------------------------------------------------------------------
# record_experience
# ---------------------------------------------------------------------------


def test_record_experience_adds_to_pool():
    w = WatcherComponent()
    w.record_experience(
        stage_name="s1",
        signal_category="epistemic",
        intervention_kind="escalate",
        intervention_reason="stagnant",
    )
    assert len(w.experience_pool) == 1


def test_record_experience_preserves_fields():
    w = WatcherComponent()
    fp = _fp()
    rec = w.record_experience(
        fingerprint=fp,
        stage_name="router",
        signal_category="epistemic",
        signal_detail={"status": "stagnant"},
        intervention_kind="escalate",
        intervention_reason="test reason",
        outcome_success=True,
    )
    assert rec.fingerprint is fp
    assert rec.stage_name == "router"
    assert rec.signal_category == "epistemic"
    assert rec.intervention_kind == "escalate"
    assert rec.outcome_success is True


# ---------------------------------------------------------------------------
# retrieve_similar_experiences
# ---------------------------------------------------------------------------


def test_retrieve_filters_by_stage_name():
    w = WatcherComponent()
    w.record_experience(stage_name="s1", signal_category="epistemic",
                        intervention_kind="retry", intervention_reason="")
    w.record_experience(stage_name="s2", signal_category="epistemic",
                        intervention_kind="retry", intervention_reason="")
    results = w.retrieve_similar_experiences(stage_name="s1")
    assert len(results) == 1
    assert results[0].stage_name == "s1"


def test_retrieve_filters_by_signal_category():
    w = WatcherComponent()
    w.record_experience(stage_name="s1", signal_category="epistemic",
                        intervention_kind="retry", intervention_reason="")
    w.record_experience(stage_name="s1", signal_category="somatic",
                        intervention_kind="halt", intervention_reason="")
    results = w.retrieve_similar_experiences(signal_category="somatic")
    assert len(results) == 1
    assert results[0].signal_category == "somatic"


def test_retrieve_filters_by_fingerprint_shape():
    w = WatcherComponent()
    w.record_experience(fingerprint=_fp(task_shape="sequential"), stage_name="s1",
                        signal_category="epistemic", intervention_kind="retry", intervention_reason="")
    w.record_experience(fingerprint=_fp(task_shape="parallel"), stage_name="s1",
                        signal_category="epistemic", intervention_kind="escalate", intervention_reason="")
    results = w.retrieve_similar_experiences(fingerprint=_fp(task_shape="sequential"))
    assert len(results) == 1
    assert results[0].fingerprint.task_shape == "sequential"


def test_retrieve_respects_limit():
    w = WatcherComponent()
    for i in range(10):
        w.record_experience(stage_name="s1", signal_category="epistemic",
                            intervention_kind="retry", intervention_reason=str(i))
    results = w.retrieve_similar_experiences(limit=3)
    assert len(results) == 3


def test_retrieve_returns_newest_first():
    w = WatcherComponent()
    old = ExperienceRecord(
        fingerprint=None, stage_name="s1", signal_category="epistemic",
        intervention_kind="retry", intervention_reason="old",
        recorded_at=datetime(2026, 1, 1),
    )
    new = ExperienceRecord(
        fingerprint=None, stage_name="s1", signal_category="epistemic",
        intervention_kind="escalate", intervention_reason="new",
        recorded_at=datetime(2026, 3, 1),
    )
    w.experience_pool.extend([old, new])
    results = w.retrieve_similar_experiences()
    assert results[0].intervention_reason == "new"


def test_retrieve_empty_pool_returns_empty():
    w = WatcherComponent()
    assert w.retrieve_similar_experiences() == []


# ---------------------------------------------------------------------------
# recommend_intervention
# ---------------------------------------------------------------------------


def test_recommend_returns_most_successful_kind():
    w = WatcherComponent()
    # 2 successful escalates, 1 successful retry
    w.record_experience(stage_name="s1", signal_category="epistemic",
                        intervention_kind="escalate", intervention_reason="",
                        outcome_success=True)
    w.record_experience(stage_name="s1", signal_category="epistemic",
                        intervention_kind="escalate", intervention_reason="",
                        outcome_success=True)
    w.record_experience(stage_name="s1", signal_category="epistemic",
                        intervention_kind="retry", intervention_reason="",
                        outcome_success=True)
    result = w.recommend_intervention(stage_name="s1", signal_category="epistemic")
    assert result == InterventionKind.ESCALATE


def test_recommend_returns_none_on_empty_pool():
    w = WatcherComponent()
    assert w.recommend_intervention(stage_name="s1", signal_category="epistemic") is None


def test_recommend_returns_none_when_no_successes():
    w = WatcherComponent()
    w.record_experience(stage_name="s1", signal_category="epistemic",
                        intervention_kind="retry", intervention_reason="",
                        outcome_success=False)
    assert w.recommend_intervention(stage_name="s1", signal_category="epistemic") is None


# ---------------------------------------------------------------------------
# Integration with _decide_intervention
# ---------------------------------------------------------------------------


def test_experience_pool_consulted_when_rules_return_none():
    """Experience pool recommends escalate when signals are non-trivial but rules don't trigger."""
    from operon_ai.patterns.watcher import WatcherSignal, SignalCategory

    w = WatcherComponent()
    w.set_fingerprint(_fp())
    # Pre-populate: escalate was successful for epistemic signals on s1
    w.record_experience(
        fingerprint=_fp(), stage_name="s1", signal_category="epistemic",
        intervention_kind="escalate", intervention_reason="past",
        outcome_success=True,
    )
    w.on_run_start("task", {})
    # Inject a non-trivial signal that doesn't trigger rules (healthy status, value > 0.3)
    signals = [WatcherSignal(
        category=SignalCategory.EPISTEMIC, source="epiplexity",
        stage_name="s1", value=0.5,
        detail={"status": "healthy", "integral": 0.5},
    )]
    result = w._decide_intervention(_stage("s1"), _FakeResult(), signals)
    assert result is not None
    assert result.kind == InterventionKind.ESCALATE
    assert "experience-based" in result.reason


def test_rule_based_decision_takes_priority_over_experience():
    """Rule-based RETRY on FAILURE should fire even if experience suggests ESCALATE."""
    w = WatcherComponent()
    w.set_fingerprint(_fp())
    w.record_experience(
        fingerprint=_fp(), stage_name="s1", signal_category="epistemic",
        intervention_kind="escalate", intervention_reason="past",
        outcome_success=True,
    )
    w.on_run_start("task", {})
    result = w._decide_intervention(_stage("s1"), _FakeResult(action_type="FAILURE"), [])
    assert result is not None
    assert result.kind == InterventionKind.RETRY


def test_backward_compat_empty_pool_changes_nothing():
    w = WatcherComponent()
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    assert WATCHER_STATE_KEY not in state
    assert w.summary()["total_interventions"] == 0
