"""Tests for CognitiveMode annotations and watcher integration."""

from typing import Any

from operon_ai import CognitiveMode, SkillStage, WatcherComponent
from operon_ai.patterns.types import resolve_cognitive_mode
from operon_ai.patterns.watcher import WATCHER_STATE_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, model_alias="fast", action_type="EXECUTE"):
        self.output = "ok"
        self.model_alias = model_alias
        self.action_type = action_type
        self.metadata = {}


# ---------------------------------------------------------------------------
# CognitiveMode resolution
# ---------------------------------------------------------------------------


def test_cognitive_mode_default_is_none():
    stage = SkillStage(name="s", role="R")
    assert stage.cognitive_mode is None


def test_resolve_infers_observational_from_fast():
    stage = SkillStage(name="s", role="R", mode="fast")
    assert resolve_cognitive_mode(stage) == CognitiveMode.OBSERVATIONAL


def test_resolve_infers_observational_from_fixed():
    stage = SkillStage(name="s", role="R", mode="fixed")
    assert resolve_cognitive_mode(stage) == CognitiveMode.OBSERVATIONAL


def test_resolve_infers_action_oriented_from_deep():
    stage = SkillStage(name="s", role="R", mode="deep")
    assert resolve_cognitive_mode(stage) == CognitiveMode.ACTION_ORIENTED


def test_resolve_infers_action_oriented_from_fuzzy():
    stage = SkillStage(name="s", role="R", mode="fuzzy")
    assert resolve_cognitive_mode(stage) == CognitiveMode.ACTION_ORIENTED


def test_explicit_override_wins():
    stage = SkillStage(name="s", role="R", mode="deep", cognitive_mode=CognitiveMode.OBSERVATIONAL)
    assert resolve_cognitive_mode(stage) == CognitiveMode.OBSERVATIONAL


# ---------------------------------------------------------------------------
# Watcher cognitive mode signals
# ---------------------------------------------------------------------------


def test_watcher_detects_mode_mismatch_observational_on_deep():
    w = WatcherComponent()
    stage = SkillStage(name="s1", role="R", mode="fast", cognitive_mode=CognitiveMode.OBSERVATIONAL)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(stage, _FakeResult(model_alias="deep"), state, {})
    # Should have a cognitive_mode signal with mismatch=True
    cm_signals = [s for s in w.signals if s.source == "cognitive_mode"]
    assert len(cm_signals) >= 1
    assert cm_signals[0].detail["mismatch"] is True


def test_watcher_no_mismatch_on_aligned_mode():
    w = WatcherComponent()
    stage = SkillStage(name="s1", role="R", mode="fast", cognitive_mode=CognitiveMode.OBSERVATIONAL)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(stage, _FakeResult(model_alias="fast"), state, {})
    cm_signals = [s for s in w.signals if s.source == "cognitive_mode"]
    assert len(cm_signals) >= 1
    assert cm_signals[0].detail["mismatch"] is False


def test_watcher_mode_balance():
    w = WatcherComponent()
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(
        SkillStage(name="s1", role="R", mode="fast", cognitive_mode=CognitiveMode.OBSERVATIONAL),
        _FakeResult(model_alias="fast"), state, {},
    )
    w.on_stage_result(
        SkillStage(name="s2", role="R", mode="deep", cognitive_mode=CognitiveMode.ACTION_ORIENTED),
        _FakeResult(model_alias="deep"), state, {},
    )
    balance = w.mode_balance()
    assert balance["observational"] == 1
    assert balance["action_oriented"] == 1
    assert balance["balance_ratio"] == 0.5
    assert balance["mismatches"] == 0
