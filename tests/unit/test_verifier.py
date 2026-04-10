"""Tests for VerifierComponent — rubric-based quality evaluation."""

from operon_ai.patterns.verifier import VerifierComponent, VerifierConfig
from operon_ai.patterns.watcher import WatcherSignal, SignalCategory


def _mock_stage(name="test_stage"):
    class S:
        pass
    s = S()
    s.name = name
    return s


def _mock_result(output="test output"):
    class R:
        pass
    r = R()
    r.output = output
    r.model_alias = "fast"
    r.action_type = "EXECUTE"
    return r


def test_verifier_emits_signal_on_low_quality():
    """Verifier emits a signal when quality is below threshold."""
    def rubric(output, stage_name):
        return 0.3  # Below default 0.5 threshold

    verifier = VerifierComponent(rubric=rubric)
    verifier.on_run_start("task", {})

    shared = {}
    verifier.on_stage_result(_mock_stage(), _mock_result(), shared, {})

    assert len(verifier.signals) == 1
    sig = verifier.signals[0]
    assert sig.category == SignalCategory.EPISTEMIC
    assert sig.source == "verifier"
    assert sig.detail["quality"] == 0.3
    assert sig.detail["below_threshold"] is True
    assert sig.value == 0.7  # severity = 1 - quality

    # Signal deposited in shared_state for WatcherComponent
    assert "_verifier_signals" in shared
    assert len(shared["_verifier_signals"]) == 1


def test_verifier_emits_signal_on_high_quality():
    """Verifier still emits signal when quality is high (but below_threshold=False)."""
    def rubric(output, stage_name):
        return 0.9

    verifier = VerifierComponent(rubric=rubric)
    verifier.on_run_start("task", {})

    shared = {}
    verifier.on_stage_result(_mock_stage(), _mock_result(), shared, {})

    assert len(verifier.signals) == 1
    assert verifier.signals[0].detail["below_threshold"] is False
    assert verifier.signals[0].value == pytest.approx(0.1)


def test_verifier_no_rubric_no_signal():
    """Without a rubric, no signals are emitted."""
    verifier = VerifierComponent(rubric=None)
    verifier.on_run_start("task", {})

    shared = {}
    verifier.on_stage_result(_mock_stage(), _mock_result(), shared, {})

    assert len(verifier.signals) == 0
    assert "_verifier_signals" not in shared


def test_verifier_tracks_quality_scores():
    """Quality scores are accumulated across stages."""
    def rubric(output, stage_name):
        return 0.5 if stage_name == "a" else 0.9

    verifier = VerifierComponent(rubric=rubric)
    verifier.on_run_start("task", {})

    verifier.on_stage_result(_mock_stage("a"), _mock_result(), {}, {})
    verifier.on_stage_result(_mock_stage("b"), _mock_result(), {}, {})

    assert len(verifier.quality_scores) == 2
    assert verifier.quality_scores[0] == ("a", 0.5)
    assert verifier.quality_scores[1] == ("b", 0.9)
    assert verifier.mean_quality() == pytest.approx(0.7)


def test_verifier_clamps_quality():
    """Quality scores are clamped to [0, 1]."""
    def rubric(output, stage_name):
        return 1.5

    verifier = VerifierComponent(rubric=rubric)
    verifier.on_run_start("task", {})
    verifier.on_stage_result(_mock_stage(), _mock_result(), {}, {})

    assert verifier.quality_scores[0][1] == 1.0


def test_verifier_handles_rubric_exception():
    """Rubric exceptions are caught silently (no signal emitted)."""
    def rubric(output, stage_name):
        raise ValueError("broken rubric")

    verifier = VerifierComponent(rubric=rubric)
    verifier.on_run_start("task", {})
    verifier.on_stage_result(_mock_stage(), _mock_result(), {}, {})

    assert len(verifier.signals) == 0


def test_verifier_custom_threshold():
    """Custom threshold changes what counts as low quality."""
    def rubric(output, stage_name):
        return 0.6  # Above default 0.5, below custom 0.7

    config = VerifierConfig(quality_low_threshold=0.7)
    verifier = VerifierComponent(rubric=rubric, config=config)
    verifier.on_run_start("task", {})

    shared = {}
    verifier.on_stage_result(_mock_stage(), _mock_result(), shared, {})

    assert verifier.signals[0].detail["below_threshold"] is True


def test_verifier_resets_on_run_start():
    """on_run_start clears accumulated state."""
    def rubric(output, stage_name):
        return 0.5

    verifier = VerifierComponent(rubric=rubric)
    verifier.on_run_start("task1", {})
    verifier.on_stage_result(_mock_stage(), _mock_result(), {}, {})
    assert len(verifier.signals) == 1

    verifier.on_run_start("task2", {})
    assert len(verifier.signals) == 0
    assert len(verifier.quality_scores) == 0


import pytest
