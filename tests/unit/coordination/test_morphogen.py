"""Tests for morphogen gradients and orchestrator state updates."""

import pytest

from operon_ai.coordination.morphogen import (
    GradientOrchestrator,
    MorphogenGradient,
    MorphogenType,
)


def test_budget_hint_critical_precedes_low_warning():
    """A critically low budget should emit the critical hint, not the generic one."""
    gradient = MorphogenGradient()
    gradient.set(MorphogenType.BUDGET, 0.05)

    hints = gradient.get_strategy_hints()

    assert "CRITICAL: Token budget nearly exhausted. Summarize and conclude." in hints
    assert "Token budget low - be concise and efficient." not in hints


def test_report_step_result_records_actual_deltas():
    """Update history should capture the applied delta for each morphogen."""
    orchestrator = GradientOrchestrator(silent=True)

    orchestrator.report_step_result(
        success=True,
        tokens_used=200,
        total_budget=1000,
        complexity_estimate=0.8,
    )

    deltas = {update.morphogen_type: update.delta for update in orchestrator.update_history}

    assert deltas[MorphogenType.CONFIDENCE] == pytest.approx(0.1)
    assert deltas[MorphogenType.BUDGET] == pytest.approx(-0.2)
    assert deltas[MorphogenType.COMPLEXITY] == pytest.approx(0.3)


def test_report_step_result_uses_clamped_confidence_delta():
    """Confidence deltas should reflect the clamped value that was actually applied."""
    orchestrator = GradientOrchestrator(silent=True)
    orchestrator.gradient.set(MorphogenType.CONFIDENCE, 0.95)

    orchestrator.report_step_result(success=True)

    confidence_update = next(
        update
        for update in orchestrator.update_history
        if update.morphogen_type == MorphogenType.CONFIDENCE
    )
    assert confidence_update.delta == pytest.approx(0.05)


def test_manual_updates_record_delta_not_absolute_value():
    """Manual urgency/risk changes should record the change from the prior value."""
    orchestrator = GradientOrchestrator(silent=True)

    orchestrator.set_urgency(0.7)
    orchestrator.set_risk(0.1)

    urgency_update, risk_update = orchestrator.update_history
    assert urgency_update.delta == pytest.approx(0.4)
    assert risk_update.delta == pytest.approx(-0.2)
