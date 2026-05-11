"""Unit tests for the Helpfulness/Harmfulness metric.

Anchors against the ratios reported in Ta et al., *Reinforced Agent*
(arXiv:2604.27233): o3-mini at 3:1 benefit-to-risk, GPT-4o at 2.1:1.
"""

from __future__ import annotations

import pytest

from eval.patterns.reviewer_gate_hh_metric import (
    HelpfulnessHarmfulnessMetric,
    TrajectoryOutcome,
    compute_hh,
    is_net_positive,
)


def _base(task_id: str, correct: bool) -> TrajectoryOutcome:
    return TrajectoryOutcome(task_id=task_id, correct=correct, intervened=False)


def _reviewed(task_id: str, correct: bool, intervened: bool = True) -> TrajectoryOutcome:
    return TrajectoryOutcome(task_id=task_id, correct=correct, intervened=intervened)


class TestAllImprovements:
    """Every base error is corrected; no correct response is degraded."""

    def test_all_corrected_yields_h_one_harm_zero(self) -> None:
        base = [_base("t1", False), _base("t2", False), _base("t3", True)]
        reviewed = [_reviewed("t1", True), _reviewed("t2", True), _reviewed("t3", True, intervened=False)]
        m = compute_hh(base, reviewed)
        assert m.helpfulness == 1.0
        assert m.harmfulness == 0.0
        assert m.benefit_risk_ratio is None  # zero denominator
        assert m.n_corrected == 2
        assert m.n_degraded == 0
        assert is_net_positive(m) is True


class TestAllRegressions:
    """Every base correct response is degraded; no base error is fixed."""

    def test_all_degraded_yields_h_zero_harm_one(self) -> None:
        base = [_base("t1", True), _base("t2", True)]
        reviewed = [_reviewed("t1", False), _reviewed("t2", False)]
        m = compute_hh(base, reviewed)
        assert m.helpfulness == 0.0
        assert m.harmfulness == 1.0
        assert m.benefit_risk_ratio == 0.0
        assert m.n_corrected == 0
        assert m.n_degraded == 2
        assert is_net_positive(m) is False


class TestMixedRatios:
    """Mixed populations producing the paper's reported ratios."""

    def test_three_to_one_ratio_matches_o3_mini(self) -> None:
        """3:1 benefit-to-risk: 3 corrections out of 4 errors, 1 degradation out of 4 correct.

        helpfulness = 3/4 = 0.75
        harmfulness = 1/4 = 0.25
        ratio = 0.75 / 0.25 = 3.0
        """
        base = [
            _base("e1", False), _base("e2", False), _base("e3", False), _base("e4", False),
            _base("c1", True), _base("c2", True), _base("c3", True), _base("c4", True),
        ]
        reviewed = [
            _reviewed("e1", True), _reviewed("e2", True), _reviewed("e3", True), _reviewed("e4", False),
            _reviewed("c1", True), _reviewed("c2", True), _reviewed("c3", True), _reviewed("c4", False),
        ]
        m = compute_hh(base, reviewed)
        assert m.helpfulness == pytest.approx(0.75)
        assert m.harmfulness == pytest.approx(0.25)
        assert m.benefit_risk_ratio == pytest.approx(3.0)
        assert is_net_positive(m) is True

    def test_approx_two_one_ratio_matches_gpt4o(self) -> None:
        """2.1:1: 21 corrections / 30 errors, 10 degradations / 30 correct."""
        base = [_base(f"e{i}", False) for i in range(30)] + [_base(f"c{i}", True) for i in range(30)]
        reviewed = (
            [_reviewed(f"e{i}", i < 21) for i in range(30)]
            + [_reviewed(f"c{i}", i >= 10) for i in range(30)]
        )
        m = compute_hh(base, reviewed)
        assert m.helpfulness == pytest.approx(21 / 30)
        assert m.harmfulness == pytest.approx(10 / 30)
        assert m.benefit_risk_ratio == pytest.approx(2.1)
        assert is_net_positive(m) is True


class TestEdgeCases:
    def test_empty_lists_yield_zero_metric(self) -> None:
        m = compute_hh([], [])
        assert m.helpfulness == 0.0
        assert m.harmfulness == 0.0
        assert m.benefit_risk_ratio is None
        assert m.n_base_errors == 0
        assert m.n_base_correct == 0

    def test_mismatched_task_ids_raise_valueerror(self) -> None:
        base = [_base("t1", True)]
        reviewed = [_reviewed("t2", True)]
        with pytest.raises(ValueError, match="same task_ids"):
            compute_hh(base, reviewed)

    def test_order_independence_via_task_id_pairing(self) -> None:
        """Pairing must be by task_id, not list position."""
        base = [_base("t1", False), _base("t2", True)]
        reviewed = [_reviewed("t2", True, intervened=False), _reviewed("t1", True)]
        m = compute_hh(base, reviewed)
        assert m.n_corrected == 1
        assert m.n_degraded == 0
        assert m.helpfulness == 1.0
        assert m.harmfulness == 0.0

    def test_break_even_not_net_positive(self) -> None:
        """A reviewer that fixes 1 error and breaks 1 correct (in equal populations) is not net positive."""
        base = [_base("e1", False), _base("c1", True)]
        reviewed = [_reviewed("e1", True), _reviewed("c1", False)]
        m = compute_hh(base, reviewed)
        assert m.helpfulness == 1.0
        assert m.harmfulness == 1.0
        assert m.benefit_risk_ratio == pytest.approx(1.0)
        assert is_net_positive(m) is False  # strict inequality


class TestResultShape:
    """Guard the public dataclass surface."""

    def test_metric_is_frozen(self) -> None:
        m = HelpfulnessHarmfulnessMetric(
            helpfulness=0.5, harmfulness=0.2, benefit_risk_ratio=2.5,
            n_base_errors=4, n_base_correct=5, n_corrected=2, n_degraded=1,
        )
        with pytest.raises((AttributeError, Exception)):
            m.helpfulness = 0.0  # type: ignore[misc]
