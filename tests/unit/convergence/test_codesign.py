"""Tests for operon_ai.convergence.codesign -- Zardini co-design composition."""

from __future__ import annotations

import pytest

from operon_ai.convergence.codesign import (
    DesignProblem,
    compose_parallel,
    compose_series,
    feasibility_check,
    feedback_fixed_point,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _double_dp() -> DesignProblem:
    """DP that doubles every numeric value in the resource dict."""
    return DesignProblem(
        name="double",
        evaluate_fn=lambda r: {k: v * 2 for k, v in r.items()},
    )


def _increment_dp() -> DesignProblem:
    """DP that increments every numeric value by 1."""
    return DesignProblem(
        name="increment",
        evaluate_fn=lambda r: {k: v + 1 for k, v in r.items()},
    )


def _threshold_dp(key: str, minimum: float) -> DesignProblem:
    """DP that is only feasible when resource[key] >= minimum."""
    return DesignProblem(
        name=f"threshold({key}>={minimum})",
        evaluate_fn=lambda r: {**r, "passed": True},
        feasibility_fn=lambda r: r.get(key, 0) >= minimum,
    )


# ---------------------------------------------------------------------------
# DesignProblem basics
# ---------------------------------------------------------------------------

class TestDesignProblem:

    def test_design_problem_evaluate(self) -> None:
        dp = _double_dp()
        result = dp.evaluate({"x": 3, "y": 5})
        assert result == {"x": 6, "y": 10}

    def test_design_problem_feasibility(self) -> None:
        dp_always = _double_dp()
        assert dp_always.is_feasible({"x": 0}) is True

        dp_gated = _threshold_dp("budget", 100.0)
        assert dp_gated.is_feasible({"budget": 50.0}) is False
        assert dp_gated.is_feasible({"budget": 100.0}) is True


# ---------------------------------------------------------------------------
# Series composition
# ---------------------------------------------------------------------------

class TestComposeSeries:

    def test_compose_series(self) -> None:
        """dp1 doubles, dp2 increments => double then increment."""
        composite = compose_series(_double_dp(), _increment_dp())
        result = composite.evaluate({"a": 3})
        # 3 * 2 = 6, then 6 + 1 = 7
        assert result == {"a": 7}

    def test_compose_series_infeasible(self) -> None:
        """If dp1 is infeasible the composite must be infeasible."""
        dp1 = _threshold_dp("budget", 100.0)
        dp2 = _increment_dp()
        composite = compose_series(dp1, dp2)
        assert composite.is_feasible({"budget": 10.0}) is False

    def test_compose_series_name(self) -> None:
        composite = compose_series(_double_dp(), _increment_dp())
        assert composite.name == "double→increment"

    def test_compose_series_custom_name(self) -> None:
        composite = compose_series(_double_dp(), _increment_dp(), name="my_pipe")
        assert composite.name == "my_pipe"


# ---------------------------------------------------------------------------
# Parallel composition
# ---------------------------------------------------------------------------

class TestComposeParallel:

    def test_compose_parallel(self) -> None:
        """Parallel merges outputs from both DPs."""
        dp1 = DesignProblem(
            name="left",
            evaluate_fn=lambda r: {"sum": r["a"] + r["b"]},
        )
        dp2 = DesignProblem(
            name="right",
            evaluate_fn=lambda r: {"product": r["a"] * r["b"]},
        )
        composite = compose_parallel(dp1, dp2)
        result = composite.evaluate({"a": 3, "b": 4})
        assert result == {"sum": 7, "product": 12}

    def test_compose_parallel_infeasible(self) -> None:
        """One infeasible branch makes the whole composite infeasible."""
        dp_ok = _double_dp()
        dp_gated = _threshold_dp("budget", 100.0)
        composite = compose_parallel(dp_ok, dp_gated)
        assert composite.is_feasible({"budget": 10.0}) is False

    def test_compose_parallel_name(self) -> None:
        composite = compose_parallel(_double_dp(), _increment_dp())
        assert composite.name == "(double‖increment)"


# ---------------------------------------------------------------------------
# Feedback fixed-point
# ---------------------------------------------------------------------------

class TestFeedbackFixedPoint:

    def test_feedback_fixed_point_converges(self) -> None:
        """Scoring loop that converges: halve the distance to target each step."""
        dp = DesignProblem(
            name="halver",
            evaluate_fn=lambda r: {"score": r["score"] + (1.0 - r["score"]) / 2},
        )
        final, iters, converged = feedback_fixed_point(
            dp, {"score": 0.0}, convergence_key="score", epsilon=0.01,
        )
        assert converged is True
        assert iters < 100
        assert abs(final["score"] - 1.0) < 0.02

    def test_feedback_fixed_point_diverges(self) -> None:
        """Non-converging loop returns converged=False."""
        dp = DesignProblem(
            name="grower",
            evaluate_fn=lambda r: {"score": r["score"] + 1.0},
        )
        final, iters, converged = feedback_fixed_point(
            dp, {"score": 0.0}, convergence_key="score", epsilon=0.01,
            max_iterations=10,
        )
        assert converged is False
        assert iters == 10

    def test_feedback_fixed_point_exact_match(self) -> None:
        """State unchanged on first step => converged on iteration 1."""
        dp = DesignProblem(
            name="identity",
            evaluate_fn=lambda r: dict(r),
        )
        final, iters, converged = feedback_fixed_point(dp, {"x": 42})
        assert converged is True
        assert iters == 1
        assert final == {"x": 42}


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

class TestFeasibilityCheck:

    def test_feasibility_check_feasible(self) -> None:
        dp = _double_dp()
        info = feasibility_check(dp, {"x": 5})
        assert info["feasible"] is True
        assert info["functionalities"] == {"x": 10}
        assert "reason" not in info

    def test_feasibility_check_infeasible(self) -> None:
        dp = _threshold_dp("budget", 100.0)
        info = feasibility_check(dp, {"budget": 10.0})
        assert info["feasible"] is False
        assert "reason" in info
        assert isinstance(info["reason"], str)
        assert "functionalities" not in info
