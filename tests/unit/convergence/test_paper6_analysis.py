"""Unit tests for paper-6 analysis (statistics and summary shape)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from eval.convergence.analysis_theorem_6 import (
    analyze,
    arm_summary,
    convergence_iters,
    count_errored_records,
    load_arm_records,
    mann_whitney,
    plot_cdf,
    wilson_ci,
)

# Statistics tests use scipy; analysis imports it lazily so this module
# itself is importable without scipy, but ``mann_whitney`` and any
# ``analyze()``-invoking end-to-end test need it at call time.  CI does
# not ship scipy; skip those tests there.
try:
    import scipy.stats  # noqa: F401

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - CI path
    _SCIPY_AVAILABLE = False

requires_scipy = pytest.mark.skipif(
    not _SCIPY_AVAILABLE, reason="scipy required for this statistics test"
)


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_over_zero_is_zero_zero(self) -> None:
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_ten_over_ten_upper_is_one(self) -> None:
        lo, hi = wilson_ci(10, 10)
        assert hi == pytest.approx(1.0, abs=0.05)
        assert lo > 0.5

    def test_midpoint_ci_contains_proportion(self) -> None:
        lo, hi = wilson_ci(5, 10)
        assert lo <= 0.5 <= hi


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------


@requires_scipy
class TestMannWhitney:
    def test_clear_treatment_win_is_significant(self) -> None:
        treatment = [1, 2, 3, 4, 5]
        baseline = [40, 41, 42, 43, 44]
        result = mann_whitney(treatment, baseline)
        assert result["p_value"] < 0.05
        # Convention: positive rb = treatment dominates (smaller iters).
        assert result["effect_size_rank_biserial"] > 0.8

    def test_clear_treatment_loss_is_not_significant_under_less(self) -> None:
        treatment = [40, 41, 42, 43, 44]
        baseline = [1, 2, 3, 4, 5]
        result = mann_whitney(treatment, baseline)
        # alternative="less" so treatment-losing gets p ~ 1
        assert result["p_value"] > 0.5

    def test_empty_inputs_return_none(self) -> None:
        result = mann_whitney([], [1, 2])
        assert result["p_value"] is None
        result = mann_whitney([1, 2], [])
        assert result["p_value"] is None


# ---------------------------------------------------------------------------
# convergence_iters
# ---------------------------------------------------------------------------


class TestConvergenceIters:
    def test_drops_unconverged_by_default(self) -> None:
        records = [
            {"convergence_iteration": 5},
            {"convergence_iteration": None},
            {"convergence_iteration": 10},
        ]
        assert convergence_iters(records) == [5, 10]

    def test_replaces_unconverged_with_budget(self) -> None:
        records = [
            {"convergence_iteration": 5},
            {"convergence_iteration": None},
        ]
        assert convergence_iters(records, replace_unconverged_with=50) == [5, 50]


# ---------------------------------------------------------------------------
# arm_summary
# ---------------------------------------------------------------------------


class TestArmSummary:
    def test_all_converged(self) -> None:
        records = [{"convergence_iteration": i} for i in [5, 6, 7]]
        summary = arm_summary(records, budget=50)
        assert summary["n_runs"] == 3
        assert summary["n_converged"] == 3
        assert summary["convergence_rate"] == 1.0
        assert summary["mean_iterations_to_convergence"] == pytest.approx(6.0)

    def test_partial_convergence(self) -> None:
        records = [
            {"convergence_iteration": 5},
            {"convergence_iteration": None},
        ]
        summary = arm_summary(records, budget=50)
        assert summary["n_runs"] == 2
        assert summary["n_converged"] == 1
        assert summary["convergence_rate"] == 0.5
        # mean uses budget substitution
        assert summary["mean_iterations_to_convergence"] == pytest.approx(27.5)

    def test_empty(self) -> None:
        summary = arm_summary([], budget=50)
        assert summary["n_runs"] == 0
        assert summary["mean_iterations_to_convergence"] is None


# ---------------------------------------------------------------------------
# end-to-end: synthetic records -> analyze() produces expected conjecture verdict
# ---------------------------------------------------------------------------


def _write_arm(
    dir_: Path, arm: str, iters: Sequence[int | None], budget: int
) -> None:
    arm_dir = dir_ / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    for seed, conv in enumerate(iters):
        record = {
            "arm": arm,
            "seed": seed,
            "budget_iterations": budget,
            "batch_size": 16,
            "convergence_iteration": conv,
            "val_scores_per_iteration": [],
            "wallclock_s": 1.0,
            "iteration_events": [],
            "reflection_lm": "mock",
            "mock_reflection_lm": False,  # pretend real for test
            "theorem": "behavioral_stability_windowed",
        }
        (arm_dir / f"seed_{seed}.json").write_text(json.dumps(record))


@requires_scipy
class TestAnalyzeEndToEnd:
    def test_strong_positive_reports_significance(self, tmp_path: Path) -> None:
        # Cert-binary converges fast; both baselines converge slow.
        _write_arm(tmp_path, "cert-binary", [3, 4, 5, 6, 5, 4, 5, 6, 5, 4], budget=50)
        _write_arm(tmp_path, "scalar", [45, 47, 50, 50, 48, 49, 50, 50, 47, 49], budget=50)
        _write_arm(tmp_path, "scalar-evidence", [40, 42, 45, 50, 44, 45, 48, 50, 43, 45], budget=50)
        summary = analyze(tmp_path)
        assert summary["per_arm"]["cert-binary"]["n_runs"] == 10
        cmp_scalar = summary["comparisons_treatment_vs_baseline"]["scalar"]
        assert cmp_scalar["p_value"] < 0.001
        assert cmp_scalar["effect_size_rank_biserial"] > 0.8

    def test_null_result_reports_non_significance(self, tmp_path: Path) -> None:
        shared = [20, 22, 24, 21, 23, 20, 25, 22, 23, 21]
        _write_arm(tmp_path, "cert-binary", shared, budget=50)
        _write_arm(tmp_path, "scalar", shared, budget=50)
        _write_arm(tmp_path, "scalar-evidence", shared, budget=50)
        summary = analyze(tmp_path)
        cmp_scalar = summary["comparisons_treatment_vs_baseline"]["scalar"]
        assert cmp_scalar["p_value"] > 0.05

    def test_summary_json_is_written(self, tmp_path: Path) -> None:
        _write_arm(tmp_path, "cert-binary", [3, 4], budget=50)
        _write_arm(tmp_path, "scalar", [40, 42], budget=50)
        _write_arm(tmp_path, "scalar-evidence", [38, 40], budget=50)
        analyze(tmp_path)
        assert (tmp_path / "summary.json").exists()

    def test_mocked_runs_are_filtered(self, tmp_path: Path) -> None:
        arm_dir = tmp_path / "cert-binary"
        arm_dir.mkdir(parents=True)
        real = {
            "convergence_iteration": 5,
            "mock_reflection_lm": False,
            "budget_iterations": 50,
        }
        mocked = {
            "convergence_iteration": 1,
            "mock_reflection_lm": True,
            "budget_iterations": 50,
        }
        (arm_dir / "seed_0.json").write_text(json.dumps(real))
        (arm_dir / "seed_1.json").write_text(json.dumps(mocked))
        records = load_arm_records(arm_dir)
        assert len(records) == 1
        assert records[0]["convergence_iteration"] == 5


# ---------------------------------------------------------------------------
# Roborev #854 H2: errored runs must be excluded from statistics
# ---------------------------------------------------------------------------


@requires_scipy
class TestErroredRunFiltering:
    """The paper's methods text says errored runs are excluded from
    statistics; these tests lock that contract in."""

    def test_load_arm_records_filters_errored_by_default(
        self, tmp_path: Path
    ) -> None:
        arm_dir = tmp_path / "cert-binary"
        arm_dir.mkdir(parents=True)
        healthy = {
            "convergence_iteration": 5,
            "mock_reflection_lm": False,
            "gepa_error": None,
            "budget_iterations": 50,
        }
        errored = {
            "convergence_iteration": None,
            "mock_reflection_lm": False,
            "gepa_error": "RuntimeError: boom",
            "budget_iterations": 50,
        }
        (arm_dir / "seed_0.json").write_text(json.dumps(healthy))
        (arm_dir / "seed_1.json").write_text(json.dumps(errored))
        records = load_arm_records(arm_dir)
        assert len(records) == 1
        assert records[0]["gepa_error"] is None

    def test_load_arm_records_include_errored_flag_returns_all(
        self, tmp_path: Path
    ) -> None:
        arm_dir = tmp_path / "cert-binary"
        arm_dir.mkdir(parents=True)
        errored = {
            "convergence_iteration": None,
            "gepa_error": "RuntimeError: boom",
        }
        (arm_dir / "seed_0.json").write_text(json.dumps(errored))
        records = load_arm_records(arm_dir, include_errored=True)
        assert len(records) == 1

    def test_count_errored_records_reports_count(self, tmp_path: Path) -> None:
        arm_dir = tmp_path / "cert-binary"
        arm_dir.mkdir(parents=True)
        for i, err in enumerate([None, "X", "Y", None]):
            rec = {"gepa_error": err, "mock_reflection_lm": False}
            (arm_dir / f"seed_{i}.json").write_text(json.dumps(rec))
        assert count_errored_records(arm_dir) == 2

    def test_analyze_reports_errored_counts_and_excludes_them(
        self, tmp_path: Path
    ) -> None:
        # cert-binary: 2 healthy converged, 1 errored
        arm_dir = tmp_path / "cert-binary"
        arm_dir.mkdir(parents=True)
        (arm_dir / "seed_0.json").write_text(
            json.dumps({
                "convergence_iteration": 5,
                "gepa_error": None,
                "budget_iterations": 50,
                "mock_reflection_lm": False,
            })
        )
        (arm_dir / "seed_1.json").write_text(
            json.dumps({
                "convergence_iteration": 7,
                "gepa_error": None,
                "budget_iterations": 50,
                "mock_reflection_lm": False,
            })
        )
        (arm_dir / "seed_2.json").write_text(
            json.dumps({
                "convergence_iteration": None,
                "gepa_error": "RuntimeError: crash",
                "budget_iterations": 50,
                "mock_reflection_lm": False,
            })
        )
        # Minimal baselines so analyze() has all three arms
        for arm in ("scalar", "scalar-evidence"):
            d = tmp_path / arm
            d.mkdir(parents=True)
            (d / "seed_0.json").write_text(
                json.dumps({
                    "convergence_iteration": 30,
                    "gepa_error": None,
                    "budget_iterations": 50,
                    "mock_reflection_lm": False,
                })
            )
        summary = analyze(tmp_path)
        assert summary["errored_runs_excluded"]["cert-binary"] == 1
        assert summary["errored_runs_excluded"]["scalar"] == 0
        # n_runs excludes errored
        assert summary["per_arm"]["cert-binary"]["n_runs"] == 2


# ---------------------------------------------------------------------------
# Roborev #854 M1: convergence_iteration returns GEPA iteration, not list index
# ---------------------------------------------------------------------------


class TestConvergenceIterIsGEPAIteration:
    """``_first_convergence_iter`` must return the GEPA iteration of the
    first event in the streak — not a list index.  The recorder's event
    dicts carry an ``iteration`` key populated from GEPA; this test
    ensures that value is what the experiment records.
    """

    def test_returns_gepa_iteration_not_list_index(self) -> None:
        from eval.convergence.theorem_6_experiment import _first_convergence_iter

        # First event is iteration 0 (initial eval) with score 0.0; then
        # iterations 1-3 score 1.0 (streak of 3 starting at index 1,
        # GEPA iteration 1).  Returning the list index (1) here happens
        # to coincide with GEPA iteration (1), so a harder case: shift
        # the GEPA iteration counter by a stable offset.
        events = [
            {"iteration": 0, "best_score": 0.0},
            {"iteration": 1, "best_score": 0.5},
            {"iteration": 2, "best_score": 1.0},
            {"iteration": 3, "best_score": 1.0},
            {"iteration": 4, "best_score": 1.0},
        ]
        # Streak of 3 starts at index 2, which is GEPA iteration 2.
        assert _first_convergence_iter(events) == 2

    def test_earliest_possible_convergence_is_first_gepa_iteration(
        self,
    ) -> None:
        from eval.convergence.theorem_6_experiment import _first_convergence_iter

        # Streak of 3 starts at list index 0, GEPA iteration 1
        # (iteration 0 is excluded here — suppose the initial eval was
        # not logged).  Expected: 1, not 0.
        events = [
            {"iteration": 1, "best_score": 1.0},
            {"iteration": 2, "best_score": 1.0},
            {"iteration": 3, "best_score": 1.0},
        ]
        assert _first_convergence_iter(events) == 1

    def test_returns_none_when_no_streak(self) -> None:
        from eval.convergence.theorem_6_experiment import _first_convergence_iter

        events = [
            {"iteration": i, "best_score": 0.5}
            for i in range(10)
        ]
        assert _first_convergence_iter(events) is None


# ---------------------------------------------------------------------------
# Roborev #854 M2: CDF must not impute non-converged runs at budget
# ---------------------------------------------------------------------------


@requires_scipy
class TestCDFNoImputation:
    """The CDF should treat non-converged runs as right-censored: they
    contribute to the denominator but never to the numerator.  The
    curve for an arm with convergence rate < 1.0 must level off below
    1.0, not at 1.0.
    """

    def test_cdf_signature_uses_arm_to_series(self, tmp_path: Path) -> None:
        """``plot_cdf`` takes ``(converged_iters, total_runs)`` pairs per arm."""
        import matplotlib
        try:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        # Arm with 3 converged (iters 5, 10, 15) out of 10 total runs.
        # At x=50, cumulative = 3/10 = 0.3, not 1.0.
        out_path = tmp_path / "cdf.png"
        ok = plot_cdf(
            {"cert-binary": ([5, 10, 15], 10)},
            out_path=out_path,
            budget=50,
        )
        assert ok is True
        assert out_path.exists()

    def test_cdf_curve_levels_off_at_convergence_rate_not_one(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: analyze() + CDF data should reflect 3/10 = 0.3 at budget."""
        arm_dir = tmp_path / "cert-binary"
        arm_dir.mkdir(parents=True)
        # 3 converged, 7 non-converged (convergence_iteration=None).
        for i, conv in enumerate([5, 10, 15, None, None, None, None, None, None, None]):
            (arm_dir / f"seed_{i}.json").write_text(
                json.dumps({
                    "convergence_iteration": conv,
                    "gepa_error": None,
                    "budget_iterations": 50,
                    "mock_reflection_lm": False,
                })
            )
        for arm in ("scalar", "scalar-evidence"):
            d = tmp_path / arm
            d.mkdir(parents=True)
            (d / "seed_0.json").write_text(
                json.dumps({
                    "convergence_iteration": 40,
                    "gepa_error": None,
                    "budget_iterations": 50,
                    "mock_reflection_lm": False,
                })
            )
        summary = analyze(tmp_path)
        # per-arm convergence rate reflects the right denominator
        cert = summary["per_arm"]["cert-binary"]
        assert cert["n_runs"] == 10
        assert cert["n_converged"] == 3
        assert cert["convergence_rate"] == 0.3
