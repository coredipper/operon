"""Unit tests for the paper-6 synthetic signal harness."""

from __future__ import annotations

from eval.convergence.synthetic_signal_harness import (
    SEED_CANDIDATE_TEXT,
    SEED_COMPONENT_NAME,
    SyntheticDataset,
    TaskConfig,
    Trajectory,
    parse_throttle,
    run_rollout,
    seed_candidate,
)


class TestParseThrottle:
    """Tests for parse_throttle."""

    def test_parses_equals_form(self) -> None:
        assert parse_throttle("policy_throttle = 0.3") == 0.3

    def test_parses_colon_form(self) -> None:
        assert parse_throttle("policy_throttle: 0.45") == 0.45

    def test_case_insensitive(self) -> None:
        assert parse_throttle("POLICY_THROTTLE = 0.2") == 0.2

    def test_missing_marker_returns_default(self) -> None:
        assert parse_throttle("no marker here") == 1.0

    def test_empty_returns_default(self) -> None:
        assert parse_throttle("") == 1.0

    def test_clips_above_one(self) -> None:
        assert parse_throttle("policy_throttle = 5") == 1.0

    def test_clips_below_zero(self) -> None:
        # regex requires an optional decimal — negative numbers don't match,
        # so they fall back to default.  This is intentional: the LM is
        # meant to emit non-negative floats.
        assert parse_throttle("policy_throttle = -0.5") == 1.0

    def test_extra_text_around_marker(self) -> None:
        assert (
            parse_throttle("some prose. policy_throttle = 0.25 more prose.")
            == 0.25
        )


class TestRunRollout:
    """Tests for run_rollout — determinism and theorem-parameter shape."""

    def test_is_deterministic_under_fixed_seed(self) -> None:
        out1 = run_rollout("policy_throttle = 0.5", 7, run_seed=42)
        out2 = run_rollout("policy_throttle = 0.5", 7, run_seed=42)
        assert out1[1].window_means == out2[1].window_means

    def test_different_data_inst_ids_produce_different_trajectories(self) -> None:
        _, t1, _ = run_rollout("policy_throttle = 0.5", 0, run_seed=42)
        _, t2, _ = run_rollout("policy_throttle = 0.5", 1, run_seed=42)
        assert t1.window_means != t2.window_means

    def test_different_run_seeds_produce_different_trajectories(self) -> None:
        _, t1, _ = run_rollout("policy_throttle = 0.5", 0, run_seed=1)
        _, t2, _ = run_rollout("policy_throttle = 0.5", 0, run_seed=2)
        assert t1.window_means != t2.window_means

    def test_high_throttle_violates_threshold(self) -> None:
        cfg = TaskConfig()
        _, traj, _ = run_rollout(SEED_CANDIDATE_TEXT, 0, config=cfg, run_seed=0)
        assert all(m > cfg.threshold for m in traj.window_means)
        assert traj.violating_windows == tuple(
            range(cfg.windows_per_rollout)
        )

    def test_low_throttle_passes_threshold(self) -> None:
        cfg = TaskConfig()
        _, traj, _ = run_rollout(
            "policy_throttle = 0.1", 0, config=cfg, run_seed=0
        )
        assert all(m <= cfg.threshold for m in traj.window_means)
        assert traj.violating_windows == ()

    def test_returns_theorem_parameters(self) -> None:
        _, _, params = run_rollout("policy_throttle = 0.5", 0, run_seed=0)
        assert "signal_values" in params
        assert "threshold" in params
        assert isinstance(params["signal_values"], tuple)

    def test_trajectory_is_dataclass_shape(self) -> None:
        _, traj, _ = run_rollout("policy_throttle = 0.5", 0, run_seed=0)
        assert isinstance(traj, Trajectory)
        assert traj.parsed_throttle == 0.5


class TestSyntheticDataset:
    """Tests for SyntheticDataset."""

    def test_iter_and_len_match(self) -> None:
        d = SyntheticDataset(size=5, offset=100)
        assert list(d) == [100, 101, 102, 103, 104]
        assert len(d) == 5

    def test_getitem_respects_offset(self) -> None:
        d = SyntheticDataset(size=3, offset=10)
        assert d[0] == 10
        assert d[2] == 12


class TestSeedCandidate:
    """Tests for the starting candidate mapping."""

    def test_uses_seed_component_name(self) -> None:
        cand = seed_candidate()
        assert set(cand.keys()) == {SEED_COMPONENT_NAME}

    def test_starts_with_failing_throttle(self) -> None:
        cand = seed_candidate()
        assert parse_throttle(cand[SEED_COMPONENT_NAME]) >= 1.0
