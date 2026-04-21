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
        # Negative values parse and clip to 0 (the best-case throttle
        # from the LM's perspective — the theorem passes trivially).
        # Previous restrictive regex defaulted to 1.0; the broader
        # header-location parser (Roborev #867) accepts negatives and
        # clamps.
        assert parse_throttle("policy_throttle = -0.5") == 0.0

    def test_marker_on_its_own_line(self) -> None:
        """Real assignments are expected to start a line (possibly
        indented).  Leading prose + trailing prose on *other* lines is
        fine; mid-line mentions are treated as prose (see
        test_prose_mention_does_not_override)."""
        text = "preamble text\npolicy_throttle = 0.25\ntrailing explanation"
        assert parse_throttle(text) == 0.25

    def test_indented_marker(self) -> None:
        """Leading whitespace on the assignment line is allowed."""
        assert parse_throttle("  policy_throttle = 0.7") == 0.7

    def test_multiple_assignments_last_wins(self) -> None:
        """Regression for Roborev #864 H.

        LM mutations commonly append a revised value without removing
        the old one.  The harness must read the *last* assignment —
        otherwise valid mutations score against a stale throttle and
        convergence rates are systematically wrong.
        """
        text = "policy_throttle = 1.0\n# I've refined this:\npolicy_throttle = 0.3"
        assert parse_throttle(text) == 0.3

    def test_multiple_assignments_three_way(self) -> None:
        text = "policy_throttle: 0.9\npolicy_throttle = 0.6\npolicy_throttle=0.2"
        assert parse_throttle(text) == 0.2

    def test_invalid_last_assignment_falls_back_to_default(self) -> None:
        """Regression for Roborev #867.

        If the FINAL policy_throttle assignment is malformed (non-numeric
        RHS, empty RHS, or just punctuation), parse_throttle must fall
        back to _DEFAULT_THROTTLE — NOT to an earlier valid assignment.
        Falling through to a stale earlier value would silently score
        an invalid LM revision as if it were the prior valid one.
        """
        # Non-numeric word as last assignment.
        assert (
            parse_throttle(
                "policy_throttle = 0.3\npolicy_throttle = nope"
            )
            == 1.0
        )
        # Empty RHS.
        assert parse_throttle("policy_throttle = 0.3\npolicy_throttle =") == 1.0
        # Just a dash.
        assert (
            parse_throttle(
                "policy_throttle = 0.3\npolicy_throttle = -"
            )
            == 1.0
        )

    def test_valid_last_overrides_invalid_earlier(self) -> None:
        """Mirror of the above: an invalid earlier assignment does not
        prevent the valid last assignment from being read."""
        assert (
            parse_throttle("policy_throttle = nope\npolicy_throttle = 0.2")
            == 0.2
        )

    def test_prose_mention_does_not_override(self) -> None:
        """Regression for Roborev #868.

        A real assignment followed by a prose mention of the keyword
        (e.g. an LM quoting an earlier attempt in explanatory text)
        must NOT be treated as overriding.  The prose mention is not
        at a line start, so the anchored header regex ignores it.
        """
        text = (
            "policy_throttle = 0.2\n"
            "Previous bad output: policy_throttle = nope"
        )
        assert parse_throttle(text) == 0.2

    def test_prose_mention_with_valid_number_also_ignored(self) -> None:
        """Even numeric-looking prose mentions mid-line are prose."""
        text = (
            "policy_throttle = 0.2\n"
            "For reference, the old value was policy_throttle = 0.9."
        )
        assert parse_throttle(text) == 0.2

    def test_quoted_example_does_not_override(self) -> None:
        """Example: an LM that explains its reasoning by quoting an
        earlier attempt.  The explanation must not be read as an
        assignment."""
        text = (
            "policy_throttle = 0.3\n"
            "(As a reminder, initial was policy_throttle = 1.0.)"
        )
        assert parse_throttle(text) == 0.3


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
