"""Unit tests for the paper-6 synthetic signal harness."""

from __future__ import annotations

from eval.convergence.synthetic_signal_harness import (
    SEED_CANDIDATE_TEXT,
    SEED_COMPONENT_NAME,
    SyntheticDataset,
    TaskConfig,
    Trajectory,
    candidate_text_with_throttle,
    parse_throttle,
    run_rollout,
    seed_candidate,
)


def _config(value: object) -> str:
    """Return a well-formed ``CONFIG: policy_throttle = value`` line.

    Concentrates the prefix boilerplate in one place so tests read as
    the *scenario* they describe rather than as formatting concerns.
    """
    return f"CONFIG: policy_throttle = {value}"


class TestParseThrottleConfigPrefix:
    """The parser's contract: the assignment MUST appear on a line
    prefixed with ``CONFIG:`` (optionally indented).  Anything else
    is prose and ignored.

    The prefix — rather than a markdown fenced code block — was chosen
    because GEPA's default reflective proposer strips markdown fences
    during normalization, so a fenced marker would be lost across the
    mutation round trip (Roborev #870).  A plain-text prefix survives
    intact."""

    def test_parses_equals_form(self) -> None:
        assert parse_throttle(_config(0.3)) == 0.3

    def test_parses_colon_form(self) -> None:
        assert parse_throttle("CONFIG: policy_throttle: 0.45") == 0.45

    def test_case_insensitive_config_and_keyword(self) -> None:
        assert parse_throttle("config: POLICY_THROTTLE = 0.2") == 0.2
        assert parse_throttle("CONFIG: POLICY_THROTTLE = 0.15") == 0.15

    def test_clips_above_one(self) -> None:
        assert parse_throttle(_config(5)) == 1.0

    def test_clips_below_zero(self) -> None:
        """Negative values clip to 0 — best-case throttle semantics."""
        assert parse_throttle(_config(-0.5)) == 0.0

    def test_indented_marker(self) -> None:
        """Leading whitespace on the CONFIG line is allowed."""
        assert parse_throttle("  CONFIG: policy_throttle = 0.7") == 0.7

    def test_last_config_line_wins(self) -> None:
        """Regression for Roborev #864 H: append-to-revise."""
        text = f"{_config(1.0)}\n# refined:\n{_config(0.3)}"
        assert parse_throttle(text) == 0.3

    def test_multiple_config_lines_three_way(self) -> None:
        text = f"{_config(0.9)}\n{_config(0.6)}\n{_config(0.2)}"
        assert parse_throttle(text) == 0.2


class TestParseThrottleMalformedFinalAssignment:
    """Regression for Roborev #867 — malformed FINAL CONFIG: line must
    default, never fall through to an earlier valid one."""

    def test_non_numeric_last(self) -> None:
        text = f"{_config(0.3)}\nCONFIG: policy_throttle = nope"
        assert parse_throttle(text) == 1.0

    def test_empty_rhs_last(self) -> None:
        text = f"{_config(0.3)}\nCONFIG: policy_throttle ="
        assert parse_throttle(text) == 1.0

    def test_just_dash_last(self) -> None:
        text = f"{_config(0.3)}\nCONFIG: policy_throttle = -"
        assert parse_throttle(text) == 1.0

    def test_valid_last_overrides_invalid_earlier(self) -> None:
        text = f"CONFIG: policy_throttle = nope\n{_config(0.2)}"
        assert parse_throttle(text) == 0.2


class TestParseThrottleOutsideConfigIgnored:
    """Regression for Roborev #868, #869, and #870 — assignments
    without the ``CONFIG:`` prefix are prose and must never override
    the real config."""

    def test_bare_assignment_without_config_prefix_returns_default(
        self,
    ) -> None:
        """An LM that drops the CONFIG: prefix produces a malformed
        prompt; parse_throttle returns _DEFAULT_THROTTLE so reflection
        feedback signals the LM to restore the protocol."""
        assert parse_throttle("policy_throttle = 0.3") == 1.0

    def test_quoted_example_on_own_line_ignored(self) -> None:
        """Roborev #869: a later ``policy_throttle = 1.0`` on its own
        line that lacks the CONFIG: prefix must NOT override the real
        config."""
        text = f"{_config(0.2)}\n\nExample of prior output:\npolicy_throttle = 1.0"
        assert parse_throttle(text) == 0.2

    def test_prose_mid_line_mention_ignored(self) -> None:
        """Roborev #868 generalized: mid-line prose never counts."""
        text = (
            _config(0.2)
            + "\nFor reference: previous bad output said policy_throttle = nope"
        )
        assert parse_throttle(text) == 0.2

    def test_quoted_parenthetical_example(self) -> None:
        text = _config(0.3) + "\n(As a reminder, initial was policy_throttle = 1.0.)"
        assert parse_throttle(text) == 0.3

    def test_full_prose_paragraph_with_throttle_mentions(self) -> None:
        """An LM that writes a reasoning paragraph mentioning multiple
        throttle values but uses CONFIG: only on the real one."""
        text = _config(0.25) + (
            "\n\nReasoning:\n"
            "policy_throttle = 1.0  was the seed\n"
            "policy_throttle = 0.7  was my previous attempt\n"
            "so I have now set it to 0.25 above."
        )
        assert parse_throttle(text) == 0.25

    def test_non_config_markdown_fence_is_ignored(self) -> None:
        """Markdown code fences (config-tagged or not) do NOT count —
        only the CONFIG: prefix does.  Regression for Roborev #870:
        a later ``text`` fence cannot override, and earlier attempts
        at fence-based protocols are not accidentally re-accepted."""
        text = (
            _config(0.3)
            + "\n\nExample:\n```config\npolicy_throttle = 1.0\n```"
        )
        assert parse_throttle(text) == 0.3

    def test_prose_mention_with_config_keyword_not_at_line_start_ignored(
        self,
    ) -> None:
        """Roborev #870 spirit: even ``CONFIG:`` inside prose must be
        at line start to count."""
        text = (
            _config(0.3)
            + "\nAs seen in CONFIG: policy_throttle = 1.0 (prior)"
        )
        assert parse_throttle(text) == 0.3


class TestParseThrottleMissingMarker:
    """Missing / empty / irrelevant text defaults safely."""

    def test_empty_returns_default(self) -> None:
        assert parse_throttle("") == 1.0

    def test_missing_marker_returns_default(self) -> None:
        assert parse_throttle("no marker here") == 1.0

    def test_config_keyword_without_throttle_returns_default(self) -> None:
        assert parse_throttle("CONFIG: something_else = 5") == 1.0

    def test_seed_prompt_parses_as_failing(self) -> None:
        """Sanity: the built-in seed prompt uses the CONFIG: protocol
        and the parser reads it as the starting 1.0 value."""
        assert parse_throttle(SEED_CANDIDATE_TEXT) == 1.0


class TestRunRollout:
    """Tests for run_rollout — determinism and theorem-parameter shape."""

    def test_is_deterministic_under_fixed_seed(self) -> None:
        cand = candidate_text_with_throttle(0.5)
        out1 = run_rollout(cand, 7, run_seed=42)
        out2 = run_rollout(cand, 7, run_seed=42)
        assert out1[1].window_means == out2[1].window_means

    def test_different_data_inst_ids_produce_different_trajectories(self) -> None:
        cand = candidate_text_with_throttle(0.5)
        _, t1, _ = run_rollout(cand, 0, run_seed=42)
        _, t2, _ = run_rollout(cand, 1, run_seed=42)
        assert t1.window_means != t2.window_means

    def test_different_run_seeds_produce_different_trajectories(self) -> None:
        cand = candidate_text_with_throttle(0.5)
        _, t1, _ = run_rollout(cand, 0, run_seed=1)
        _, t2, _ = run_rollout(cand, 0, run_seed=2)
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
            candidate_text_with_throttle(0.1), 0, config=cfg, run_seed=0
        )
        assert all(m <= cfg.threshold for m in traj.window_means)
        assert traj.violating_windows == ()

    def test_returns_theorem_parameters(self) -> None:
        _, _, params = run_rollout(
            candidate_text_with_throttle(0.5), 0, run_seed=0
        )
        assert "signal_values" in params
        assert "threshold" in params
        assert isinstance(params["signal_values"], tuple)

    def test_trajectory_is_dataclass_shape(self) -> None:
        _, traj, _ = run_rollout(
            candidate_text_with_throttle(0.5), 0, run_seed=0
        )
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
