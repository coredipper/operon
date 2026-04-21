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


def _fenced(body: str, lang: str = "config") -> str:
    """Wrap ``body`` in a markdown-style fenced code block.

    Concentrates the fence boilerplate in one place so tests read as
    the *scenario* they describe rather than as formatting concerns.
    """
    return f"```{lang}\n{body}\n```"


class TestParseThrottleFenced:
    """The parser's contract (Roborev #869): config MUST live inside a
    fenced code block.  Assignments outside a fence are prose and
    ignored."""

    def test_parses_equals_form(self) -> None:
        assert parse_throttle(_fenced("policy_throttle = 0.3")) == 0.3

    def test_parses_colon_form(self) -> None:
        assert parse_throttle(_fenced("policy_throttle: 0.45")) == 0.45

    def test_case_insensitive(self) -> None:
        assert parse_throttle(_fenced("POLICY_THROTTLE = 0.2")) == 0.2

    def test_clips_above_one(self) -> None:
        assert parse_throttle(_fenced("policy_throttle = 5")) == 1.0

    def test_clips_below_zero(self) -> None:
        """Negative values clip to 0 — best-case throttle semantics."""
        assert parse_throttle(_fenced("policy_throttle = -0.5")) == 0.0

    def test_indented_marker(self) -> None:
        assert parse_throttle(_fenced("  policy_throttle = 0.7")) == 0.7

    def test_tilde_fences_also_accepted(self) -> None:
        """``~~~`` is the alternative markdown fence; LMs may emit either."""
        assert parse_throttle("~~~config\npolicy_throttle = 0.4\n~~~") == 0.4

    def test_last_assignment_in_fence_wins(self) -> None:
        """Regression for Roborev #864 H: append-to-revise inside fence."""
        text = _fenced(
            "policy_throttle = 1.0\n# refined:\npolicy_throttle = 0.3"
        )
        assert parse_throttle(text) == 0.3

    def test_multi_fence_last_fence_last_assignment_wins(self) -> None:
        """If the LM emits multiple fences, the overall last assignment
        across their concatenated scopes wins."""
        text = (
            _fenced("policy_throttle = 0.9", lang="")
            + "\n\nExplanation text.\n\n"
            + _fenced("policy_throttle = 0.2")
        )
        assert parse_throttle(text) == 0.2


class TestParseThrottleMalformedFinalAssignment:
    """Regression for Roborev #867 — malformed FINAL assignment must
    default, never fall through to an earlier valid one."""

    def test_non_numeric_last(self) -> None:
        text = _fenced("policy_throttle = 0.3\npolicy_throttle = nope")
        assert parse_throttle(text) == 1.0

    def test_empty_rhs_last(self) -> None:
        text = _fenced("policy_throttle = 0.3\npolicy_throttle =")
        assert parse_throttle(text) == 1.0

    def test_just_dash_last(self) -> None:
        text = _fenced("policy_throttle = 0.3\npolicy_throttle = -")
        assert parse_throttle(text) == 1.0

    def test_valid_last_overrides_invalid_earlier(self) -> None:
        text = _fenced("policy_throttle = nope\npolicy_throttle = 0.2")
        assert parse_throttle(text) == 0.2


class TestParseThrottleOutsideFenceIgnored:
    """Regression for Roborev #868 and #869 — assignments outside a
    fenced code block are prose and must never override the real
    fenced config."""

    def test_no_fence_returns_default(self) -> None:
        """An LM that drops the fence produces a malformed prompt;
        parse_throttle returns _DEFAULT_THROTTLE so reflection feedback
        signals the LM to restore the config fence."""
        assert parse_throttle("policy_throttle = 0.3") == 1.0

    def test_quoted_example_on_own_line_after_fence(self) -> None:
        """Roborev #869: a line-start ``policy_throttle = 1.0`` that
        follows the real config as an "example" must NOT override the
        real value."""
        text = _fenced("policy_throttle = 0.2") + (
            "\n\nExample of prior output:\npolicy_throttle = 1.0"
        )
        assert parse_throttle(text) == 0.2

    def test_prose_mid_line_mention_ignored(self) -> None:
        """Roborev #868: mid-line prose mentions never counted as real
        assignments.  Kept here to guard the invariant now that it
        applies outside the fence too."""
        text = _fenced("policy_throttle = 0.2") + (
            "\nFor reference: previous bad output said policy_throttle = nope"
        )
        assert parse_throttle(text) == 0.2

    def test_quoted_parenthetical_example(self) -> None:
        text = _fenced("policy_throttle = 0.3") + (
            "\n(As a reminder, initial was policy_throttle = 1.0.)"
        )
        assert parse_throttle(text) == 0.3

    def test_full_prose_paragraph_after_fence(self) -> None:
        """An LM that writes a whole paragraph of reasoning after the
        fence — containing multiple line-start throttle mentions — must
        still read the fenced value.  Regression for the spirit of #869."""
        text = _fenced("policy_throttle = 0.25") + (
            "\n\nReasoning:\n"
            "policy_throttle = 1.0  was the seed\n"
            "policy_throttle = 0.7  was my previous attempt\n"
            "so I have now set it to 0.25 above."
        )
        assert parse_throttle(text) == 0.25


class TestParseThrottleMissingMarker:
    """Missing / empty / irrelevant text defaults safely."""

    def test_empty_returns_default(self) -> None:
        assert parse_throttle("") == 1.0

    def test_missing_marker_returns_default(self) -> None:
        assert parse_throttle("no marker here") == 1.0

    def test_fence_without_assignment_returns_default(self) -> None:
        assert parse_throttle(_fenced("# nothing to see here")) == 1.0

    def test_seed_prompt_parses_as_failing(self) -> None:
        """Sanity: the built-in seed prompt wraps its assignment in a
        fence and the parser reads it as the starting 1.0 value."""
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
