"""Unit tests for the paper-6 synthetic signal harness."""

from __future__ import annotations

import pytest

from eval.convergence.synthetic_signal_harness import (
    CANDIDATE_COMPONENTS,
    PROMPT_COMPONENT_NAME,
    SEED_CANDIDATE,
    SEED_COMPONENT_NAME,
    THROTTLE_COMPONENT_NAME,
    SyntheticDataset,
    TaskConfig,
    Trajectory,
    candidate_dict_with_throttle,
    parse_throttle,
    run_rollout,
    seed_candidate,
)


# ---------------------------------------------------------------------------
# parse_throttle (post-#872 contract: read the numeric component directly)
# ---------------------------------------------------------------------------


class TestParseThrottleNumericComponent:
    """The parser reads ``candidate["policy_throttle"]`` and does
    exactly one thing with it: ``float()`` + clip to [0, 1].  No
    regex, no line-start anchoring, no prose inspection."""

    def test_parses_valid_number(self) -> None:
        assert parse_throttle({
            PROMPT_COMPONENT_NAME: "ignored",
            THROTTLE_COMPONENT_NAME: "0.3",
        }) == 0.3

    def test_clips_above_one(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "5"}) == 1.0

    def test_clips_below_zero(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "-0.5"}) == 0.0

    def test_accepts_whitespace_around_value(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "  0.4\n"}) == 0.4

    def test_integer_string_parses(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "1"}) == 1.0

    def test_scientific_notation_parses(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "2.5e-1"}) == 0.25


class TestParseThrottleMalformedComponent:
    """Malformed or missing numeric component → ``_DEFAULT_THROTTLE``."""

    def test_missing_component_returns_default(self) -> None:
        assert parse_throttle({PROMPT_COMPONENT_NAME: "only prose"}) == 1.0

    def test_empty_component_returns_default(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: ""}) == 1.0

    def test_whitespace_only_component_returns_default(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "   \n\t"}) == 1.0

    def test_non_numeric_string_returns_default(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "nope"}) == 1.0

    def test_just_dash_returns_default(self) -> None:
        assert parse_throttle({THROTTLE_COMPONENT_NAME: "-"}) == 1.0

    def test_non_string_value_returns_default(self) -> None:
        """Guard against accidental non-string values in the component
        (shouldn't happen under GEPA's dict[str, str] contract, but
        harden just in case)."""
        assert parse_throttle({THROTTLE_COMPONENT_NAME: None}) == 1.0  # type: ignore[dict-item]


class TestParseThrottleSpoofingClassEliminated:
    """Regression for the entire Roborev #864–#872 series.

    The defining feature of the root-cause fix is that the throttle
    lives in its own component — the prose prompt is NEVER parsed.
    Therefore no amount of malicious / confusing prose in the prompt
    component can affect the parsed throttle.  These tests construct
    the worst-case prose payloads from the prior review series and
    assert the throttle reads correctly regardless."""

    def test_prose_with_embedded_assignments_is_ignored(self) -> None:
        """Prose full of plausible assignments in every format the
        previous pattern-based parsers tried to handle — line-start,
        mid-line, fenced, CONFIG:-prefixed — has zero effect on the
        parsed throttle.  The numeric component is authoritative."""
        malicious_prose = (
            "policy_throttle = 1.0\n"
            "CONFIG: policy_throttle = 0.99\n"
            "```config\npolicy_throttle = 0.95\n```\n"
            "Example of prior output:\n"
            "CONFIG: policy_throttle = 0.9\n"
            "And a sneaky quoted example: CONFIG: policy_throttle = 0.85\n"
        )
        assert (
            parse_throttle({
                PROMPT_COMPONENT_NAME: malicious_prose,
                THROTTLE_COMPONENT_NAME: "0.3",
            })
            == 0.3
        )

    def test_empty_prose_with_valid_throttle_parses(self) -> None:
        """The prose component can be empty; only the numeric matters."""
        assert parse_throttle({
            PROMPT_COMPONENT_NAME: "",
            THROTTLE_COMPONENT_NAME: "0.25",
        }) == 0.25

    def test_prose_containing_only_throttle_text_is_still_ignored(
        self,
    ) -> None:
        """Worst case: the prose contains exactly the numeric string
        that the PARSED throttle happens to have.  Even then, it is
        the numeric component that's read, not the prose."""
        assert parse_throttle({
            PROMPT_COMPONENT_NAME: "0.9",  # looks numeric but is prose
            THROTTLE_COMPONENT_NAME: "0.3",
        }) == 0.3


class TestParseThrottleRejectsLegacyStringInput:
    """Guard against legacy callers that still pass a prose string."""

    def test_passing_str_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="candidate dict"):
            parse_throttle("CONFIG: policy_throttle = 0.3")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Candidate shape / helpers
# ---------------------------------------------------------------------------


class TestSeedCandidate:
    """Seed candidate is a two-component dict starting at throttle 1.0."""

    def test_contains_both_components(self) -> None:
        c = seed_candidate()
        assert PROMPT_COMPONENT_NAME in c
        assert THROTTLE_COMPONENT_NAME in c

    def test_throttle_starts_at_default(self) -> None:
        c = seed_candidate()
        assert parse_throttle(c) == 1.0

    def test_seed_candidate_returns_fresh_copy(self) -> None:
        """Mutating the returned dict must not affect SEED_CANDIDATE."""
        c = seed_candidate()
        c[THROTTLE_COMPONENT_NAME] = "0.0"
        assert SEED_CANDIDATE[THROTTLE_COMPONENT_NAME] == "1.0"

    def test_candidate_components_exposes_canonical_names(self) -> None:
        assert PROMPT_COMPONENT_NAME in CANDIDATE_COMPONENTS
        assert THROTTLE_COMPONENT_NAME in CANDIDATE_COMPONENTS

    def test_seed_component_name_is_backward_compat_alias(self) -> None:
        """Pre-#872 callers imported ``SEED_COMPONENT_NAME`` as the
        name of the one mutable component.  Under the new contract
        it points at the prose component for compatibility."""
        assert SEED_COMPONENT_NAME == PROMPT_COMPONENT_NAME


class TestCandidateDictBuilder:
    """The public ``candidate_dict_with_throttle`` helper builds
    well-formed candidates for tests and synthetic callers."""

    def test_builder_produces_parseable_candidate(self) -> None:
        for v in (0.0, 0.1, 0.25, 0.5, 1.0):
            c = candidate_dict_with_throttle(v)
            assert parse_throttle(c) == v

    def test_builder_clips_out_of_range(self) -> None:
        assert parse_throttle(candidate_dict_with_throttle(5)) == 1.0
        assert parse_throttle(candidate_dict_with_throttle(-0.5)) == 0.0

    def test_builder_returns_both_components(self) -> None:
        c = candidate_dict_with_throttle(0.3)
        assert set(c.keys()) == {PROMPT_COMPONENT_NAME, THROTTLE_COMPONENT_NAME}

    def test_text_builder_alias_is_removed(self) -> None:
        """The pre-#872 ``candidate_text_with_throttle`` name is
        intentionally NOT re-exported (Roborev #873): it would return
        a dict despite its ``_text`` suffix, which misled the one
        in-repo caller that wrapped the result under
        SEED_COMPONENT_NAME."""
        import eval.convergence.synthetic_signal_harness as harness_module

        assert not hasattr(harness_module, "candidate_text_with_throttle")


# ---------------------------------------------------------------------------
# run_rollout
# ---------------------------------------------------------------------------


class TestRunRollout:
    """Tests for run_rollout — determinism and theorem-parameter shape."""

    def test_is_deterministic_under_fixed_seed(self) -> None:
        cand = candidate_dict_with_throttle(0.5)
        out1 = run_rollout(cand, 7, run_seed=42)
        out2 = run_rollout(cand, 7, run_seed=42)
        assert out1[1].window_means == out2[1].window_means

    def test_different_data_inst_ids_produce_different_trajectories(self) -> None:
        cand = candidate_dict_with_throttle(0.5)
        _, t1, _ = run_rollout(cand, 0, run_seed=42)
        _, t2, _ = run_rollout(cand, 1, run_seed=42)
        assert t1.window_means != t2.window_means

    def test_different_run_seeds_produce_different_trajectories(self) -> None:
        cand = candidate_dict_with_throttle(0.5)
        _, t1, _ = run_rollout(cand, 0, run_seed=1)
        _, t2, _ = run_rollout(cand, 0, run_seed=2)
        assert t1.window_means != t2.window_means

    def test_high_throttle_violates_threshold(self) -> None:
        cfg = TaskConfig()
        _, traj, _ = run_rollout(seed_candidate(), 0, config=cfg, run_seed=0)
        assert all(m > cfg.threshold for m in traj.window_means)
        assert traj.violating_windows == tuple(
            range(cfg.windows_per_rollout)
        )

    def test_low_throttle_passes_threshold(self) -> None:
        cfg = TaskConfig()
        _, traj, _ = run_rollout(
            candidate_dict_with_throttle(0.1), 0, config=cfg, run_seed=0
        )
        assert all(m <= cfg.threshold for m in traj.window_means)
        assert traj.violating_windows == ()

    def test_returns_theorem_parameters(self) -> None:
        _, _, params = run_rollout(
            candidate_dict_with_throttle(0.5), 0, run_seed=0
        )
        assert "signal_values" in params
        assert "threshold" in params
        assert isinstance(params["signal_values"], tuple)

    def test_trajectory_is_dataclass_shape(self) -> None:
        _, traj, _ = run_rollout(
            candidate_dict_with_throttle(0.5), 0, run_seed=0
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
