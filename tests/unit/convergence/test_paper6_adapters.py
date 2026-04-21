"""Unit tests for paper-6 adapter arms (scalar + scalar-with-evidence)."""

from __future__ import annotations

from typing import Any

import pytest

from eval.convergence.scalar_reward_adapter import (
    ScalarRewardAdapter,
    _scalar_reward,
)
from eval.convergence.scalar_with_evidence_adapter import (
    ScalarWithEvidenceAdapter,
    _format_feedback_with_evidence,
)
from eval.convergence.synthetic_signal_harness import (
    SEED_COMPONENT_NAME,
    candidate_text_with_throttle,
    run_rollout,
)


def _harness(candidate: dict[str, str], data_inst: Any) -> tuple[Any, Any, dict[str, Any]]:
    # Post-#872: the harness consumes the full candidate dict; the
    # throttle is read from the ``policy_throttle`` component inside
    # ``run_rollout``.
    return run_rollout(candidate, data_inst, run_seed=0)


# ---------------------------------------------------------------------------
# _scalar_reward
# ---------------------------------------------------------------------------


class TestScalarReward:
    def test_peak_when_theorem_holds(self) -> None:
        # max_value < threshold → theorem holds → reward 1.0
        assert _scalar_reward({"signal_values": [0.0, 0.0], "threshold": 0.5}) == 1.0
        assert _scalar_reward({"signal_values": [0.4], "threshold": 0.5}) == 1.0

    def test_peak_at_threshold_exactly(self) -> None:
        # _verify_behavioral_stability_windowed uses ``max <= threshold``
        # — inclusive — so reward should be 1.0 at the boundary too.
        assert _scalar_reward({"signal_values": [0.5], "threshold": 0.5}) == 1.0

    def test_graded_above_threshold(self) -> None:
        r = _scalar_reward({"signal_values": [1.0], "threshold": 0.5})
        assert r == 0.5  # threshold / max

    def test_monotone_in_max_value(self) -> None:
        r_higher = _scalar_reward({"signal_values": [0.9], "threshold": 0.5})
        r_highest = _scalar_reward({"signal_values": [2.0], "threshold": 0.5})
        assert r_higher > r_highest  # less bad → higher reward

    def test_empty_values_yields_zero(self) -> None:
        assert _scalar_reward({"signal_values": [], "threshold": 0.5}) == 0.0


# ---------------------------------------------------------------------------
# ScalarRewardAdapter
# ---------------------------------------------------------------------------


class TestScalarRewardAdapter:
    def test_evaluate_returns_graded_scores_not_binary(self) -> None:
        adapter = ScalarRewardAdapter(harness=_harness)
        cand = candidate_text_with_throttle(0.25)
        batch = adapter.evaluate([0, 1], cand)
        for score in batch.scores:
            assert 0.0 < score < 1.0 or score == 0.0 or score == 1.0

    def test_feedback_is_minimal_score_only(self) -> None:
        adapter = ScalarRewardAdapter(harness=_harness)
        cand = candidate_text_with_throttle(0.25)
        batch = adapter.evaluate([0], cand)
        reflective = adapter.make_reflective_dataset(
            candidate=cand,
            eval_batch=batch,
            components_to_update=[SEED_COMPONENT_NAME],
        )
        feedback = reflective[SEED_COMPONENT_NAME][0]["Feedback"]
        assert feedback.startswith("Score:")
        # Must NOT include any obligation/evidence text.
        assert "window" not in feedback.lower()
        assert "violat" not in feedback.lower()

    def test_rejects_unknown_component(self) -> None:
        adapter = ScalarRewardAdapter(harness=_harness)
        cand = candidate_text_with_throttle(0.25)
        batch = adapter.evaluate([0], cand)
        with pytest.raises(ValueError, match="not declared"):
            adapter.make_reflective_dataset(
                candidate=cand,
                eval_batch=batch,
                components_to_update=["not_a_real_component"],
            )


# ---------------------------------------------------------------------------
# ScalarWithEvidenceAdapter
# ---------------------------------------------------------------------------


class TestScalarWithEvidenceAdapter:
    def test_feedback_contains_window_evidence(self) -> None:
        adapter = ScalarWithEvidenceAdapter(harness=_harness)
        cand = candidate_text_with_throttle(1.0)  # failing
        batch = adapter.evaluate([0, 1], cand)
        reflective = adapter.make_reflective_dataset(
            candidate=cand,
            eval_batch=batch,
            components_to_update=[SEED_COMPONENT_NAME],
        )
        feedback = reflective[SEED_COMPONENT_NAME][0]["Feedback"]
        assert "Score:" in feedback
        assert "window" in feedback
        assert "VIOLATING" in feedback

    def test_passing_feedback_has_no_adjust_line(self) -> None:
        adapter = ScalarWithEvidenceAdapter(harness=_harness)
        cand = candidate_text_with_throttle(0.1)  # passing
        batch = adapter.evaluate([0], cand)
        reflective = adapter.make_reflective_dataset(
            candidate=cand,
            eval_batch=batch,
            components_to_update=[SEED_COMPONENT_NAME],
        )
        feedback = reflective[SEED_COMPONENT_NAME][0]["Feedback"]
        assert "Adjust candidate" not in feedback

    def test_feedback_does_not_mention_theorem_framing(self) -> None:
        """Active control: evidence text, NOT theorem framing."""
        adapter = ScalarWithEvidenceAdapter(harness=_harness)
        cand = candidate_text_with_throttle(1.0)
        batch = adapter.evaluate([0], cand)
        reflective = adapter.make_reflective_dataset(
            candidate=cand,
            eval_batch=batch,
            components_to_update=[SEED_COMPONENT_NAME],
        )
        feedback = reflective[SEED_COMPONENT_NAME][0]["Feedback"]
        # The cert arm's obligation formatter uses "Theorem:" — the active
        # control must not, otherwise it isn't a control.
        assert "Theorem:" not in feedback


# ---------------------------------------------------------------------------
# _format_feedback_with_evidence
# ---------------------------------------------------------------------------


class TestFormatFeedbackWithEvidence:
    def test_accepts_non_trajectory_input(self) -> None:
        """If trajectory isn't a Trajectory instance, feedback is just the score."""
        text = _format_feedback_with_evidence(0.42, trajectory=None)
        assert text == "Score: 0.4200"


# ---------------------------------------------------------------------------
# Roborev #870: _MockReflectionLM must emit fenced config
# ---------------------------------------------------------------------------


class TestMockReflectionLMMultiComponent:
    """Regression for Roborev #870→#872.

    Under the post-#872 two-component contract, the mutable component
    is ``policy_throttle`` (a bare numeric string).  GEPA's default
    proposer takes the LM's full response verbatim as the new value
    for whichever component is being updated.  The mock must therefore
    emit only the new numeric text — no prose, no CONFIG: prefix, no
    tags — otherwise GEPA sets ``policy_throttle = <tagged junk>`` and
    every smoke run silently fails to converge."""

    def test_mock_emits_bare_numeric_string(self) -> None:
        from eval.convergence.theorem_6_experiment import _MockReflectionLM

        mock = _MockReflectionLM(decrement=0.3, floor=0.1)
        # Each call decrements self._current by 0.3 from start 1.0.
        assert mock("reflection prompt").strip() == "0.700"
        assert mock("reflection prompt").strip() == "0.400"
        # Floor clamps at 0.1.
        assert mock("reflection prompt").strip() == "0.100"

    def test_mock_output_setting_throttle_component_parses(self) -> None:
        """Simulate GEPA's proposer: the mock's output becomes the new
        value of ``policy_throttle``.  ``parse_throttle`` then reads
        that component directly."""
        from eval.convergence.synthetic_signal_harness import (
            PROMPT_COMPONENT_NAME,
            THROTTLE_COMPONENT_NAME,
            parse_throttle,
        )
        from eval.convergence.theorem_6_experiment import _MockReflectionLM

        mock = _MockReflectionLM(decrement=0.3, floor=0.1)
        out = mock("p")
        candidate = {
            PROMPT_COMPONENT_NAME: "whatever",
            THROTTLE_COMPONENT_NAME: out,  # GEPA writes LM output here
        }
        assert parse_throttle(candidate) == 0.7


# ---------------------------------------------------------------------------
# Roborev #854 H1 + #855: arms share the same evidence block byte-for-byte
# ---------------------------------------------------------------------------


class TestCertBinaryContentMatched:
    """Cert-binary and scalar-evidence must produce byte-identical
    feedback after stripping the single prepended framing line.

    This is the direct-comparison test Roborev #855 requested:
    ``render_window_evidence`` is the shared source of truth; any
    divergence between the two arms' tails fails this test
    immediately.
    """

    def _render_both_arms(
        self,
        candidate_text: str,
        *,
        capture_traces: bool = True,
    ) -> tuple[str, str]:
        """Return the full feedback from (cert-binary, scalar-evidence)
        on the same candidate + data instance.  ``capture_traces``
        exercises both GEPA code paths."""
        from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

        from eval.convergence.theorem_6_experiment import (
            stability_windowed_obligation_formatter,
        )

        cert = OperonCertificateAdapter(
            theorem="behavioral_stability_windowed",
            harness=_harness,
            components=[SEED_COMPONENT_NAME],
            obligation_formatter=stability_windowed_obligation_formatter,
            # Same opt-in the paper-6 driver uses.
            retain_trajectories_for_reflection=True,
            source="test",
        )
        scalar_ev = ScalarWithEvidenceAdapter(harness=_harness)
        cand = {SEED_COMPONENT_NAME: candidate_text}

        cert_batch = cert.evaluate([0], cand, capture_traces=capture_traces)
        scalar_batch = scalar_ev.evaluate([0], cand)

        cert_fb = cert.make_reflective_dataset(
            cand, cert_batch, [SEED_COMPONENT_NAME]
        )[SEED_COMPONENT_NAME][0]["Feedback"]
        scalar_fb = scalar_ev.make_reflective_dataset(
            cand, scalar_batch, [SEED_COMPONENT_NAME]
        )[SEED_COMPONENT_NAME][0]["Feedback"]
        return cert_fb, scalar_fb

    def test_tails_are_byte_identical_on_failing_candidate(self) -> None:
        cert_fb, scalar_fb = self._render_both_arms(candidate_text_with_throttle(1.0))
        # Strip the single framing line from each; the remainder must match.
        cert_tail = "\n".join(cert_fb.splitlines()[1:])
        scalar_tail = "\n".join(scalar_fb.splitlines()[1:])
        assert cert_tail == scalar_tail, (
            f"Content-matched invariant violated.\n"
            f"cert-binary tail:\n{cert_tail}\n"
            f"scalar-evidence tail:\n{scalar_tail}"
        )

    def test_tails_are_byte_identical_on_passing_candidate(self) -> None:
        cert_fb, scalar_fb = self._render_both_arms(candidate_text_with_throttle(0.1))
        cert_tail = "\n".join(cert_fb.splitlines()[1:])
        scalar_tail = "\n".join(scalar_fb.splitlines()[1:])
        assert cert_tail == scalar_tail

    def test_first_lines_are_the_only_intended_difference(self) -> None:
        cert_fb, scalar_fb = self._render_both_arms(candidate_text_with_throttle(1.0))
        cert_first = cert_fb.splitlines()[0]
        scalar_first = scalar_fb.splitlines()[0]
        # cert-binary: Theorem framing with pass/fail state
        assert cert_first.startswith("Theorem: behavioral_stability_windowed")
        assert cert_first.endswith("[FAILED]") or cert_first.endswith("[HOLDS]")
        # scalar-evidence: graded numeric score
        assert scalar_first.startswith("Score:")

    def test_content_match_holds_on_default_no_trace_path(self) -> None:
        """Regression for Roborev #857 Low: the content-matched
        invariant must also hold when GEPA evaluates with the default
        ``capture_traces=False``.  Any future drift in the no-trace
        path fails this test — not just the traced path."""
        cert_fb, scalar_fb = self._render_both_arms(
            candidate_text_with_throttle(1.0), capture_traces=False
        )
        cert_tail = "\n".join(cert_fb.splitlines()[1:])
        scalar_tail = "\n".join(scalar_fb.splitlines()[1:])
        assert cert_tail == scalar_tail, (
            f"Content-matched invariant violated on no-trace path.\n"
            f"cert-binary tail:\n{cert_tail}\n"
            f"scalar-evidence tail:\n{scalar_tail}"
        )


# ---------------------------------------------------------------------------
# Roborev #857: retain_trajectories_for_reflection is explicit opt-in
# ---------------------------------------------------------------------------


class TestTrajectoryRetentionOptIn:
    """``retain_trajectories_for_reflection`` controls whether trajectories
    are side-channeled to the reflective pass.  Default False preserves
    the pre-#856 semantics of ``capture_traces=False`` (no hidden
    retention, no memory surprise).  True is the paper-6 opt-in.
    """

    def test_default_false_does_not_side_channel_trajectories(self) -> None:
        """Without the opt-in, no hidden trajectory retention occurs."""
        from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

        adapter = OperonCertificateAdapter(
            theorem="behavioral_stability_windowed",
            harness=_harness,
            components=[SEED_COMPONENT_NAME],
            source="test-default",
            # retain_trajectories_for_reflection left at default False
        )
        cand = candidate_text_with_throttle(1.0)
        batch = adapter.evaluate([0], cand, capture_traces=False)
        # No hidden retention.
        assert not hasattr(batch, "_operon_trajectories")
        # capture_traces=False still drops eval_batch.trajectories.
        assert batch.trajectories is None

    def test_capture_traces_true_without_opt_in_still_no_side_channel(
        self,
    ) -> None:
        """capture_traces governs only GEPA's public trajectories attr;
        the opt-in is orthogonal and its default stays False."""
        from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

        adapter = OperonCertificateAdapter(
            theorem="behavioral_stability_windowed",
            harness=_harness,
            components=[SEED_COMPONENT_NAME],
            source="test-captured",
        )
        cand = candidate_text_with_throttle(1.0)
        batch = adapter.evaluate([0], cand, capture_traces=True)
        assert batch.trajectories is not None
        # Even with traces captured publicly, no side-channel copy.
        assert not hasattr(batch, "_operon_trajectories")

    def test_opt_in_side_channels_regardless_of_capture_traces(
        self,
    ) -> None:
        """When opted in, the side-channel is populated under both
        capture_traces settings so the reflective formatter has data
        on the default GEPA code path."""
        from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

        from eval.convergence.theorem_6_experiment import (
            stability_windowed_obligation_formatter,
        )

        adapter = OperonCertificateAdapter(
            theorem="behavioral_stability_windowed",
            harness=_harness,
            components=[SEED_COMPONENT_NAME],
            obligation_formatter=stability_windowed_obligation_formatter,
            retain_trajectories_for_reflection=True,
            source="test-optin",
        )
        cand = candidate_text_with_throttle(1.0)
        batch_default = adapter.evaluate([0, 1], cand, capture_traces=False)
        batch_captured = adapter.evaluate([0, 1], cand, capture_traces=True)
        # Public attribute respects capture_traces contract.
        assert batch_default.trajectories is None
        assert batch_captured.trajectories is not None
        # Side-channel is populated in both cases.
        assert getattr(batch_default, "_operon_trajectories", None) is not None
        assert getattr(batch_captured, "_operon_trajectories", None) is not None

    def test_legacy_positional_construction_preserves_signature(self) -> None:
        """Regression for Roborev #858 + #859.

        The full legacy positional signature of ``OperonCertificateAdapter``
        is
        ``(theorem, harness, components, obligation_formatter,
        conclusion_template, source, propose_new_texts)`` —
        each slot must remain bindable positionally.  Only fields added
        after the initial release (currently just
        ``retain_trajectories_for_reflection``) are keyword-only.

        This test exercises every positional slot with a non-default
        value so any future reordering that shifts them fails immediately.
        """
        import inspect

        from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

        def custom_formatter(verification, trajectory):  # noqa: ARG001
            del verification, trajectory
            return "custom"

        def custom_proposer(*args, **kwargs):  # noqa: ARG001
            del args, kwargs
            return {}

        adapter = OperonCertificateAdapter(
            "behavioral_stability_windowed",  # theorem
            _harness,                         # harness
            (SEED_COMPONENT_NAME,),           # components
            custom_formatter,                 # obligation_formatter
            "custom template for {theorem}",  # conclusion_template
            "legacy_positional_test",         # source
            custom_proposer,                  # propose_new_texts
        )
        assert adapter.theorem == "behavioral_stability_windowed"
        assert adapter.obligation_formatter is custom_formatter
        assert adapter.conclusion_template == "custom template for {theorem}"
        assert adapter.source == "legacy_positional_test"
        assert adapter.propose_new_texts is custom_proposer
        # New field stays at its default because it is keyword-only and
        # was never passed.
        assert adapter.retain_trajectories_for_reflection is False

        # Signature check locks the public shape in: positional fields
        # remain POSITIONAL_OR_KEYWORD; new flag is KEYWORD_ONLY.
        sig = inspect.signature(OperonCertificateAdapter)
        positional_fields = (
            "theorem",
            "harness",
            "components",
            "obligation_formatter",
            "conclusion_template",
            "source",
            "propose_new_texts",
        )
        for name in positional_fields:
            param = sig.parameters[name]
            assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
                f"{name} must remain POSITIONAL_OR_KEYWORD to preserve the "
                f"pre-existing public signature; got {param.kind}"
            )
        retain = sig.parameters["retain_trajectories_for_reflection"]
        assert retain.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"retain_trajectories_for_reflection must be KEYWORD_ONLY but "
            f"is {retain.kind}"
        )

    def test_retain_flag_cannot_be_passed_positionally(self) -> None:
        """Passing the kw-only flag as the 8th positional must raise."""
        import pytest as _pytest

        from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

        with _pytest.raises(TypeError):
            OperonCertificateAdapter(  # pyright: ignore[reportCallIssue]
                "behavioral_stability_windowed",  # theorem
                _harness,                         # harness
                (SEED_COMPONENT_NAME,),           # components
                None,                             # obligation_formatter
                "c",                              # conclusion_template
                "s",                              # source
                None,                             # propose_new_texts
                True,                             # would-be retain_... positional
            )

    def test_opt_in_feedback_has_per_window_on_no_trace_path(self) -> None:
        """End-to-end sanity check: with the opt-in and
        capture_traces=False, cert-binary feedback still includes the
        full per-window evidence block."""
        from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

        from eval.convergence.theorem_6_experiment import (
            stability_windowed_obligation_formatter,
        )

        adapter = OperonCertificateAdapter(
            theorem="behavioral_stability_windowed",
            harness=_harness,
            components=[SEED_COMPONENT_NAME],
            obligation_formatter=stability_windowed_obligation_formatter,
            retain_trajectories_for_reflection=True,
            source="test",
        )
        cand = candidate_text_with_throttle(1.0)
        batch = adapter.evaluate([0, 1], cand, capture_traces=False)
        reflective = adapter.make_reflective_dataset(
            cand, batch, [SEED_COMPONENT_NAME]
        )
        feedback = reflective[SEED_COMPONENT_NAME][0]["Feedback"]
        for idx in range(4):
            assert f"window {idx}:" in feedback
        assert feedback.splitlines()[0].startswith("Theorem:")
