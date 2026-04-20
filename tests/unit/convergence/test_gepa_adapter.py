"""Tests for the GEPA certificate adapter.

These tests verify the adapter's contract with GEPA's expected interface
(``evaluate`` / ``make_reflective_dataset``) without requiring ``gepa``
to be installed.  The fixtures build tiny harnesses that produce
trajectories whose theorem parameters are known to pass or fail.
"""

from __future__ import annotations

from typing import Any

import pytest

from operon_ai.convergence.gepa_adapter import (
    EvaluationBatch,
    OperonCertificateAdapter,
    default_obligation_formatter,
)
from operon_ai.core.certificate import (
    Certificate,
    register_verify_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _passing_quality_harness(
    candidate: dict[str, str], data_inst: Any
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Harness that emits scores well above the quality threshold."""
    trajectory = {"stage_outputs": [f"handled:{data_inst}"]}
    output = f"out({candidate.get('planner_prompt', '?')}:{data_inst})"
    parameters = {
        "scores": [0.9, 0.95, 0.88],
        "threshold": 0.7,
    }
    return output, trajectory, parameters


def _failing_quality_harness(
    candidate: dict[str, str], data_inst: Any
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Harness that emits scores below the quality threshold."""
    trajectory = {"stage_outputs": [f"handled:{data_inst}"]}
    output = f"out({candidate.get('planner_prompt', '?')}:{data_inst})"
    parameters = {
        "scores": [0.2, 0.3, 0.1],
        "threshold": 0.7,
    }
    return output, trajectory, parameters


# ---------------------------------------------------------------------------
# OperonCertificateAdapter construction
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    """Tests for adapter instantiation and theorem validation."""

    def test_unknown_theorem_raises(self) -> None:
        with pytest.raises(KeyError, match="is not registered"):
            OperonCertificateAdapter(
                theorem="no_such_theorem_exists",
                harness=_passing_quality_harness,
                components=["planner_prompt"],
            )

    def test_registered_theorem_from_static_registry_is_accepted(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_passing_quality_harness,
            components=["planner_prompt"],
        )
        assert adapter.theorem == "behavioral_quality"
        assert adapter.components == ("planner_prompt",)

    def test_dynamic_registry_theorems_are_accepted(self) -> None:
        """Downstream-registered theorems should be usable by the adapter."""
        def _always_true(_params):  # noqa: ARG001 - verify-fn protocol signature
            del _params
            return True, {"reason": "dynamic-test"}

        register_verify_fn("dynamic_test_theorem_adapter", _always_true)
        adapter = OperonCertificateAdapter(
            theorem="dynamic_test_theorem_adapter",
            harness=_passing_quality_harness,
            components=["a"],
        )
        assert adapter.theorem == "dynamic_test_theorem_adapter"


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for OperonCertificateAdapter.evaluate."""

    def test_passing_harness_yields_score_one(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_passing_quality_harness,
            components=["planner_prompt"],
        )
        batch = [1, 2, 3]
        result = adapter.evaluate(batch=batch, candidate={"planner_prompt": "x"})
        assert isinstance(result, EvaluationBatch)
        assert result.scores == [1.0, 1.0, 1.0]
        assert len(result.outputs) == 3

    def test_failing_harness_yields_score_zero(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_failing_quality_harness,
            components=["planner_prompt"],
        )
        batch = [1, 2]
        result = adapter.evaluate(batch=batch, candidate={"planner_prompt": "x"})
        assert result.scores == [0.0, 0.0]

    def test_capture_traces_true_populates_trajectories(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_passing_quality_harness,
            components=["planner_prompt"],
        )
        result = adapter.evaluate(
            batch=[1, 2],
            candidate={"planner_prompt": "x"},
            capture_traces=True,
        )
        assert result.trajectories is not None
        assert len(result.trajectories) == 2

    def test_capture_traces_false_trajectories_none(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_passing_quality_harness,
            components=["planner_prompt"],
        )
        result = adapter.evaluate(
            batch=[1, 2],
            candidate={"planner_prompt": "x"},
            capture_traces=False,
        )
        assert result.trajectories is None

    def test_num_metric_calls_reflects_batch_size(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_passing_quality_harness,
            components=["planner_prompt"],
        )
        result = adapter.evaluate(
            batch=[1, 2, 3, 4, 5],
            candidate={"planner_prompt": "x"},
        )
        assert result.num_metric_calls == 5


# ---------------------------------------------------------------------------
# make_reflective_dataset()
# ---------------------------------------------------------------------------


class TestReflectiveDataset:
    """Tests for OperonCertificateAdapter.make_reflective_dataset."""

    def test_feedback_contains_obligation_text_on_failure(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_failing_quality_harness,
            components=["planner_prompt"],
        )
        candidate = {"planner_prompt": "planner"}
        batch = [1, 2]
        eval_batch = adapter.evaluate(batch, candidate, capture_traces=True)
        reflective = adapter.make_reflective_dataset(
            candidate=candidate,
            eval_batch=eval_batch,
            components_to_update=["planner_prompt"],
        )
        assert "planner_prompt" in reflective
        records = reflective["planner_prompt"]
        assert len(records) == 2
        # Feedback must flag the failure and include an obligation marker.
        for record in records:
            feedback = record["Feedback"]
            assert "Theorem: behavioral_quality" in feedback
            assert "FAILED" in feedback
            assert "Unmet obligation" in feedback

    def test_feedback_records_holds_status_on_success(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_passing_quality_harness,
            components=["planner_prompt"],
        )
        candidate = {"planner_prompt": "planner"}
        eval_batch = adapter.evaluate([1, 2], candidate, capture_traces=True)
        reflective = adapter.make_reflective_dataset(
            candidate=candidate,
            eval_batch=eval_batch,
            components_to_update=["planner_prompt"],
        )
        for record in reflective["planner_prompt"]:
            assert "HOLDS" in record["Feedback"]
            assert "Unmet obligation" not in record["Feedback"]

    def test_multiple_components_each_receive_records(self) -> None:
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_failing_quality_harness,
            components=["a", "b"],
        )
        candidate = {"a": "aa", "b": "bb"}
        eval_batch = adapter.evaluate([1], candidate, capture_traces=True)
        reflective = adapter.make_reflective_dataset(
            candidate=candidate,
            eval_batch=eval_batch,
            components_to_update=["a", "b"],
        )
        assert set(reflective.keys()) == {"a", "b"}
        assert len(reflective["a"]) == 1
        assert len(reflective["b"]) == 1

    def test_bypass_evaluate_yields_empty_feedback(self) -> None:
        """If a caller constructs an EvaluationBatch without going through
        evaluate(), the verifications side-channel is absent and the
        reflective dataset is empty."""
        adapter = OperonCertificateAdapter(
            theorem="behavioral_quality",
            harness=_passing_quality_harness,
            components=["planner_prompt"],
        )
        foreign_batch = EvaluationBatch(
            outputs=["x"], scores=[1.0], trajectories=["t"]
        )
        reflective = adapter.make_reflective_dataset(
            candidate={"planner_prompt": "x"},
            eval_batch=foreign_batch,
            components_to_update=["planner_prompt"],
        )
        assert reflective == {"planner_prompt": []}


# ---------------------------------------------------------------------------
# default_obligation_formatter
# ---------------------------------------------------------------------------


class TestDefaultObligationFormatter:
    """Tests for the default obligation-formatter helper."""

    def test_formats_failed_verification(self) -> None:
        cert = Certificate.from_theorem(
            theorem="behavioral_quality",
            parameters={"scores": [0.1, 0.2], "threshold": 0.9},
            conclusion="test",
            source="test",
        )
        verification = cert.verify()
        text = default_obligation_formatter(verification, trajectory=None)
        assert "FAILED" in text
        assert "Unmet obligation" in text
        assert "mean" in text  # evidence key surfaces

    def test_formats_held_verification(self) -> None:
        cert = Certificate.from_theorem(
            theorem="behavioral_quality",
            parameters={"scores": [0.9, 0.95], "threshold": 0.5},
            conclusion="test",
            source="test",
        )
        verification = cert.verify()
        text = default_obligation_formatter(verification, trajectory=None)
        assert "HOLDS" in text
        assert "Unmet obligation" not in text
