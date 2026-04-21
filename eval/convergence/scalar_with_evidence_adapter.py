"""Scalar-with-evidence GEPA adapter — paper 6 active control arm.

This arm exists to separate two confounds:

    (a) the certificate arm's *binarization* of the score, and
    (b) the certificate arm's *obligation text* (window k violated by m).

The certificate-binary arm carries both.  The scalar-reward arm carries
neither.  This arm carries (b) only — it reports the same graded scalar
reward as :class:`ScalarRewardAdapter`, but its reflection feedback
includes the violating-window description that the cert arm's
obligation formatter would have produced.

If cert-binary beats scalar-reward but ties with scalar-with-evidence,
the conjecture's explanatory force is in the evidence text, not in
binarization.  If cert-binary beats both, binarization adds signal
(which is the conjecture).  If cert-binary loses to one of the
scalar arms, the conjecture is refuted.

This adapter is deliberately NOT a subclass of
:class:`ScalarRewardAdapter` — composition-over-inheritance keeps the
arm definitions independently auditable, and the diff between the two
``make_reflective_dataset`` bodies is the load-bearing comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from operon_ai.convergence.gepa_adapter import EvaluationBatch

from .scalar_reward_adapter import _scalar_reward
from .synthetic_signal_harness import (
    SEED_COMPONENT_NAME,
    Trajectory,
    render_window_evidence,
)


@dataclass
class ScalarWithEvidenceAdapter:
    """GEPA adapter: graded scalar reward + text-matched obligation evidence.

    ``harness`` must return trajectories whose ``violating_windows``
    and ``window_means`` attributes can be read directly (any object
    duck-typed to :class:`Trajectory`).  The evidence text is rendered
    from those fields into the form the cert arm's obligation
    formatter produces — matched token-for-token on the window index
    and mean value, minus the "Theorem: ..." framing.
    """

    harness: Callable[[dict[str, str], Any], tuple[Any, Any, dict[str, Any]]]
    components: Sequence[str] = field(
        default_factory=lambda: (SEED_COMPONENT_NAME,)
    )
    propose_new_texts: Any = None  # GEPAAdapter protocol hook

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))

    # -- GEPA protocol ------------------------------------------------------

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,  # noqa: ARG002 - part of protocol; always-captured here
    ) -> EvaluationBatch[Any, Any]:
        del capture_traces  # evidence text always needs trajectories
        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[Any] = []

        for data_inst in batch:
            output, trajectory, parameters = self.harness(candidate, data_inst)
            score = _scalar_reward(parameters)
            outputs.append(output)
            scores.append(score)
            trajectories.append(trajectory)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Any, Any],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        unknown = [c for c in components_to_update if c not in self.components]
        if unknown:
            raise ValueError(
                "components_to_update contains names not declared on this "
                f"adapter: {unknown!r}"
            )

        outputs = list(eval_batch.outputs)
        scores = list(eval_batch.scores)
        trajectories = list(eval_batch.trajectories or [None] * len(outputs))

        records: list[dict[str, Any]] = []
        for output, score, trajectory in zip(outputs, scores, trajectories):
            records.append({
                "Inputs": candidate,
                "Generated Outputs": output,
                "Feedback": _format_feedback_with_evidence(score, trajectory),
            })

        return {component: records for component in components_to_update}


# ---------------------------------------------------------------------------
# Evidence-matched feedback formatter
# ---------------------------------------------------------------------------


def _format_feedback_with_evidence(score: float, trajectory: Any) -> str:
    """Render ``Score: <value>`` + shared per-window evidence block.

    Uses :func:`render_window_evidence` — the single source of truth
    shared with cert-binary's formatter.  Per Roborev #855, the two arms
    differ only in the prepended line (``Score:`` here,
    ``Theorem: ... [state]`` in cert-binary); everything after that line
    is byte-identical, and a direct-comparison test guards the invariant.
    """
    header = f"Score: {score:.4f}"
    if not isinstance(trajectory, Trajectory):
        return header
    return header + "\n" + render_window_evidence(trajectory)
