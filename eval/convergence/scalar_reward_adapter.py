"""Scalar-reward GEPA adapter — paper 6 baseline arm.

The conjecture being tested (Theorem 3 in the external-frameworks memo)
is that replacing a scalar reward with a certificate-based binary
evaluator + obligation text improves GEPA's convergence rate on tasks
with enumerable failure modes.

This adapter is the naive-reward baseline.  It reports a graded score
and *no structured obligation text* — reflection feedback is just the
raw numeric score, which is the form GEPA was originally designed to
consume.

Design notes
------------
- No ``gepa`` import; ``EvaluationBatch`` is reused from
  ``operon_ai.convergence.gepa_adapter`` where it's already defined as
  a structural mirror.
- The reward formula is ``max(0, (τ - max(window_means)) / τ)``, which
  is monotone in the knob being tuned (``policy_throttle``) and goes
  to zero at the threshold.  The choice is intentionally simple — a
  more clever reward would be a confound for the experiment.
- Reflection text is the most minimal form GEPA will accept: the score
  as a single sentence.  This is what a user writing a "toy" GEPA
  evaluator would produce without domain engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from operon_ai.convergence.gepa_adapter import EvaluationBatch

from .synthetic_signal_harness import SEED_COMPONENT_NAME


@dataclass
class ScalarRewardAdapter:
    """GEPA adapter producing a graded scalar reward and minimal feedback.

    Parameters
    ----------
    harness:
        ``(candidate, data_inst) -> (output, trajectory, parameters)``.
        The ``parameters`` dict must contain ``signal_values`` (iterable
        of per-window means) and ``threshold`` keys so this adapter can
        read the same underlying evidence the certificate arm sees.
    components:
        Mutable component names in the candidate mapping.  Defaults to
        the single seed component used by the paper-6 harness.
    """

    harness: Callable[[dict[str, str], Any], tuple[Any, Any, dict[str, Any]]]
    components: Sequence[str] = field(
        default_factory=lambda: (SEED_COMPONENT_NAME,)
    )
    # GEPAAdapter protocol hook — must exist even when unused.  See
    # ``OperonCertificateAdapter.propose_new_texts`` for rationale.
    propose_new_texts: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))

    # -- GEPA protocol ------------------------------------------------------

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Any, Any]:
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
            trajectories=trajectories if capture_traces else None,
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

        records: list[dict[str, Any]] = []
        for output, score in zip(outputs, scores):
            records.append({
                "Inputs": candidate,
                "Generated Outputs": output,
                "Feedback": f"Score: {score:.4f}",
            })

        return {component: records for component in components_to_update}


# ---------------------------------------------------------------------------
# Reward formula
# ---------------------------------------------------------------------------


def _scalar_reward(parameters: Mapping[str, Any]) -> float:
    """Graded reward that peaks at 1.0 exactly when the theorem holds.

    - ``max(values) <= threshold`` → reward = 1.0 (theorem holds)
    - ``max(values) > threshold``  → reward = ``threshold / max(values)``,
      strictly in (0, 1) and monotonically decreasing in ``max(values)``.

    The 1.0-at-holding convention matches the cert-binary arm's binary
    score: both arms converge when ``verify().holds is True``, so the
    cross-arm convergence detector can use the same threshold.  The
    partial-credit formula for the failing regime still gives the
    reward-gradient arm a usable directional signal — an LM that
    reduces the throttle will see reward rise monotonically.
    """
    values = list(parameters.get("signal_values", ()))
    threshold = float(parameters.get("threshold", 1.0))
    if not values or threshold <= 0:
        return 0.0
    max_value = max(values)
    if max_value <= threshold:
        return 1.0
    return max(0.0, min(1.0, threshold / max_value))
