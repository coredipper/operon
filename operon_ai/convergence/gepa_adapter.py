"""GEPA adapter — turns Operon certificates into counterexample-guided feedback.

This module implements a :class:`GEPAAdapter`-shaped class (duck-typed against
``gepa-ai/gepa``) so Operon certificates can drive GEPA's reflective-evolution
optimizer.  No ``gepa`` import is required — the adapter produces the shapes
GEPA consumes, and callers pass an instance into ``gepa.optimize(...)``.

Design (see ``docs/site/external-frameworks.md`` §3.2):

- GEPA expects ``evaluate(batch, candidate, capture_traces) -> EvaluationBatch``
  where ``EvaluationBatch`` has ``outputs``, ``scores``, ``trajectories``.
- Replacing GEPA's usual real-valued ``scores`` with ``{0.0, 1.0}`` from
  :meth:`Certificate.verify` turns the optimizer's reward-gradient search
  into counterexample-guided synthesis (Theorem 3 in the memo).
- ``make_reflective_dataset`` emits per-component feedback where
  ``"Feedback"`` is the formatted obligation text from the certificate's
  verification evidence — so GEPA's reflection LM sees *why* a candidate
  failed, not just that it did.

The adapter never imports ``gepa``; it only produces objects shaped like
``gepa.core.adapter.EvaluationBatch``.  That lets downstream users install
``gepa`` optionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

from ..core.certificate import (
    Certificate,
    CertificateVerification,
    resolve_verify_fn,
)

# ---------------------------------------------------------------------------
# Type variables mirror GEPA's generics so type-checkers can line them up.
# ---------------------------------------------------------------------------

DataInst = TypeVar("DataInst")
Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")


@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """Structurally-compatible mirror of ``gepa.core.adapter.EvaluationBatch``.

    Defined locally so the adapter has no hard dependency on ``gepa``.  When
    ``gepa`` is installed and the adapter is passed into ``gepa.optimize``,
    duck typing handles the bridge — GEPA reads the same attribute names.
    """

    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None
    objective_scores: list[dict[str, float]] | None = None
    num_metric_calls: int | None = None


# ---------------------------------------------------------------------------
# Default obligation formatter
# ---------------------------------------------------------------------------


def default_obligation_formatter(
    verification: CertificateVerification,
    trajectory: Any,  # noqa: ARG001 - part of callback protocol; unused here
) -> str:
    """Render a failed verification as reflection-ready feedback text.

    Format is keyed on the verification evidence dict.  Each evidence key
    becomes a ``- key: value`` line.  A leading summary line names the
    theorem and whether it held.  The ``trajectory`` argument is part of
    the callback protocol for caller overrides — the default formatter
    does not serialize it since GEPA already packs outputs separately.
    """
    _ = trajectory
    cert = verification.certificate
    status = "HOLDS" if verification.holds else "FAILED"
    lines = [
        f"Theorem: {cert.theorem} [{status}]",
        f"Conclusion: {cert.conclusion}",
        "Evidence:",
    ]
    for key, value in verification.evidence.items():
        lines.append(f"  - {key}: {value}")
    if not verification.holds:
        lines.append(
            "Unmet obligation: adjust candidate so the evidence above "
            "satisfies the theorem's predicate."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class OperonCertificateAdapter(Generic[DataInst, Trajectory, RolloutOutput]):
    """GEPA adapter that scores candidates via Operon certificates.

    Parameters
    ----------
    theorem:
        Name of a theorem registered in either ``_VERIFY_REGISTRY`` or
        ``_THEOREM_FN_PATHS``.  Resolved eagerly at construction; raises
        :class:`KeyError` if the theorem is unknown.
    harness:
        ``(candidate, data_inst) -> (output, trajectory, theorem_parameters)``.
        Called once per ``(candidate, data_inst)`` pair during ``evaluate``.
        The returned ``theorem_parameters`` are handed to the certificate's
        verify function via :meth:`Certificate.from_theorem`.
    components:
        The set of candidate component names this adapter is willing to
        reflect on.  Typically the names of mutable prompt slots in the
        caller's program (e.g. ``["planner_prompt", "executor_prompt"]``).
    obligation_formatter:
        Optional override for ``default_obligation_formatter``.  Receives
        the :class:`CertificateVerification` and the trajectory; returns
        the reflection feedback text.
    conclusion_template:
        Optional template for the certificate's ``conclusion`` field.
        ``{theorem}`` is substituted.  Default: ``"{theorem} on GEPA candidate"``.
    source:
        Value used for the certificate's ``source`` field.  Default:
        ``"gepa_adapter"``.
    """

    theorem: str
    harness: Callable[[dict[str, str], Any], tuple[Any, Any, dict[str, Any]]]
    components: Sequence[str] = field(default_factory=tuple)
    obligation_formatter: Callable[[CertificateVerification, Any], str] = (
        default_obligation_formatter
    )
    conclusion_template: str = "{theorem} on GEPA candidate"
    source: str = "gepa_adapter"

    def __post_init__(self) -> None:
        if resolve_verify_fn(self.theorem) is None:
            raise KeyError(
                f"Theorem {self.theorem!r} is not registered. "
                "Register via operon_ai.core.certificate.register_verify_fn "
                "or use a theorem name from the built-in registry."
            )
        # Immutable sequence of component names for reflective dataset keys.
        object.__setattr__(self, "components", tuple(self.components))

    # -- GEPA protocol methods ---------------------------------------------

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Any, Any]:
        """Score a candidate against a batch via certificate verification.

        Each ``data_inst`` in ``batch`` is fed to :attr:`harness`, which
        returns ``(output, trajectory, parameters)``.  A certificate is
        constructed from ``parameters`` and verified; the score is
        ``1.0`` if verification holds, else ``0.0``.
        """
        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[Any] = []
        verifications: list[CertificateVerification] = []

        for data_inst in batch:
            output, trajectory, parameters = self.harness(candidate, data_inst)
            cert = Certificate.from_theorem(
                theorem=self.theorem,
                parameters=parameters,
                conclusion=self.conclusion_template.format(theorem=self.theorem),
                source=self.source,
            )
            verification = cert.verify()
            outputs.append(output)
            scores.append(1.0 if verification.holds else 0.0)
            trajectories.append(trajectory)
            verifications.append(verification)

        # Cache the verifications on the batch so make_reflective_dataset
        # can read them without re-running the harness.
        batch_obj = EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            num_metric_calls=len(batch),
        )
        # Side-channel: attach verifications for the reflective pass.
        # GEPA ignores unknown attributes; this lets reflection access them
        # without plumbing a second return value.
        object.__setattr__(batch_obj, "_operon_verifications", verifications)
        return batch_obj

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Any, Any],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Emit per-component feedback derived from certificate obligations.

        Feedback structure matches GEPA's convention: each component maps
        to a list of records, one per batch item, each with ``"Inputs"``,
        ``"Generated Outputs"``, and ``"Feedback"`` keys.  The
        ``"Feedback"`` string is produced by :attr:`obligation_formatter`.
        """
        verifications: list[CertificateVerification] = getattr(
            eval_batch, "_operon_verifications", []
        )
        trajectories = list(eval_batch.trajectories or [None] * len(eval_batch.outputs))
        outputs = list(eval_batch.outputs)

        if not verifications:
            # evaluate() must have been called before reflection; if an
            # external caller constructed the EvaluationBatch directly,
            # we cannot produce obligation-grounded feedback.
            return {component: [] for component in components_to_update}

        records: list[dict[str, Any]] = []
        for output, trajectory, verification in zip(
            outputs, trajectories, verifications
        ):
            records.append({
                "Inputs": candidate,
                "Generated Outputs": output,
                "Feedback": self.obligation_formatter(verification, trajectory),
            })

        return {component: records for component in components_to_update}
