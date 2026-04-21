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
    # GEPAAdapter protocol: ``propose_new_texts`` is an optional hook
    # that lets the adapter override the default LM-based reflection
    # proposer.  We don't override (we want GEPA's standard reflective
    # proposer), but GEPA dereferences this attribute unconditionally,
    # so it must exist.  ``None`` tells GEPA to fall back to the
    # default proposer.  Kept positional to preserve the pre-existing
    # public signature (Roborev #859).
    propose_new_texts: Any = None
    # Trajectory retention policy for the reflective pass.  When False
    # (default), ``capture_traces`` governs retention exactly as GEPA
    # expects — no hidden retention, no memory/privacy surprise.  Set to
    # True only when the caller's ``obligation_formatter`` needs
    # trajectory content to render full evidence (e.g. the paper-6
    # stability_windowed formatter).  In that case the adapter
    # side-channels trajectories for reflection regardless of
    # ``capture_traces``.  Keyword-only so that inserting the field
    # does not shift any existing positional __init__ arity
    # (Roborev #858).
    retain_trajectories_for_reflection: bool = field(default=False, kw_only=True)

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
        # Always side-channel verifications — they are compact
        # (theorem name, params, holds-bool, evidence dict) and the
        # reflective pass needs them unconditionally to score.
        object.__setattr__(batch_obj, "_operon_verifications", verifications)
        # Only side-channel raw trajectories when the caller explicitly
        # opted in.  Callers who leave ``retain_trajectories_for_reflection``
        # at its default (False) get the pre-#856 behavior: no hidden
        # retention beyond what ``capture_traces`` grants.  Callers whose
        # obligation formatter needs trajectory content (paper-6
        # stability_windowed) opt in explicitly so the retention shows
        # up at the call site (Roborev #857 fix).
        if self.retain_trajectories_for_reflection:
            object.__setattr__(batch_obj, "_operon_trajectories", trajectories)
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

        If :attr:`components` is non-empty, ``components_to_update`` must
        be a subset of it — :class:`ValueError` is raised otherwise.  This
        enforces the configuration contract declared at adapter
        construction and catches typos or unintended component names.  If
        :attr:`components` is empty, any component name is accepted
        (backward-compatible no-allowlist mode).
        """
        if self.components:
            unknown = [c for c in components_to_update if c not in self.components]
            if unknown:
                raise ValueError(
                    "components_to_update contains names not declared on this "
                    f"adapter: {unknown!r}. Declared components: "
                    f"{list(self.components)!r}."
                )

        verifications: list[CertificateVerification] = getattr(
            eval_batch, "_operon_verifications", []
        )
        # Prefer the side-channel trajectories populated by ``evaluate``
        # — they are always present regardless of ``capture_traces``.
        # Fall back to ``eval_batch.trajectories`` (which GEPA may have
        # populated) and finally to a list of None sentinels so the zip
        # below always has something to iterate over.
        sideband_trajectories = getattr(eval_batch, "_operon_trajectories", None)
        if sideband_trajectories is not None:
            trajectories = list(sideband_trajectories)
        else:
            trajectories = list(
                eval_batch.trajectories or [None] * len(eval_batch.outputs)
            )
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
