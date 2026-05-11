"""Helpfulness/Harmfulness metric for the ``reviewer_gate`` pattern.

Implements the population-level tradeoff metric introduced by Ta et al.,
*Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents*
(arXiv:2604.27233):

- **Helpfulness** — fraction of base-agent errors that the reviewer's
  feedback corrects.
- **Harmfulness** — fraction of correct base-agent responses that the
  reviewer's feedback degrades.

A reviewer is **net positive** if ``helpfulness > harmfulness``.
The benefit-to-risk ratio (``helpfulness / harmfulness``) is the
ratio reported in the paper (3:1 for o3-mini, 2.1:1 for GPT-4o on
BFCL irrelevance detection).

This is an eval utility, not a Certificate theorem. Use it to compare
two trajectory populations — one without the reviewer (base) and one
with — over a fixed task set with ground-truth correctness labels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryOutcome:
    """One task outcome from a single run (base or reviewed).

    ``correct`` is the ground-truth label for the trajectory's final
    response. ``intervened`` indicates whether the reviewer modified the
    trajectory relative to what the base agent would have produced; on
    the base run it is always ``False`` by definition.
    """

    task_id: str
    correct: bool
    intervened: bool = False


@dataclass(frozen=True)
class HelpfulnessHarmfulnessMetric:
    """Result of comparing a reviewed run against a base run.

    Helpfulness and harmfulness are both in ``[0, 1]``. ``benefit_risk_ratio``
    is ``helpfulness / harmfulness``; ``None`` when ``harmfulness == 0`` (no
    risk denominator). Count fields surface the underlying tallies for audit
    and downstream aggregation.
    """

    helpfulness: float
    harmfulness: float
    benefit_risk_ratio: float | None
    n_base_errors: int
    n_base_correct: int
    n_corrected: int
    n_degraded: int


def compute_hh(
    base: list[TrajectoryOutcome],
    reviewed: list[TrajectoryOutcome],
) -> HelpfulnessHarmfulnessMetric:
    """Compute the Helpfulness/Harmfulness metric over paired runs.

    ``base`` and ``reviewed`` must cover the same set of ``task_id``s;
    each ``task_id`` appearing exactly once in each list. The function
    pairs outcomes by ``task_id`` (not list position) so callers can
    pass lists in any order.

    Args:
        base: Outcomes from running the agent without the reviewer.
        reviewed: Outcomes from running the agent with the reviewer.

    Returns:
        A :class:`HelpfulnessHarmfulnessMetric` summarising the pair.

    Raises:
        ValueError: if the two lists do not share the same ``task_id`` set.
    """
    base_by_id = {o.task_id: o for o in base}
    reviewed_by_id = {o.task_id: o for o in reviewed}

    if base_by_id.keys() != reviewed_by_id.keys():
        only_base = base_by_id.keys() - reviewed_by_id.keys()
        only_reviewed = reviewed_by_id.keys() - base_by_id.keys()
        raise ValueError(
            "base and reviewed must cover the same task_ids; "
            f"missing in reviewed: {sorted(only_base)!r}; "
            f"missing in base: {sorted(only_reviewed)!r}"
        )

    n_base_errors = sum(1 for o in base_by_id.values() if not o.correct)
    n_base_correct = sum(1 for o in base_by_id.values() if o.correct)

    n_corrected = sum(
        1
        for tid, b in base_by_id.items()
        if not b.correct and reviewed_by_id[tid].correct
    )
    n_degraded = sum(
        1
        for tid, b in base_by_id.items()
        if b.correct and not reviewed_by_id[tid].correct
    )

    helpfulness = n_corrected / n_base_errors if n_base_errors > 0 else 0.0
    harmfulness = n_degraded / n_base_correct if n_base_correct > 0 else 0.0
    benefit_risk_ratio = helpfulness / harmfulness if harmfulness > 0 else None

    return HelpfulnessHarmfulnessMetric(
        helpfulness=helpfulness,
        harmfulness=harmfulness,
        benefit_risk_ratio=benefit_risk_ratio,
        n_base_errors=n_base_errors,
        n_base_correct=n_base_correct,
        n_corrected=n_corrected,
        n_degraded=n_degraded,
    )


def is_net_positive(metric: HelpfulnessHarmfulnessMetric) -> bool:
    """Whether the reviewer's helpfulness exceeds its harmfulness.

    A net-positive reviewer corrects more base-agent errors than it
    degrades correct responses. The decision threshold is strict
    inequality — a reviewer that is exactly break-even is *not*
    counted as net positive.
    """
    return metric.helpfulness > metric.harmfulness
