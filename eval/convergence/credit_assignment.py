"""Credit assignment for the C6 convergence evaluation harness.

Distributes credit across stages in a multi-agent run, tracking which
roles contributed most and which required interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageCredit:
    """Credit attributed to a single stage in a run."""

    stage_name: str
    role: str
    contribution: float
    was_intervention_target: bool


def assign_credit(
    roles: tuple[str, ...],
    success: bool,
    risk_score: float,
    rng: Any,
) -> list[StageCredit]:
    """Assign credit to each role in a run.

    For successful runs, credit is distributed proportionally with some
    randomness.  For failed runs, negative credit is assigned, weighted
    toward higher-risk stages.

    Parameters
    ----------
    roles:
        Role names for each stage.
    success:
        Whether the run succeeded.
    risk_score:
        Composite risk score from the adapter analysis.
    rng:
        A random.Random instance for reproducibility.

    Returns
    -------
    list[StageCredit]
        One credit entry per role.
    """
    n = len(roles)
    if n == 0:
        return []

    # Generate raw weights.
    raw = [rng.random() for _ in range(n)]
    total = sum(raw)
    if total == 0:
        total = 1.0

    # Base contribution: positive for success, negative for failure.
    base_sign = 1.0 if success else -1.0

    credits: list[StageCredit] = []
    for i, role in enumerate(roles):
        weight = raw[i] / total
        contribution = base_sign * weight

        # Intervention target: higher probability for higher risk stages.
        # The last stages in a pipeline are more likely to be intervention targets.
        intervention_prob = risk_score * (i + 1) / n
        was_target = rng.random() < intervention_prob

        credits.append(StageCredit(
            stage_name=f"stage_{i}",
            role=role,
            contribution=round(contribution, 4),
            was_intervention_target=was_target,
        ))

    return credits


def aggregate_credit(all_credits: list[list[StageCredit]]) -> dict[str, float]:
    """Aggregate credit across multiple runs by role.

    Returns a dict mapping role name to total accumulated credit.
    """
    totals: dict[str, float] = {}
    for run_credits in all_credits:
        for credit in run_credits:
            totals[credit.role] = totals.get(credit.role, 0.0) + credit.contribution
    return {role: round(val, 4) for role, val in totals.items()}
