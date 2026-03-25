"""Pattern-first topology advice."""

from __future__ import annotations

from ..core.epistemic import (
    TopologyClass,
    TopologyRecommendation,
    recommend_topology,
)
from .types import TopologyAdvice


def advise_topology(
    *,
    task_shape: str,
    tool_count: int,
    subtask_count: int,
    error_tolerance: float = 0.1,
    library: object | None = None,
    fingerprint: object | None = None,
) -> TopologyAdvice:
    """Return a pattern-first recommendation from simple task inputs.

    If ``library`` (a PatternLibrary) and ``fingerprint`` (a TaskFingerprint)
    are provided, the advice includes a ``suggested_template`` from the library.
    """
    advice = _compute_advice(task_shape, tool_count, subtask_count, error_tolerance)
    if library is not None and fingerprint is not None:
        advice = _enrich_with_library(advice, library, fingerprint)
    return advice


def _compute_advice(
    task_shape: str,
    tool_count: int,
    subtask_count: int,
    error_tolerance: float,
) -> TopologyAdvice:
    """Core topology advice logic."""
    normalized = task_shape.strip().lower().replace("-", "_")

    if normalized in {"sequential", "pipeline", "serial"}:
        raw = recommend_topology(
            num_subtasks=subtask_count,
            subtasks_independent=False,
            num_tools=tool_count,
            error_tolerance=error_tolerance,
        )
        if error_tolerance <= 0.05:
            return TopologyAdvice(
                recommended_pattern="single_worker_with_reviewer",
                suggested_api="reviewer_gate(...)",
                topology=raw.recommended,
                rationale=(
                    "The task looks sequential, so avoid manufactured handoffs. "
                    "Because error tolerance is low, add a reviewer gate instead "
                    "of splitting the task across multiple workers."
                ),
                raw=raw,
            )
        return TopologyAdvice(
            recommended_pattern="single_worker",
            suggested_api="use one worker or one nucleus-backed agent",
            topology=raw.recommended,
            rationale=(
                "The task looks sequential. Start with one worker and keep "
                "handoffs to a minimum."
            ),
            raw=raw,
        )

    if normalized in {"parallel", "independent", "parallelizable"}:
        raw = recommend_topology(
            num_subtasks=subtask_count,
            subtasks_independent=True,
            num_tools=tool_count,
            error_tolerance=error_tolerance,
        )
        rationale = (
            "The task looks decomposable. Start with a specialist swarm "
            "and keep the coordinator light."
        )
        if tool_count > subtask_count * 2:
            rationale += " Tool density is high, so central routing becomes part of the design."
        return TopologyAdvice(
            recommended_pattern="specialist_swarm",
            suggested_api="specialist_swarm(...)",
            topology=raw.recommended,
            rationale=rationale,
            raw=raw,
        )

    if normalized in {"mixed", "hybrid"}:
        raw = TopologyRecommendation(
            recommended=TopologyClass.HYBRID,
            rationale="Mixed independence and tool density suggest hybrid topology",
        )
        return TopologyAdvice(
            recommended_pattern="specialist_swarm",
            suggested_api="specialist_swarm(...)",
            topology=raw.recommended,
            rationale=(
                "The task mixes independent and dependent phases. Start with a "
                "small specialist swarm, then inspect the generated analysis "
                "before adding more coordination."
            ),
            raw=raw,
        )

    raise ValueError(
        "task_shape must be one of: sequential, parallel, independent, mixed"
    )


def _enrich_with_library(advice: TopologyAdvice, library: object, fingerprint: object) -> TopologyAdvice:
    """If a library and fingerprint are available, attach the best template."""
    try:
        ranked = library.top_templates_for(fingerprint)  # type: ignore[union-attr]
        if ranked:
            template, score = ranked[0]
            return TopologyAdvice(
                recommended_pattern=advice.recommended_pattern,
                suggested_api=advice.suggested_api,
                topology=advice.topology,
                rationale=advice.rationale + f" Library match: '{template.name}' (score={score:.3f}).",
                raw=advice.raw,
                suggested_template=template,
            )
    except Exception:
        pass
    return advice
