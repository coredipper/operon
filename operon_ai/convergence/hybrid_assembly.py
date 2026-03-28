"""Hybrid assembly -- combine library-based adaptive assembly with LLM-generated fallback.

When the pattern library already contains a high-scoring template for the task
fingerprint, the adaptive assembly path is reused.  Otherwise a callable
``template_generator`` produces a fresh template (mimicking AutoSwarmBuilder),
registers it in the library, and builds a managed organism from the generated
stage specs.
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from ..organelles.nucleus import Nucleus
from ..patterns.managed import managed_organism, ManagedOrganism
from ..patterns.adaptive import adaptive_skill_organism, AdaptiveSkillOrganism
from ..patterns.repository import PatternLibrary, PatternTemplate, TaskFingerprint
from ..patterns.types import SkillStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stages_from_template(template: PatternTemplate) -> list[SkillStage]:
    """Convert a PatternTemplate's stage_specs into SkillStage instances.

    Each stage_spec dict is expected to have ``"name"``, ``"role"``, and
    ``"mode"`` keys.  A deterministic placeholder handler is attached so the
    stage can execute without a live provider.
    """
    stages: list[SkillStage] = []
    for spec in template.stage_specs:
        name = spec["name"]
        role = spec.get("role", name)
        mode = spec.get("mode", "fuzzy")

        def _placeholder_handler(
            task: str,
            state: dict[str, Any],
            outputs: dict[str, Any],
            stage: Any,
            *,
            _name: str = name,
            _role: str = role,
        ) -> str:
            return f"[{_name}/{_role}] processed: {task}"

        stages.append(
            SkillStage(
                name=name,
                role=role,
                mode=mode,
                handler=_placeholder_handler,
            )
        )
    return stages


# ---------------------------------------------------------------------------
# Default template generator
# ---------------------------------------------------------------------------


def default_template_generator(task: str) -> PatternTemplate:
    """Create a generic 3-stage template (plan -> execute -> review) for any task.

    This is the fallback when no LLM-based generator is configured.  The
    returned template uses a ``"skill_organism"`` topology with three stages:

    - **planner** (fuzzy) -- decompose the task into steps
    - **executor** (fuzzy) -- carry out the plan
    - **reviewer** (fixed) -- verify the result

    A :class:`TaskFingerprint` is derived from the task string using simple
    heuristics (sentence count as subtask proxy).
    """
    sentences = [
        s.strip()
        for s in task.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]

    fingerprint = TaskFingerprint(
        task_shape="sequential",
        tool_count=0,
        subtask_count=max(1, len(sentences)),
        required_roles=("planner", "executor", "reviewer"),
    )

    return PatternTemplate(
        template_id=uuid4().hex[:8],
        name=f"generated_{uuid4().hex[:6]}",
        topology="skill_organism",
        stage_specs=(
            {"name": "planner", "role": "planner", "mode": "fuzzy"},
            {"name": "executor", "role": "executor", "mode": "fuzzy"},
            {"name": "reviewer", "role": "reviewer", "mode": "fixed"},
        ),
        intervention_policy={"mode": "default"},
        fingerprint=fingerprint,
        tags=("generated", "default"),
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def hybrid_skill_organism(
    task: str,
    *,
    library: PatternLibrary,
    fingerprint: TaskFingerprint,
    fast_nucleus: Nucleus | None = None,
    deep_nucleus: Nucleus | None = None,
    template_generator: Callable[[str], PatternTemplate] | None = None,
    score_threshold: float = 0.5,
    **organism_kwargs: Any,
) -> AdaptiveSkillOrganism | ManagedOrganism:
    """Build an organism by combining library lookup with generated fallback.

    Logic:

    1. Query ``library.top_templates_for(fingerprint, limit=1)``.
    2. If the top template's score >= *score_threshold*, delegate to
       :func:`adaptive_skill_organism` (the existing adaptive assembly path).
    3. Otherwise, if *template_generator* is provided, call it to produce a new
       :class:`PatternTemplate`, register it in the library, and build a
       :func:`managed_organism` from its stage specs.
    4. If no generator and no good templates, raise :class:`ValueError`.

    Returns either an :class:`AdaptiveSkillOrganism` or a
    :class:`ManagedOrganism`, both of which expose a ``.run(task)`` method.
    """
    # 1. Query library
    ranked = library.top_templates_for(fingerprint, limit=1)

    # 2. High-scoring template available -- use adaptive path
    if ranked:
        _template, score = ranked[0]
        if score >= score_threshold:
            return adaptive_skill_organism(
                task,
                library=library,
                fingerprint=fingerprint,
                fast_nucleus=fast_nucleus,
                deep_nucleus=deep_nucleus,
                **organism_kwargs,
            )

    # 3. Fallback to generator — force direct-stage path (no library) so the
    #    generated template is used, not an older library entry.
    if template_generator is not None:
        generated = template_generator(task)
        library.register_template(generated)
        stages = _stages_from_template(generated)
        return managed_organism(
            stages=stages,
            fast_nucleus=fast_nucleus,
            deep_nucleus=deep_nucleus,
            **organism_kwargs,
        )

    # 4. No options left
    raise ValueError("No templates available and no generator provided")
