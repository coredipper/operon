"""Swarms compiler -- compile Operon organisms into Swarms workflow config dicts.

Produces serializable dicts that the Swarms framework can consume.
No ``swarms`` imports -- only ``operon_ai`` types.

Design principles:
  - Output is a plain dict (JSON-serializable).
  - Users install ``swarms`` separately; this module never imports it.
  - Each :class:`SkillStage` becomes one agent entry.
  - Sequential vs. graph topology is inferred from the stage structure.
"""

from __future__ import annotations

from typing import Any

from ..patterns.managed import ManagedOrganism
from ..patterns.organism import SkillOrganism
from ..patterns.types import CognitiveMode, SkillStage, resolve_cognitive_mode
from .types import RuntimeConfig

# ---------------------------------------------------------------------------
# Model selection heuristic
# ---------------------------------------------------------------------------

_MODE_TO_MODEL: dict[str, str] = {
    "fast": "gpt-5.4-mini",
    "fixed": "gpt-5.4-mini",
    "deterministic": "gpt-5.4-mini",
    "fuzzy": "gpt-5.4",
    "deep": "gpt-5.4",
}


def _model_for_stage(stage: SkillStage, provider: str) -> str:
    """Pick a model name hint from the stage mode and provider."""
    mode = stage.mode.strip().lower().replace("-", "_")
    base = _MODE_TO_MODEL.get(mode, "gpt-5.4")
    if provider == "anthropic":
        return base.replace("gpt-5.4-mini", "claude-haiku-4-5-20251001").replace(
            "gpt-5.4", "claude-sonnet-4-6-20260301"
        )
    if provider == "mock":
        return f"mock-{mode}"
    return base


# ---------------------------------------------------------------------------
# Topology detection
# ---------------------------------------------------------------------------


def _is_linear_chain(stages: tuple[SkillStage, ...]) -> bool:
    """Return True when stages form a simple linear pipeline.

    Since :class:`SkillStage` has no explicit dependency list, the default
    organism execution is always linear.
    """
    if len(stages) <= 1:
        return True
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def organism_to_swarms(
    organism: SkillOrganism,
    *,
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Compile a :class:`SkillOrganism` into a Swarms workflow config dict.

    Output dict shape::

        {
            "workflow_type": "SequentialWorkflow" | "GraphWorkflow",
            "agents": [
                {
                    "name": str,
                    "role": str,
                    "system_prompt": str,
                    "model": str,
                    "timeout": float,
                },
                ...
            ],
            "edges": [("from", "to"), ...],
            "config": {"max_loops": int, "autosave": bool, ...},
        }

    Sequential if all stages form a linear chain, Graph otherwise.

    Parameters
    ----------
    organism:
        The :class:`SkillOrganism` to compile.
    config:
        Optional runtime hints (provider, timeout, retries).

    Returns
    -------
    dict[str, Any]
        A plain dict consumable by the Swarms ``SequentialWorkflow`` or
        ``GraphWorkflow`` constructor.

    Raises
    ------
    ValueError
        If the organism has no stages.
    """
    cfg = config or RuntimeConfig()
    stages = organism.stages

    if not stages:
        raise ValueError("Cannot compile an organism with no stages to Swarms")

    # Build agent list.
    agents: list[dict[str, Any]] = []
    for stage in stages:
        agents.append({
            "name": stage.name,
            "role": stage.role,
            "system_prompt": stage.instructions,
            "model": _model_for_stage(stage, cfg.provider),
            "timeout": cfg.timeout,
        })

    # Build edges: linear chain from stage order.
    linear = _is_linear_chain(stages)
    edges: list[tuple[str, str]] = []
    for i in range(len(stages) - 1):
        edges.append((stages[i].name, stages[i + 1].name))

    workflow_type = "SequentialWorkflow" if linear else "GraphWorkflow"

    return {
        "workflow_type": workflow_type,
        "agents": agents,
        "edges": edges,
        "config": {
            "max_loops": cfg.max_retries + 1,
            "autosave": True,
            "provider": cfg.provider,
            "sandbox": cfg.sandbox,
        },
    }


def managed_to_swarms(
    organism: ManagedOrganism,
    *,
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Compile a :class:`ManagedOrganism`, extracting the inner SkillOrganism.

    Parameters
    ----------
    organism:
        A managed organism that wraps a :class:`SkillOrganism`.
    config:
        Optional runtime hints.

    Returns
    -------
    dict[str, Any]
        Same shape as :func:`organism_to_swarms`.

    Raises
    ------
    ValueError
        If the managed organism has no inner organism.
    """
    inner = organism._organism
    if inner is None:
        # Try the adaptive assembly path.
        if organism._adaptive is not None:
            inner = organism._adaptive._organism
    if inner is None:
        raise ValueError(
            "ManagedOrganism has no inner SkillOrganism to compile"
        )
    return organism_to_swarms(inner, config=config)
