"""DeerFlow compiler -- compile Operon organisms into DeerFlow session config dicts.

Produces serializable dicts that the DeerFlow framework can consume.
No ``deerflow``, ``langchain``, or ``langgraph`` imports -- only ``operon_ai`` types.

Design principles:
  - Output is a plain dict (JSON-serializable).
  - Users install DeerFlow separately; this module never imports it.
  - First stage becomes the lead agent; remaining stages become sub-agents.
"""

from __future__ import annotations

from typing import Any

from ..patterns.managed import ManagedOrganism
from ..patterns.organism import SkillOrganism
from ..patterns.types import CognitiveMode, SkillStage, resolve_cognitive_mode
from .types import RuntimeConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instructions_to_skills(instructions: str) -> list[str]:
    """Split stage instructions into a list of skill strings.

    If the instructions contain multiple sentences, each sentence becomes a
    separate skill entry.  A single sentence is returned as-is in a list.
    """
    if not instructions or not instructions.strip():
        return []
    # Split on sentence-ending punctuation followed by whitespace.
    parts: list[str] = []
    current: list[str] = []
    for char in instructions:
        current.append(char)
        if char in ".!?" and len(current) > 1:
            sentence = "".join(current).strip()
            if sentence:
                parts.append(sentence)
            current = []
    # Remainder that didn't end with punctuation.
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts if parts else [instructions.strip()]


def _recursion_limit_from_timeout(timeout: float) -> int:
    """Approximate a DeerFlow recursion_limit from a timeout in seconds.

    Heuristic: ~2 seconds per recursion step.
    """
    return max(int(timeout / 2.0), 10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def organism_to_deerflow(
    organism: SkillOrganism,
    *,
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Compile a :class:`SkillOrganism` into a DeerFlow session config dict.

    Output dict shape::

        {
            "assistant_id": str,
            "skills": [str, ...],
            "sub_agents": [
                {"name": str, "role": str, "skills": [str, ...]},
                ...
            ],
            "recursion_limit": int,
            "sandbox": str,
            "config": {"thinking_enabled": bool},
        }

    First stage becomes the lead agent; remaining stages become sub-agents.

    Parameters
    ----------
    organism:
        The :class:`SkillOrganism` to compile.
    config:
        Optional runtime hints (provider, timeout, sandbox).

    Returns
    -------
    dict[str, Any]
        A plain dict consumable by a DeerFlow session constructor.

    Raises
    ------
    ValueError
        If the organism has no stages.
    """
    cfg = config or RuntimeConfig()
    stages = organism.stages

    if not stages:
        raise ValueError(
            "Cannot compile an organism with no stages to DeerFlow"
        )

    # First stage is the lead agent.
    lead = stages[0]
    lead_skills = _instructions_to_skills(lead.instructions)

    # Remaining stages become sub-agents.
    sub_agents: list[dict[str, Any]] = []
    for stage in stages[1:]:
        sub_agents.append({
            "name": stage.name,
            "role": stage.role,
            "skills": _instructions_to_skills(stage.instructions),
        })

    # Determine thinking_enabled from the lead stage's cognitive mode.
    cog_mode = resolve_cognitive_mode(lead)
    thinking_enabled = cog_mode != CognitiveMode.OBSERVATIONAL

    return {
        "assistant_id": lead.name,
        "skills": lead_skills,
        "sub_agents": sub_agents,
        "recursion_limit": _recursion_limit_from_timeout(cfg.timeout),
        "sandbox": cfg.sandbox,
        "config": {
            "thinking_enabled": thinking_enabled,
        },
    }


def managed_to_deerflow(
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
        Same shape as :func:`organism_to_deerflow`.

    Raises
    ------
    ValueError
        If the managed organism has no inner organism.
    """
    inner = organism._organism
    if inner is None:
        if organism._adaptive is not None:
            inner = organism._adaptive._organism
    if inner is None:
        raise ValueError(
            "ManagedOrganism has no inner SkillOrganism to compile"
        )
    return organism_to_deerflow(inner, config=config)
