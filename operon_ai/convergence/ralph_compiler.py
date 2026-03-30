"""Ralph compiler -- compile Operon organisms into Ralph orchestrator config dicts.

Produces serializable dicts that the Ralph framework can consume.
No ``ralph`` imports -- only ``operon_ai`` types.

Design principles:
  - Output is a plain dict (JSON-serializable).
  - Each :class:`SkillStage` becomes one hat entry.
  - Stage cognitive mode determines hat pattern.
  - Sequential stage order becomes event subscriptions.
"""

from __future__ import annotations

from typing import Any

from ..patterns.managed import ManagedOrganism
from ..patterns.organism import SkillOrganism
from ..patterns.types import SkillStage, resolve_cognitive_mode, CognitiveMode
from .types import RuntimeConfig


_MODE_TO_PATTERN: dict[str, str] = {
    "fast": "code-assist",
    "fixed": "review",
    "deterministic": "review",
    "fuzzy": "code-assist",
}


def _stage_to_hat(stage: SkillStage) -> dict[str, Any]:
    """Convert a SkillStage to a Ralph hat dict."""
    pattern = _MODE_TO_PATTERN.get(stage.mode, "code-assist")
    return {
        "name": stage.name,
        "pattern": pattern,
        "system_prompt": stage.instructions or f"{stage.role} stage",
    }


def _build_events(stages: tuple[SkillStage, ...]) -> list[dict[str, Any]]:
    """Build sequential event subscriptions from stage order."""
    events = []
    for i in range(len(stages) - 1):
        events.append({
            "from": stages[i].name,
            "event": f"{stages[i].role}.complete",
            "to": stages[i + 1].name,
        })
    return events


def organism_to_ralph(
    organism: SkillOrganism,
    *,
    config: RuntimeConfig | None = None,
    backend: str = "claude",
) -> dict[str, Any]:
    """Compile a SkillOrganism into a Ralph orchestrator config dict.

    Returns a dict with keys: backend, hats, events, backpressure,
    iteration_limit.
    """
    if not organism.stages:
        raise ValueError("Cannot compile organism with no stages")

    cfg = config or RuntimeConfig()
    hats = [_stage_to_hat(s) for s in organism.stages]
    events = _build_events(organism.stages)

    # Add backpressure gates if any stage is observational (review mode).
    backpressure = []
    for s in organism.stages:
        cm = resolve_cognitive_mode(s)
        if cm == CognitiveMode.OBSERVATIONAL:
            backpressure = ["tests", "lint"]
            break

    return {
        "backend": backend,
        "hats": hats,
        "events": events,
        "backpressure": backpressure,
        "iteration_limit": cfg.max_retries * len(organism.stages) if cfg.max_retries > 1 else 10,
    }


def managed_to_ralph(
    organism: ManagedOrganism,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compile a ManagedOrganism by extracting its inner SkillOrganism."""
    inner = organism._organism or (
        organism._adaptive._organism if organism._adaptive else None
    )
    if inner is None:
        raise ValueError("ManagedOrganism has no inner SkillOrganism")
    return organism_to_ralph(inner, **kwargs)
