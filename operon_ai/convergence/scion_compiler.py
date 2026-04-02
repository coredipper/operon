"""Scion compiler -- compile Operon organisms into Scion grove config dicts.

Produces serializable dicts that the Scion framework can consume.
No ``scion`` imports -- only ``operon_ai`` types.

Design principles:
  - Output is a plain dict (JSON-serializable).
  - Each :class:`SkillStage` becomes a containerized agent.
  - Every agent gets an isolated git worktree by default.
  - A dedicated watcher agent monitors via OTEL telemetry.
  - Messaging topology mirrors stage order.
"""

from __future__ import annotations

from typing import Any

from ..patterns.managed import ManagedOrganism
from ..patterns.organism import SkillOrganism
from ..patterns.types import SkillStage
from .types import RuntimeConfig


def _stage_to_agent(stage: SkillStage, runtime_profile: str) -> dict[str, Any]:
    """Convert a SkillStage to a Scion agent definition."""
    skills = []
    if stage.instructions:
        skills = [s.strip() for s in stage.instructions.split(".") if s.strip()][:5]
    if not skills:
        skills = [stage.role]

    return {
        "name": stage.name,
        "role": stage.role,
        "template": {
            "system_prompt": stage.instructions or f"You are the {stage.role} agent.",
            "skills": skills,
        },
        "runtime_profile": runtime_profile,
        "isolation": {
            "git_worktree": True,
            "credentials": "isolated",
        },
    }


def _build_messaging(stages: tuple[SkillStage, ...]) -> list[dict[str, Any]]:
    """Build sequential messaging topology from stage order."""
    messages = []
    for i in range(len(stages) - 1):
        messages.append({
            "from": stages[i].name,
            "to": stages[i + 1].name,
            "channel": f"{stages[i].name}_to_{stages[i + 1].name}",
        })
    return messages


def organism_to_scion(
    organism: SkillOrganism,
    *,
    config: RuntimeConfig | None = None,
    grove_name: str = "operon-deployment",
    runtime: str = "docker",
) -> dict[str, Any]:
    """Compile a SkillOrganism into a Scion grove config dict.

    Returns a dict with keys: grove, agents, messaging, watcher.
    Each stage becomes a containerized agent with isolated git worktree.
    A dedicated watcher agent is added for OTEL telemetry monitoring.
    """
    if not organism.stages:
        raise ValueError("Cannot compile organism with no stages")

    cfg = config or RuntimeConfig()
    runtime_str = cfg.sandbox if cfg.sandbox != "none" else runtime

    agents = [_stage_to_agent(s, runtime_str) for s in organism.stages]
    messaging = _build_messaging(organism.stages)

    # Add a dedicated watcher agent.
    watcher_agent = {
        "name": "operon-watcher",
        "template": {
            "system_prompt": "Monitor agent execution via OTEL telemetry. Report convergence status.",
            "skills": ["telemetry_monitoring", "convergence_detection"],
        },
        "runtime_profile": runtime_str,
        "isolation": {
            "git_worktree": False,
            "credentials": "shared",
        },
    }

    return {
        "grove": {
            "name": grove_name,
            "runtime": runtime_str,
        },
        "agents": agents + [watcher_agent],
        "messaging": messaging,
        "watcher": {
            "enabled": True,
            "agent_name": "operon-watcher",
            "telemetry": "otel",
        },
    }


def managed_to_scion(
    organism: ManagedOrganism,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compile a ManagedOrganism by extracting its inner SkillOrganism."""
    inner = organism._organism or (
        organism._adaptive._organism if organism._adaptive else None
    )
    if inner is None:
        raise ValueError("ManagedOrganism has no inner SkillOrganism")
    return organism_to_scion(inner, **kwargs)
