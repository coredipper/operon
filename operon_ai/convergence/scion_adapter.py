"""Scion convergence adapter -- parse Scion grove configs into Operon types.

Scion organises agents as containerized workers in a "grove" with messaging
channels, isolation boundaries, and an optional watcher agent.  This module
converts those plain-dict grove configs into :class:`ExternalTopology`,
:class:`SkillStage` tuples, and :class:`PatternTemplate` instances without
importing scion.

All inputs are plain dicts; no Scion imports.
"""

from __future__ import annotations

import copy
import re
from typing import Any
from ..convergence.types import ExternalTopology
from ..patterns.repository import PatternLibrary, PatternTemplate, TaskFingerprint
from ..patterns.types import CognitiveMode, SkillStage

# ---------------------------------------------------------------------------
# Runtime profile -> cognitive-mode mapping
# ---------------------------------------------------------------------------

_OBSERVATIONAL_SKILLS: frozenset[str] = frozenset({
    "telemetry_monitoring",
    "convergence_detection",
    "review",
    "verification",
    "audit",
    "monitoring",
})

_ACTION_SKILLS: frozenset[str] = frozenset({
    "code_generation",
    "execution",
    "writing",
    "deployment",
    "testing",
})


_WATCHER_SENTINEL = "operon-watcher"

_COMPILED_NAME_SUFFIX = re.compile(r"_\d+$")


def _resolve_role(agent: dict) -> str:
    """Extract semantic role from a Scion agent dict.

    Reads the explicit ``role`` field if present and non-empty (set by the
    Scion compiler).  Otherwise strips trailing ``_N`` suffixes from the
    name to recover the stable role string.
    """
    role = agent.get("role")
    if isinstance(role, str) and role.strip():
        return role.strip()
    name = agent.get("name", "unnamed")
    return _COMPILED_NAME_SUFFIX.sub("", name)


def _is_watcher(agent: dict, watcher_name: str = _WATCHER_SENTINEL) -> bool:
    """Return True if *agent* is the Scion watcher sentinel."""
    return agent.get("name") == watcher_name


def _skills_to_mode(skills: list[str]) -> tuple[str, CognitiveMode]:
    """Return (mode_string, CognitiveMode) from a Scion agent's skill list."""
    skill_set = frozenset(s.strip().lower() for s in skills)
    obs_count = len(skill_set & _OBSERVATIONAL_SKILLS)
    act_count = len(skill_set & _ACTION_SKILLS)
    if obs_count > act_count:
        return "fixed", CognitiveMode.OBSERVATIONAL
    return "fuzzy", CognitiveMode.ACTION_ORIENTED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_scion_grove(config: dict) -> ExternalTopology:
    """Parse a Scion grove config dict into an :class:`ExternalTopology`.

    Parameters
    ----------
    config:
        Plain dict with shape::

            {
                "grove": {"name": "my-grove", "runtime": "docker"},
                "agents": [
                    {
                        "name": "researcher",
                        "template": {"system_prompt": "...", "skills": [...]},
                        "runtime_profile": "medium",
                        "isolation": {"git_worktree": True, "credentials": "isolated"},
                    },
                    ...
                    {
                        "name": "operon-watcher",
                        "template": {"system_prompt": "...", "skills": [...]},
                        "isolation": {"git_worktree": False, "credentials": "shared"},
                    },
                ],
                "messaging": [
                    {"from": "researcher", "to": "writer", "channel": "..."},
                ],
                "watcher": {"enabled": True, "agent_name": "operon-watcher", "telemetry": "otel"},
            }

    Returns
    -------
    ExternalTopology
        Operon-native topology with grove agents (excluding watcher) and
        messaging edges.
    """
    agents_raw: list[dict] = config.get("agents", [])
    messaging: list[dict] = config.get("messaging", [])
    grove: dict = config.get("grove", {})
    watcher_cfg: dict = config.get("watcher", {})
    watcher_name: str = watcher_cfg.get("agent_name", "operon-watcher")

    # Build agent dicts, filtering out the watcher agent.
    agents: list[dict[str, Any]] = []
    for a in agents_raw:
        if _is_watcher(a, watcher_name):
            continue
        name = a.get("name", "unnamed")
        role = _resolve_role(a)
        template = a.get("template", {})
        isolation = a.get("isolation", {})
        agents.append({
            "name": name,
            "role": role,
            "capabilities": template.get("skills", []),
            "runtime_profile": a.get("runtime_profile", "medium"),
            "git_worktree": isolation.get("git_worktree", True),
            "credentials": isolation.get("credentials", "isolated"),
        })

    # Build edges from messaging channels.
    edges: list[tuple[str, str]] = [
        (m["from"], m["to"]) for m in messaging
        if m.get("from") != watcher_name and m.get("to") != watcher_name
    ]

    return ExternalTopology(
        source="scion",
        pattern_name="ScionGrove",
        agents=tuple(agents),
        edges=tuple(edges),
        metadata={
            "grove_name": grove.get("name", "unnamed"),
            "runtime": grove.get("runtime", "docker"),
            "watcher_enabled": watcher_cfg.get("enabled", False),
            "watcher_telemetry": watcher_cfg.get("telemetry", ""),
            "_scion_config": copy.deepcopy(config),
        },
    )


def scion_agents_to_stages(
    agents: list[dict],
    *,
    watcher_name: str = _WATCHER_SENTINEL,
) -> tuple[SkillStage, ...]:
    """Map Scion agent dicts to Operon :class:`SkillStage` objects.

    Automatically filters the watcher agent if present.

    Parameters
    ----------
    agents:
        List of agent dicts from a Scion grove config.
    watcher_name:
        Name of the watcher agent to filter out.  Defaults to
        ``"operon-watcher"``.

    Returns
    -------
    tuple[SkillStage, ...]
        One stage per non-watcher agent, with mode derived from skill
        categories and role resolved from the explicit ``role`` field
        or by stripping compiler suffixes from the name.
    """
    stages: list[SkillStage] = []
    for a in agents:
        if _is_watcher(a, watcher_name):
            continue
        name: str = a.get("name", "unnamed")
        role: str = _resolve_role(a)
        template: dict = a.get("template", {})
        skills: list[str] = template.get("skills", [])
        prompt: str = template.get("system_prompt", f"You are the {role} agent.")

        mode_str, cog_mode = _skills_to_mode(skills)

        stages.append(SkillStage(
            name=name,
            role=role,
            instructions=prompt,
            mode=mode_str,
            cognitive_mode=cog_mode,
        ))

    return tuple(stages)


def scion_to_template(config: dict) -> PatternTemplate:
    """Convert a Scion grove config to an Operon :class:`PatternTemplate`.

    Parameters
    ----------
    config:
        Same shape as for :func:`parse_scion_grove`.

    Returns
    -------
    PatternTemplate
        Ready for registration in a :class:`PatternLibrary`.
    """
    agents_raw: list[dict] = config.get("agents", [])
    grove: dict = config.get("grove", {})
    watcher_cfg: dict = config.get("watcher", {})
    watcher_name: str = watcher_cfg.get("agent_name", "operon-watcher")

    # Build stage specs from non-watcher agents.
    stage_specs: list[dict[str, Any]] = []
    for a in agents_raw:
        if _is_watcher(a, watcher_name):
            continue
        name = a.get("name", "unnamed")
        role = _resolve_role(a)
        template = a.get("template", {})
        skills = template.get("skills", [])
        mode_str, _ = _skills_to_mode(skills)
        stage_specs.append({
            "name": name,
            "role": role,
            "mode": mode_str,
        })

    roles = tuple(sorted({s["role"] for s in stage_specs})) if stage_specs else ()

    # Derive topology from messaging structure.
    ext_topo = parse_scion_grove(config)
    from .swarms_adapter import _classify_task_shape, _shape_to_topology
    task_shape = _classify_task_shape(ext_topo)
    topology = _shape_to_topology(task_shape, len(stage_specs))
    if len(stage_specs) <= 1:
        topology = "single_worker"
        task_shape = "sequential"

    fingerprint = TaskFingerprint(
        task_shape=task_shape,
        tool_count=sum(
            len(a.get("template", {}).get("skills", []))
            for a in agents_raw if a.get("name") != watcher_name
        ),
        subtask_count=len(stage_specs),
        required_roles=roles,
        tags=("scion",),
    )

    return PatternTemplate(
        template_id=PatternLibrary.make_id(),
        name=f"scion_{grove.get('name', 'grove')}",
        topology=topology,
        stage_specs=tuple(stage_specs),
        intervention_policy={
            "watcher_enabled": watcher_cfg.get("enabled", False),
            "runtime": grove.get("runtime", "docker"),
        },
        fingerprint=fingerprint,
        tags=("scion", grove.get("runtime", "docker")),
    )
