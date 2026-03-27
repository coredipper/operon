"""AnimaWorks adapter -- maps AnimaWorks org configs to Operon typed stages.

AnimaWorks organises agents into supervisor hierarchies with role templates.
This module converts those plain-dict configs into :class:`ExternalTopology`,
:class:`SkillStage` tuples, and :class:`PatternTemplate` instances without
importing any AnimaWorks code.
"""

from __future__ import annotations

from typing import Any

from ..convergence.types import ExternalTopology
from ..patterns.repository import PatternTemplate, TaskFingerprint
from ..patterns.types import CognitiveMode, SkillStage


# ---------------------------------------------------------------------------
# Role -> cognitive-mode mapping
# ---------------------------------------------------------------------------

_OBSERVATIONAL_ROLES = frozenset({"reviewer", "analyst", "auditor"})
_ACTION_ROLES = frozenset({
    "engineer", "developer", "coder",
    "manager", "supervisor", "lead",
    "writer", "documenter",
})


def _cognitive_mode_for_role(role: str) -> CognitiveMode:
    """Return the cognitive mode implied by an AnimaWorks role name."""
    role_lower = role.strip().lower()
    if role_lower in _OBSERVATIONAL_ROLES:
        return CognitiveMode.OBSERVATIONAL
    # Everything else (including unknown) is action-oriented.
    return CognitiveMode.ACTION_ORIENTED


def _stage_mode_for_role(role: str) -> str:
    """Return the ``SkillStage.mode`` string for a given role."""
    role_lower = role.strip().lower()
    if role_lower in _OBSERVATIONAL_ROLES:
        return "fixed"
    return "fuzzy"


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------

def _build_edges(
    supervisor_name: str,
    agent_names: list[str],
    communication: str,
) -> tuple[tuple[str, str], ...]:
    """Build directed edges based on the communication style.

    * ``"hierarchical"`` (default): supervisor -> each agent.
    * ``"flat"``: fully connected among agents (no supervisor edges).
    * ``"ring"``: agents connected in a ring, supervisor -> first agent.
    """
    comm = communication.strip().lower()

    if comm == "flat":
        edges: list[tuple[str, str]] = []
        for i, a in enumerate(agent_names):
            for b in agent_names[i + 1:]:
                edges.append((a, b))
                edges.append((b, a))
        return tuple(edges)

    if comm == "ring":
        edges = []
        if agent_names:
            edges.append((supervisor_name, agent_names[0]))
            for i in range(len(agent_names) - 1):
                edges.append((agent_names[i], agent_names[i + 1]))
            if len(agent_names) > 1:
                edges.append((agent_names[-1], agent_names[0]))
        return tuple(edges)

    # Default: hierarchical
    return tuple((supervisor_name, a) for a in agent_names)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_animaworks_org(org_config: dict[str, Any]) -> ExternalTopology:
    """Parse an AnimaWorks organization config into an :class:`ExternalTopology`.

    Parameters
    ----------
    org_config:
        Plain dict with keys ``"name"``, ``"supervisor"``, ``"agents"``,
        and optionally ``"communication"`` (``"hierarchical"`` | ``"flat"``
        | ``"ring"``).

    Returns
    -------
    ExternalTopology
        A source-agnostic topology representation consumable by Operon's
        epistemic analysis pipeline.
    """
    supervisor = org_config["supervisor"]
    agents_raw: list[dict[str, Any]] = org_config.get("agents", [])
    communication: str = org_config.get("communication", "hierarchical")

    # Build agent dicts (supervisor is also an agent node).
    sup_dict: dict[str, Any] = {
        "name": supervisor["name"],
        "role": supervisor.get("role", "manager"),
        "skills": supervisor.get("skills", []),
    }
    agent_dicts: list[dict[str, Any]] = [sup_dict]
    agent_names: list[str] = []
    for a in agents_raw:
        agent_dicts.append({
            "name": a["name"],
            "role": a.get("role", "worker"),
            "skills": a.get("skills", []),
        })
        agent_names.append(a["name"])

    edges = _build_edges(supervisor["name"], agent_names, communication)

    return ExternalTopology(
        source="animaworks",
        pattern_name=org_config.get("name", "animaworks_org"),
        agents=tuple(agent_dicts),
        edges=edges,
        metadata={"communication": communication},
    )


def animaworks_roles_to_stages(
    roles: list[dict[str, Any]],
) -> tuple[SkillStage, ...]:
    """Map AnimaWorks role dicts to Operon :class:`SkillStage` instances.

    Each role dict should have at least ``"name"`` and ``"role"`` keys, plus
    an optional ``"skills"`` list used to populate ``instructions``.

    Cognitive-mode mapping:

    * ``"reviewer"`` / ``"analyst"`` / ``"auditor"`` -> ``mode="fixed"``
      (observational, System A).
    * All other roles (including unknown) -> ``mode="fuzzy"``
      (action-oriented, System B).
    """
    stages: list[SkillStage] = []
    for role_dict in roles:
        name: str = role_dict.get("name", "unnamed")
        role: str = role_dict.get("role", "worker")
        skills: list[str] = role_dict.get("skills", [])

        mode = _stage_mode_for_role(role)
        cognitive = _cognitive_mode_for_role(role)
        instructions = ", ".join(skills) if skills else ""

        stages.append(
            SkillStage(
                name=name,
                role=role,
                mode=mode,
                cognitive_mode=cognitive,
                instructions=instructions,
                handler=lambda _state=None, **_kw: {"status": "ok"},
            ),
        )
    return tuple(stages)


def animaworks_to_template(org_config: dict[str, Any]) -> PatternTemplate:
    """Convert an AnimaWorks org config to a :class:`PatternTemplate`.

    The template captures the topology shape, stage specifications, and a
    :class:`TaskFingerprint` derived from the org's roles and skills.
    """
    topology = parse_animaworks_org(org_config)
    roles_list: list[dict[str, Any]] = []
    if "supervisor" in org_config:
        roles_list.append(org_config["supervisor"])
    roles_list.extend(org_config.get("agents", []))

    stages = animaworks_roles_to_stages(roles_list)

    # Build stage specs as serialisable dicts.
    stage_specs: list[dict[str, Any]] = []
    for s in stages:
        stage_specs.append({
            "name": s.name,
            "role": s.role,
            "mode": s.mode,
            "instructions": s.instructions,
        })

    all_roles = tuple(r.get("role", "worker") for r in roles_list)
    all_skills: list[str] = []
    for r in roles_list:
        all_skills.extend(r.get("skills", []))

    communication = org_config.get("communication", "hierarchical")
    if communication == "hierarchical":
        topo_type = "specialist_swarm"
    elif communication == "ring":
        topo_type = "skill_organism"
    else:
        topo_type = "specialist_swarm"

    fingerprint = TaskFingerprint(
        task_shape="parallel" if communication == "flat" else "mixed",
        tool_count=len(all_skills),
        subtask_count=len(roles_list),
        required_roles=all_roles,
        tags=tuple(all_skills),
    )

    from ..patterns.repository import PatternLibrary

    return PatternTemplate(
        template_id=PatternLibrary.make_id(),
        name=f"animaworks_{org_config.get('name', 'org')}",
        topology=topo_type,
        stage_specs=tuple(stage_specs),
        intervention_policy={"type": "default", "source": "animaworks"},
        fingerprint=fingerprint,
        tags=("animaworks", communication),
    )
