"""DeerFlow 2.0 session-config adapter.

Parses DeerFlow session configs (LangChain/LangGraph-based) into Operon's
typed analysis system.  DeerFlow uses Markdown-based skills, progressive
loading, and sub-agent spawning — this adapter maps those concepts to
:class:`ExternalTopology`, :class:`SkillStage`, and :class:`PatternTemplate`.

All inputs are plain dicts; no DeerFlow, LangChain, or LangGraph imports.
"""

from __future__ import annotations

from uuid import uuid4

from ..convergence.types import ExternalTopology
from ..patterns.repository import PatternTemplate, TaskFingerprint
from ..patterns.types import CognitiveMode, SkillStage

# ---------------------------------------------------------------------------
# Category -> cognitive-mode mapping
# ---------------------------------------------------------------------------

_OBSERVATIONAL_CATEGORIES: frozenset[str] = frozenset({
    "research",
    "verification",
    "review",
})

_ACTION_CATEGORIES: frozenset[str] = frozenset({
    "generation",
    "writing",
    "execution",
    "code",
})


def _category_to_mode(category: str) -> tuple[str, CognitiveMode]:
    """Return (mode_string, CognitiveMode) for a DeerFlow skill category."""
    cat = category.strip().lower()
    if cat in _OBSERVATIONAL_CATEGORIES:
        return "fixed", CognitiveMode.OBSERVATIONAL
    # Action-oriented or unknown both map to fuzzy / ACTION_ORIENTED.
    return "fuzzy", CognitiveMode.ACTION_ORIENTED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_deerflow_session(session_config: dict) -> ExternalTopology:
    """Parse a DeerFlow session config dict into an :class:`ExternalTopology`.

    Parameters
    ----------
    session_config:
        Plain dict with shape::

            {
                "assistant_id": "lead_agent",
                "skills": ["web_research", ...],
                "sub_agents": [
                    {"name": "researcher", "role": "researcher", "skills": [...]},
                    ...
                ],
                "recursion_limit": 100,
                "sandbox": "docker",
                "config": {"thinking_enabled": True},
            }

    Returns
    -------
    ExternalTopology
        Operon-native topology with the lead agent as hub and sub-agents as
        workers connected by hierarchical edges.
    """
    lead_id: str = session_config.get("assistant_id", "lead_agent")
    sub_agents: list[dict] = session_config.get("sub_agents", [])
    skills: list[str] = session_config.get("skills", [])
    sandbox: str = session_config.get("sandbox", "local")
    recursion_limit: int = session_config.get("recursion_limit", 100)
    config: dict = session_config.get("config", {})

    # Build agent dicts — lead + each sub-agent.
    agents: list[dict] = [
        {
            "name": lead_id,
            "role": "lead",
            "skills": tuple(skills),
        },
    ]
    for sa in sub_agents:
        agents.append({
            "name": sa.get("name", "unnamed"),
            "role": sa.get("role", "worker"),
            "skills": tuple(sa.get("skills", [])),
        })

    # Hierarchical edges: lead -> each sub-agent.
    edges: list[tuple[str, str]] = [
        (lead_id, sa.get("name", "unnamed")) for sa in sub_agents
    ]

    return ExternalTopology(
        source="deerflow",
        pattern_name="HierarchicalDeerFlow",
        agents=tuple(agents),
        edges=tuple(edges),
        metadata={
            "sandbox": sandbox,
            "recursion_limit": recursion_limit,
            "config": config,
            "skills": skills,
        },
    )


def deerflow_skills_to_stages(skills: list[dict]) -> tuple[SkillStage, ...]:
    """Map DeerFlow skill metadata dicts to Operon :class:`SkillStage` objects.

    Parameters
    ----------
    skills:
        List of skill dicts, each with ``name``, ``description``, and
        ``category`` keys.

    Returns
    -------
    tuple[SkillStage, ...]
        One stage per skill, with ``mode`` and ``cognitive_mode`` derived from
        the skill category.
    """
    stages: list[SkillStage] = []
    for skill in skills:
        name: str = skill.get("name", "unnamed_skill")
        description: str = skill.get("description", "")
        category: str = skill.get("category", "unknown")

        mode_str, cog_mode = _category_to_mode(category)

        stages.append(SkillStage(
            name=name,
            role=category if category else "general",
            instructions=description,
            mode=mode_str,
            cognitive_mode=cog_mode,
        ))

    return tuple(stages)


def deerflow_to_template(session_config: dict) -> PatternTemplate:
    """Convert a DeerFlow session config to an Operon :class:`PatternTemplate`.

    The resulting template captures the hierarchical topology, sub-agent
    stage specs, and a :class:`TaskFingerprint` derived from the session
    structure.

    Parameters
    ----------
    session_config:
        Same shape as for :func:`parse_deerflow_session`.

    Returns
    -------
    PatternTemplate
        Ready for registration in a :class:`PatternLibrary`.
    """
    sub_agents: list[dict] = session_config.get("sub_agents", [])
    skills: list[str] = session_config.get("skills", [])

    # Build stage specs from sub-agents (include lead if no sub-agents).
    stage_specs: list[dict] = []
    if not sub_agents:
        # No sub-agents: the lead assistant is the only stage.
        lead_id = session_config.get("assistant_id", "lead_agent")
        stage_specs.append({
            "name": lead_id,
            "role": "lead",
            "skills": list(skills),
        })
    else:
        for sa in sub_agents:
            stage_specs.append({
                "name": sa.get("name", "unnamed"),
                "role": sa.get("role", "worker"),
                "skills": list(sa.get("skills", [])),
            })

    # Derive roles.
    roles: list[str] = sorted({
        s.get("role", "worker") for s in stage_specs
    })

    # Determine task shape from sub-agent count.
    if len(sub_agents) <= 1:
        task_shape = "sequential"
    else:
        task_shape = "parallel"

    fingerprint = TaskFingerprint(
        task_shape=task_shape,
        tool_count=len(skills),
        subtask_count=len(stage_specs),
        required_roles=tuple(roles),
        tags=("deerflow",),
    )

    sandbox: str = session_config.get("sandbox", "local")
    recursion_limit: int = session_config.get("recursion_limit", 100)

    return PatternTemplate(
        template_id=uuid4().hex[:8],
        name=f"deerflow_{session_config.get('assistant_id', 'session')}",
        topology="specialist_swarm" if len(stage_specs) > 1 else "single_worker",
        stage_specs=tuple(stage_specs),
        intervention_policy={
            "recursion_limit": recursion_limit,
            "sandbox": sandbox,
        },
        fingerprint=fingerprint,
        tags=("deerflow", sandbox),
    )
