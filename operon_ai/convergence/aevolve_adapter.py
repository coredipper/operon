"""A-Evolve workspace adapter -- parse A-Evolve manifests into Operon types.

A-Evolve organises single-agent workspaces with skill sets, episodic/semantic
memory, and evolutionary algorithms.  This module converts those plain-dict
manifests into :class:`ExternalTopology`, :class:`SkillStage` tuples, and
:class:`PatternTemplate` instances without importing a-evolve.

All inputs are plain dicts; no A-Evolve imports.
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

from ..convergence.types import ExternalTopology
from ..patterns.repository import PatternLibrary, PatternTemplate, TaskFingerprint
from ..patterns.types import CognitiveMode, SkillStage

# ---------------------------------------------------------------------------
# Skill category -> cognitive-mode mapping
# ---------------------------------------------------------------------------

_OBSERVATIONAL_CATEGORIES: frozenset[str] = frozenset({
    "research",
    "verification",
    "review",
    "observation",
})

_ACTION_CATEGORIES: frozenset[str] = frozenset({
    "execution",
    "generation",
    "writing",
    "code",
    "editing",
})


def _category_to_mode(category: str) -> tuple[str, CognitiveMode]:
    """Return (mode_string, CognitiveMode) for an A-Evolve skill category."""
    cat = category.strip().lower()
    if cat in _OBSERVATIONAL_CATEGORIES:
        return "fixed", CognitiveMode.OBSERVATIONAL
    # Action-oriented or unknown both map to fuzzy / ACTION_ORIENTED.
    return "fuzzy", CognitiveMode.ACTION_ORIENTED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_aevolve_workspace(manifest: dict) -> ExternalTopology:
    """Parse an A-Evolve workspace manifest into an :class:`ExternalTopology`.

    Parameters
    ----------
    manifest:
        Plain dict with shape::

            {
                "name": "swe-agent",
                "entrypoints": {"solve": "solve.py"},
                "skills": ["bash_exec", "file_edit", "git_ops"],
                "memory": {
                    "episodic": "memory/episodic.jsonl",
                    "semantic": "memory/semantic.jsonl",
                },
                "evolution": {
                    "algorithm": "adaptive_evolve",
                    "gate": "holdout",
                },
            }

    Returns
    -------
    ExternalTopology
        Single-agent topology representing the workspace.
    """
    name: str = manifest.get("name", "aevolve_workspace")
    skills: list[str] = manifest.get("skills", [])
    entrypoints: dict = manifest.get("entrypoints", {})
    memory: dict = manifest.get("memory", {})
    evolution: dict = manifest.get("evolution", {})

    # Single agent -- the workspace itself.
    agent: dict[str, Any] = {
        "name": name,
        "role": "workspace",
        "capabilities": list(skills),
    }

    return ExternalTopology(
        source="aevolve",
        pattern_name="AEvolveWorkspace",
        agents=(agent,),
        edges=(),  # Single agent, no edges.
        metadata={
            "skills": skills,
            "entrypoints": entrypoints,
            "memory": memory,
            "evolution": evolution,
            "_aevolve_manifest": copy.deepcopy(manifest),
        },
    )


def aevolve_skills_to_stages(skills: list[dict]) -> tuple[SkillStage, ...]:
    """Map A-Evolve skill dicts to Operon :class:`SkillStage` objects.

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


def aevolve_to_template(manifest: dict) -> PatternTemplate:
    """Convert an A-Evolve workspace manifest to a :class:`PatternTemplate`.

    Single-agent workspace maps to ``"single_worker"`` topology.

    Parameters
    ----------
    manifest:
        Same shape as for :func:`parse_aevolve_workspace`.

    Returns
    -------
    PatternTemplate
        Ready for registration in a :class:`PatternLibrary`.
    """
    name: str = manifest.get("name", "aevolve_workspace")
    skills: list[str] = manifest.get("skills", [])
    evolution: dict = manifest.get("evolution", {})

    # Single stage spec for the workspace agent.
    stage_specs: list[dict[str, Any]] = [{
        "name": name,
        "role": "workspace",
        "skills": list(skills),
    }]

    fingerprint = TaskFingerprint(
        task_shape="sequential",
        tool_count=len(skills),
        subtask_count=1,
        required_roles=("workspace",),
        tags=("aevolve",),
    )

    algorithm = evolution.get("algorithm", "unknown")
    gate = evolution.get("gate", "unknown")

    return PatternTemplate(
        template_id=PatternLibrary.make_id(),
        name=f"aevolve_{name}",
        topology="single_worker",
        stage_specs=tuple(stage_specs),
        intervention_policy={
            "evolution_algorithm": algorithm,
            "evolution_gate": gate,
        },
        fingerprint=fingerprint,
        tags=("aevolve", algorithm),
    )
