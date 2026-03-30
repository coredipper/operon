"""Ralph convergence adapter -- parse Ralph hat configs into Operon types.

Ralph organises agents as "hats" connected by event-driven transitions with
backpressure constraints and iteration limits.  This module converts those
plain-dict configs into :class:`ExternalTopology`, :class:`SkillStage` tuples,
and :class:`PatternTemplate` instances without importing ralph-orchestrator.

All inputs are plain dicts; no Ralph imports.
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

from ..convergence.types import ExternalTopology
from ..patterns.repository import PatternLibrary, PatternTemplate, TaskFingerprint
from ..patterns.types import CognitiveMode, SkillStage

# ---------------------------------------------------------------------------
# Hat pattern -> cognitive-mode mapping
# ---------------------------------------------------------------------------

_ACTION_PATTERNS: frozenset[str] = frozenset({
    "code-assist",
    "debug",
    "pdd-to-code-assist",
})

_OBSERVATIONAL_PATTERNS: frozenset[str] = frozenset({
    "review",
    "research",
})


def _pattern_to_mode(pattern: str) -> tuple[str, CognitiveMode]:
    """Return (mode_string, CognitiveMode) for a Ralph hat pattern."""
    pat = pattern.strip().lower()
    if pat in _OBSERVATIONAL_PATTERNS:
        return "fixed", CognitiveMode.OBSERVATIONAL
    if pat in _ACTION_PATTERNS:
        return "fuzzy", CognitiveMode.ACTION_ORIENTED
    # Unknown patterns default to fuzzy / ACTION_ORIENTED.
    return "fuzzy", CognitiveMode.ACTION_ORIENTED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_ralph_config(config: dict) -> ExternalTopology:
    """Parse a Ralph config dict into an :class:`ExternalTopology`.

    Parameters
    ----------
    config:
        Plain dict with shape::

            {
                "backend": "claude",
                "hats": [
                    {"name": "coder", "pattern": "code-assist"},
                    {"name": "debugger", "pattern": "debug"},
                    {"name": "reviewer", "pattern": "review"},
                ],
                "events": [
                    {"from": "coder", "event": "code.failure", "to": "debugger"},
                    {"from": "debugger", "event": "fix.complete", "to": "reviewer"},
                ],
                "backpressure": ["tests", "lint", "typecheck"],
                "iteration_limit": 10,
            }

    Returns
    -------
    ExternalTopology
        Operon-native topology with hats as agents and events as edges.
    """
    hats: list[dict] = config.get("hats", [])
    events: list[dict] = config.get("events", [])
    backend: str = config.get("backend", "unknown")
    backpressure: list[str] = config.get("backpressure", [])
    iteration_limit: int = config.get("iteration_limit", 10)

    # Build agent dicts -- one per hat.
    agents: list[dict[str, Any]] = []
    for hat in hats:
        agents.append({
            "name": hat.get("name", "unnamed"),
            "role": hat.get("pattern", "worker"),
        })

    # Build edges from events.
    edges: list[tuple[str, str]] = [
        (ev["from"], ev["to"]) for ev in events
    ]

    return ExternalTopology(
        source="ralph",
        pattern_name="RalphHatGraph",
        agents=tuple(agents),
        edges=tuple(edges),
        metadata={
            "backend": backend,
            "backpressure": backpressure,
            "iteration_limit": iteration_limit,
            "_ralph_config": copy.deepcopy(config),
        },
    )


def ralph_hats_to_stages(hats: list[dict]) -> tuple[SkillStage, ...]:
    """Map Ralph hat dicts to Operon :class:`SkillStage` objects.

    Parameters
    ----------
    hats:
        List of hat dicts, each with ``name`` and ``pattern`` keys.

    Returns
    -------
    tuple[SkillStage, ...]
        One stage per hat, with ``mode`` and ``cognitive_mode`` derived from
        the hat pattern.
    """
    stages: list[SkillStage] = []
    for hat in hats:
        name: str = hat.get("name", "unnamed")
        pattern: str = hat.get("pattern", "unknown")

        mode_str, cog_mode = _pattern_to_mode(pattern)

        stages.append(SkillStage(
            name=name,
            role=pattern,
            instructions=f"Ralph hat: {pattern}",
            mode=mode_str,
            cognitive_mode=cog_mode,
        ))

    return tuple(stages)


def ralph_to_template(config: dict) -> PatternTemplate:
    """Convert a Ralph config to an Operon :class:`PatternTemplate`.

    The resulting template captures the event-driven topology, hat stage
    specs, and a :class:`TaskFingerprint` derived from the config structure.

    Parameters
    ----------
    config:
        Same shape as for :func:`parse_ralph_config`.

    Returns
    -------
    PatternTemplate
        Ready for registration in a :class:`PatternLibrary`.
    """
    hats: list[dict] = config.get("hats", [])
    events: list[dict] = config.get("events", [])
    backpressure: list[str] = config.get("backpressure", [])
    iteration_limit: int = config.get("iteration_limit", 10)

    # Build stage specs from hats.
    stage_specs: list[dict[str, Any]] = []
    for hat in hats:
        pattern = hat.get("pattern", "worker")
        mode_str, _ = _pattern_to_mode(pattern)
        stage_specs.append({
            "name": hat.get("name", "unnamed"),
            "role": pattern,
            "mode": mode_str,
        })

    # Derive roles.
    roles = tuple(sorted({s["role"] for s in stage_specs})) if stage_specs else ()

    # Determine topology from event structure.
    n_hats = len(hats)
    n_events = len(events)
    if n_hats <= 1:
        topology = "single_worker"
        task_shape = "sequential"
    elif n_events == 0:
        topology = "specialist_swarm"
        task_shape = "parallel"
    elif n_events == n_hats - 1:
        # Linear chain of events -> skill organism.
        topology = "skill_organism"
        task_shape = "sequential"
    else:
        topology = "specialist_swarm"
        task_shape = "mixed"

    fingerprint = TaskFingerprint(
        task_shape=task_shape,
        tool_count=len(backpressure),
        subtask_count=n_hats,
        required_roles=roles,
        tags=("ralph",),
    )

    return PatternTemplate(
        template_id=PatternLibrary.make_id(),
        name=f"ralph_{config.get('backend', 'unknown')}",
        topology=topology,
        stage_specs=tuple(stage_specs),
        intervention_policy={
            "iteration_limit": iteration_limit,
            "backpressure": backpressure,
        },
        fingerprint=fingerprint,
        tags=("ralph", config.get("backend", "unknown")),
    )
