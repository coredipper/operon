"""Convergence catalog -- seed a PatternLibrary from external frameworks.

Provides functions to populate an Operon :class:`PatternLibrary` with
templates derived from Swarms, DeerFlow, the ACG survey taxonomy, and
the atomic coding skills taxonomy (Ma et al., arXiv:2604.05013).
Each seeder follows a common pipeline:

1. Parse external config into :class:`ExternalTopology`.
2. Analyse via :func:`analyze_external_topology`.
3. Register the resulting :class:`PatternTemplate` in the library.

This module builds on the Phase C1 adapters and is intended to bootstrap
a library with a diverse set of collaboration patterns before any
runtime data has been collected.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..patterns.repository import PatternLibrary, PatternTemplate, TaskFingerprint
from .deerflow_adapter import deerflow_to_template, parse_deerflow_session
from .ralph_adapter import parse_ralph_config, ralph_to_template
from .swarms_adapter import (
    analyze_external_topology,
    parse_swarm_topology,
    topology_to_template,
)

# ---------------------------------------------------------------------------
# Swarms seeder
# ---------------------------------------------------------------------------


def seed_library_from_swarms(
    library: PatternLibrary,
    patterns: list[dict[str, Any]],
) -> int:
    """Seed *library* with Swarms workflow patterns.

    Parameters
    ----------
    library:
        Target :class:`PatternLibrary` to populate.
    patterns:
        Each dict has ``name``, ``agents``, and ``edges`` keys matching
        the shape consumed by :func:`parse_swarm_topology`.

    Returns
    -------
    int
        Number of templates successfully registered.
    """
    count = 0
    for pat in patterns:
        name: str = pat["name"]
        agents: list[dict[str, Any]] = pat["agents"]
        edges: list[tuple[str, str]] = pat.get("edges", [])

        topology = parse_swarm_topology(name, agents, edges)
        result = analyze_external_topology(topology)

        template = result.suggested_template if result.suggested_template is not None else topology_to_template(topology)
        library.register_template(template)
        count += 1

    return count


# ---------------------------------------------------------------------------
# DeerFlow seeder
# ---------------------------------------------------------------------------


def seed_library_from_deerflow(
    library: PatternLibrary,
    sessions: list[dict[str, Any]],
) -> int:
    """Seed *library* with DeerFlow session patterns.

    Parameters
    ----------
    library:
        Target :class:`PatternLibrary` to populate.
    sessions:
        Each dict matches the shape expected by :func:`parse_deerflow_session`.

    Returns
    -------
    int
        Number of templates successfully registered.
    """
    count = 0
    for session in sessions:
        topology = parse_deerflow_session(session)
        result = analyze_external_topology(topology)

        template = result.suggested_template if result.suggested_template is not None else deerflow_to_template(session)
        library.register_template(template)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Ralph seeder
# ---------------------------------------------------------------------------


def seed_library_from_ralph(
    library: PatternLibrary,
    configs: list[dict[str, Any]],
) -> int:
    """Seed *library* with Ralph hat-based workflow patterns.

    Parameters
    ----------
    library:
        Target :class:`PatternLibrary` to populate.
    configs:
        Each dict matches the shape expected by :func:`parse_ralph_config`.

    Returns
    -------
    int
        Number of templates successfully registered.
    """
    count = 0
    for config in configs:
        topology = parse_ralph_config(config)
        result = analyze_external_topology(topology)

        template = result.suggested_template if result.suggested_template is not None else ralph_to_template(config)
        library.register_template(template)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Atomic coding skills (Ma et al., arXiv:2604.05013)
# ---------------------------------------------------------------------------

# Five atomic skills that compose without negative interference.
# Each maps to a sequential or parallel SkillStage pipeline.
_ATOMIC_SKILLS: list[dict[str, Any]] = [
    {
        "name": "localize",
        "description": "Find relevant code locations for a given issue or task.",
        "task_shape": "sequential",
        "tool_count": 2,
        "subtask_count": 2,
        "roles": ("searcher", "ranker"),
        "tags": ("code", "localization", "atomic_skill"),
    },
    {
        "name": "edit",
        "description": "Modify source code to implement a change or fix.",
        "task_shape": "sequential",
        "tool_count": 2,
        "subtask_count": 2,
        "roles": ("editor", "validator"),
        "tags": ("code", "editing", "atomic_skill"),
    },
    {
        "name": "test",
        "description": "Generate and run unit tests for code changes.",
        "task_shape": "sequential",
        "tool_count": 3,
        "subtask_count": 3,
        "roles": ("analyzer", "generator", "runner"),
        "tags": ("code", "testing", "atomic_skill"),
    },
    {
        "name": "reproduce",
        "description": "Reproduce a reported issue to confirm its existence.",
        "task_shape": "sequential",
        "tool_count": 2,
        "subtask_count": 3,
        "roles": ("reader", "executor", "verifier"),
        "tags": ("code", "reproduction", "atomic_skill"),
    },
    {
        "name": "review",
        "description": "Review code for correctness, style, security, and performance.",
        "task_shape": "parallel",
        "tool_count": 4,
        "subtask_count": 4,
        "roles": ("logic_auditor", "style_checker", "security_auditor", "reporter"),
        "tags": ("code", "review", "atomic_skill"),
    },
]


def get_atomic_skill_patterns() -> list[dict[str, Any]]:
    """Return the 5 atomic coding skill patterns (Ma et al. 2604.05013).

    These skills are composable basis vectors for software engineering
    tasks: localize, edit, test, reproduce, review.

    Returns deep copies so the built-in catalog cannot be mutated.
    """
    import copy
    return copy.deepcopy(_ATOMIC_SKILLS)


def seed_library_from_atomic_skills(
    library: PatternLibrary,
    patterns: list[dict[str, Any]] | None = None,
) -> int:
    """Seed *library* with atomic coding skill patterns.

    Uses the built-in 5-skill catalog by default.  Pass *patterns*
    to override with custom skill definitions.

    Parameters
    ----------
    library:
        Target :class:`PatternLibrary` to populate.
    patterns:
        Override skill definitions.  If ``None``, uses the built-in
        catalog from Ma et al. (arXiv:2604.05013).

    Returns
    -------
    int
        Number of templates successfully registered.
    """
    skills = patterns if patterns is not None else _ATOMIC_SKILLS
    count = 0
    for skill in skills:
        fingerprint = TaskFingerprint(
            task_shape=skill["task_shape"],
            tool_count=skill["tool_count"],
            subtask_count=skill["subtask_count"],
            required_roles=tuple(skill["roles"]),
            tags=tuple(skill["tags"]),
        )

        stage_specs = tuple(
            {"name": role, "role": role, "mode": "fuzzy"}
            for role in skill["roles"]
        )

        template = PatternTemplate(
            template_id=uuid4().hex[:8],
            name=f"atomic_{skill['name']}",
            topology="skill_organism",
            stage_specs=stage_specs,
            intervention_policy={"mode": "default"},
            fingerprint=fingerprint,
            tags=tuple(skill["tags"]),
        )

        library.register_template(template)
        count += 1

    return count


# ---------------------------------------------------------------------------
# ACG survey seeder
# ---------------------------------------------------------------------------

# Each entry: (name, task_shape, topology, subtask_count, tool_count, roles,
#              graph_determination_time, graph_plasticity_mode, description)
_ACG_CATALOG: list[dict[str, Any]] = [
    {
        "name": "offline_template_search",
        "task_shape": "mixed",
        "topology": "specialist_swarm",
        "subtask_count": 4,
        "tool_count": 2,
        "roles": ("planner", "searcher", "evaluator"),
        "graph_determination_time": "static",
        "graph_plasticity_mode": "graph-level",
        "description": "MCTS-based offline search over workflow templates.",
    },
    {
        "name": "node_optimization",
        "task_shape": "sequential",
        "topology": "skill_organism",
        "subtask_count": 3,
        "tool_count": 1,
        "roles": ("optimizer", "worker"),
        "graph_determination_time": "static",
        "graph_plasticity_mode": "node-level",
        "description": "DSPy-style per-node prompt/weight optimization.",
    },
    {
        "name": "joint_optimization",
        "task_shape": "mixed",
        "topology": "specialist_swarm",
        "subtask_count": 5,
        "tool_count": 3,
        "roles": ("optimizer", "coordinator", "worker"),
        "graph_determination_time": "static",
        "graph_plasticity_mode": "joint",
        "description": "MASS-style joint optimisation of topology and prompts.",
    },
    {
        "name": "selection_pruning",
        "task_shape": "parallel",
        "topology": "specialist_swarm",
        "subtask_count": 4,
        "tool_count": 2,
        "roles": ("selector", "worker"),
        "graph_determination_time": "dynamic",
        "graph_plasticity_mode": "select",
        "description": "AgentDropout-style dynamic selection and pruning.",
    },
    {
        "name": "construct_then_execute",
        "task_shape": "sequential",
        "topology": "skill_organism",
        "subtask_count": 3,
        "tool_count": 2,
        "roles": ("planner", "executor"),
        "graph_determination_time": "dynamic",
        "graph_plasticity_mode": "generate",
        "description": "WorkflowLLM-style: generate graph then execute it.",
    },
    {
        "name": "in_execution_editing",
        "task_shape": "mixed",
        "topology": "specialist_swarm",
        "subtask_count": 4,
        "tool_count": 3,
        "roles": ("editor", "executor", "monitor"),
        "graph_determination_time": "dynamic",
        "graph_plasticity_mode": "edit",
        "description": "DyFlow-style runtime graph editing during execution.",
    },
    {
        "name": "verifier_driven",
        "task_shape": "sequential",
        "topology": "reviewer_gate",
        "subtask_count": 3,
        "tool_count": 1,
        "roles": ("generator", "verifier"),
        "graph_determination_time": "static",
        "graph_plasticity_mode": "verifier-loop",
        "description": "MermaidFlow-style verifier loop for iterative refinement.",
    },
    {
        "name": "preference_driven",
        "task_shape": "parallel",
        "topology": "specialist_swarm",
        "subtask_count": 4,
        "tool_count": 2,
        "roles": ("ranker", "generator", "worker"),
        "graph_determination_time": "dynamic",
        "graph_plasticity_mode": "preference",
        "description": "Ranking-based preference-driven topology selection.",
    },
]


def seed_library_from_acg_survey(library: PatternLibrary) -> int:
    """Register built-in ACG survey taxonomy templates in *library*.

    Populates the library with ~8 templates representing key method
    categories from the Agentic Composition Graph survey.

    Returns
    -------
    int
        Number of templates registered.
    """
    count = 0
    for entry in _ACG_CATALOG:
        fingerprint = TaskFingerprint(
            task_shape=entry["task_shape"],
            tool_count=entry["tool_count"],
            subtask_count=entry["subtask_count"],
            required_roles=tuple(entry["roles"]),
            tags=(
                "acg_survey",
                entry["name"],
                f"determination:{entry['graph_determination_time']}",
                f"plasticity:{entry['graph_plasticity_mode']}",
            ),
        )

        # Build minimal stage specs from the roles.
        stage_specs = tuple(
            {"name": role, "role": role, "mode": "fuzzy"}
            for role in entry["roles"]
        )

        template = PatternTemplate(
            template_id=uuid4().hex[:8],
            name=f"acg_{entry['name']}",
            topology=entry["topology"],
            stage_specs=stage_specs,
            intervention_policy={"mode": "default"},
            fingerprint=fingerprint,
            tags=(
                "acg_survey",
                entry["name"],
                f"determination:{entry['graph_determination_time']}",
                f"plasticity:{entry['graph_plasticity_mode']}",
            ),
        )

        library.register_template(template)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Built-in Swarms pattern catalog
# ---------------------------------------------------------------------------


def get_builtin_swarms_patterns() -> list[dict[str, Any]]:
    """Return ~10 representative Swarms patterns as plain dicts.

    These are hardcoded representations of Swarms' built-in workflow
    patterns, useful for seeding a :class:`PatternLibrary` without
    requiring the Swarms package.
    """
    return [
        {
            "name": "SequentialWorkflow",
            "agents": [
                {"name": "agent_1", "role": "worker", "skills": ["text_generation"]},
                {"name": "agent_2", "role": "worker", "skills": ["summarization"]},
                {"name": "agent_3", "role": "worker", "skills": ["review"]},
            ],
            "edges": [("agent_1", "agent_2"), ("agent_2", "agent_3")],
        },
        {
            "name": "ConcurrentWorkflow",
            "agents": [
                {"name": "worker_a", "role": "worker", "skills": ["web_search"]},
                {"name": "worker_b", "role": "worker", "skills": ["code_generation"]},
                {"name": "worker_c", "role": "worker", "skills": ["data_analysis"]},
                {"name": "worker_d", "role": "worker", "skills": ["summarization"]},
            ],
            "edges": [],
        },
        {
            "name": "HierarchicalSwarm",
            "agents": [
                {"name": "manager", "role": "manager", "skills": ["planning", "delegation"]},
                {"name": "worker_1", "role": "worker", "skills": ["research"]},
                {"name": "worker_2", "role": "worker", "skills": ["writing"]},
                {"name": "worker_3", "role": "worker", "skills": ["coding"]},
            ],
            "edges": [
                ("manager", "worker_1"),
                ("manager", "worker_2"),
                ("manager", "worker_3"),
            ],
        },
        {
            "name": "GraphWorkflow",
            "agents": [
                {"name": "entry", "role": "coordinator", "skills": ["routing"]},
                {"name": "analyst", "role": "analyst", "skills": ["data_analysis"]},
                {"name": "writer", "role": "writer", "skills": ["text_generation"]},
                {"name": "reviewer", "role": "reviewer", "skills": ["review"]},
            ],
            "edges": [
                ("entry", "analyst"),
                ("entry", "writer"),
                ("analyst", "reviewer"),
                ("writer", "reviewer"),
            ],
        },
        {
            "name": "MixtureOfAgents",
            "agents": [
                {"name": "proposer_1", "role": "proposer", "skills": ["generation"]},
                {"name": "proposer_2", "role": "proposer", "skills": ["generation"]},
                {"name": "aggregator", "role": "aggregator", "skills": ["synthesis"]},
            ],
            "edges": [
                ("proposer_1", "aggregator"),
                ("proposer_2", "aggregator"),
            ],
        },
        {
            "name": "AgentRearrange",
            "agents": [
                {"name": "alpha", "role": "worker", "skills": ["text_generation"]},
                {"name": "beta", "role": "worker", "skills": ["code_generation"]},
                {"name": "gamma", "role": "worker", "skills": ["review"]},
            ],
            "edges": [("alpha", "beta"), ("beta", "gamma")],
        },
        {
            "name": "SpreadSheetSwarm",
            "agents": [
                {"name": "cell_a", "role": "processor", "skills": ["computation"]},
                {"name": "cell_b", "role": "processor", "skills": ["computation"]},
                {"name": "cell_c", "role": "processor", "skills": ["computation"]},
                {"name": "cell_d", "role": "processor", "skills": ["computation"]},
            ],
            "edges": [],
        },
        {
            "name": "GroupChat",
            "agents": [
                {"name": "participant_1", "role": "participant", "skills": ["dialogue"]},
                {"name": "participant_2", "role": "participant", "skills": ["dialogue"]},
                {"name": "participant_3", "role": "participant", "skills": ["dialogue"]},
            ],
            "edges": [
                ("participant_1", "participant_2"),
                ("participant_1", "participant_3"),
                ("participant_2", "participant_1"),
                ("participant_2", "participant_3"),
                ("participant_3", "participant_1"),
                ("participant_3", "participant_2"),
            ],
        },
        {
            "name": "MultiAgentRouter",
            "agents": [
                {"name": "router", "role": "router", "skills": ["classification", "routing"]},
                {"name": "specialist_a", "role": "specialist", "skills": ["math"]},
                {"name": "specialist_b", "role": "specialist", "skills": ["coding"]},
                {"name": "specialist_c", "role": "specialist", "skills": ["writing"]},
            ],
            "edges": [
                ("router", "specialist_a"),
                ("router", "specialist_b"),
                ("router", "specialist_c"),
            ],
        },
        {
            "name": "ForestSwarm",
            "agents": [
                {"name": "tree1_root", "role": "coordinator", "skills": ["planning"]},
                {"name": "tree1_leaf", "role": "worker", "skills": ["execution"]},
                {"name": "tree2_root", "role": "coordinator", "skills": ["planning"]},
                {"name": "tree2_leaf", "role": "worker", "skills": ["execution"]},
            ],
            "edges": [
                ("tree1_root", "tree1_leaf"),
                ("tree2_root", "tree2_leaf"),
            ],
        },
    ]
