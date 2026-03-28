"""Convergence adapters for external agent orchestration systems.

This package provides type-level bridges between Operon's structural analysis
and external orchestration runtimes (Swarms, DeerFlow, AnimaWorks). All
adapters operate on serializable dict/JSON representations — they never import
external frameworks, keeping Operon dependency-free.

The core abstraction is :class:`ExternalTopology`, which all adapters produce
and :func:`analyze_external_topology` consumes.
"""

from .animaworks_adapter import (
    animaworks_roles_to_stages,
    animaworks_to_template,
    parse_animaworks_org,
)
from .catalog import (
    get_builtin_swarms_patterns,
    seed_library_from_acg_survey,
    seed_library_from_deerflow,
    seed_library_from_swarms,
)
from .deerflow_adapter import (
    deerflow_skills_to_stages,
    deerflow_to_template,
    parse_deerflow_session,
)
from .deerflow_skills import (
    extract_workflow_steps,
    parse_skill_frontmatter,
    skill_to_template,
    template_to_skill,
)
from .hybrid_assembly import (
    default_template_generator,
    hybrid_skill_organism,
)
from .swarms_adapter import (
    analyze_external_topology,
    parse_swarm_topology,
    swarm_to_template,
    topology_to_template,
)
from .types import AdapterResult, ExternalTopology, RuntimeConfig

__all__ = [
    # Types
    "AdapterResult",
    "ExternalTopology",
    "RuntimeConfig",
    # Shared analysis
    "analyze_external_topology",
    "topology_to_template",
    # Swarms
    "parse_swarm_topology",
    "swarm_to_template",
    # AnimaWorks
    "parse_animaworks_org",
    "animaworks_roles_to_stages",
    "animaworks_to_template",
    # DeerFlow
    "parse_deerflow_session",
    "deerflow_skills_to_stages",
    "deerflow_to_template",
    # Catalog (C2)
    "seed_library_from_swarms",
    "seed_library_from_deerflow",
    "seed_library_from_acg_survey",
    "get_builtin_swarms_patterns",
    # DeerFlow skill bridge (C2)
    "skill_to_template",
    "template_to_skill",
    "parse_skill_frontmatter",
    "extract_workflow_steps",
    # Hybrid assembly (C2)
    "hybrid_skill_organism",
    "default_template_generator",
]
