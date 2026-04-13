"""Convergence adapters for external agent orchestration systems.

This package provides type-level bridges between Operon's structural analysis
and external orchestration runtimes (Swarms, DeerFlow, AnimaWorks, Ralph,
A-Evolve, Scion). All
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
from .async_thinking import (
    AsyncOrganizer,
    AsyncThinkResult,
    async_stage_handler,
)
from .aevolve_adapter import (
    aevolve_skills_to_stages,
    aevolve_to_template,
    parse_aevolve_workspace,
)
from .aevolve_skills import (
    import_aevolve_skills,
    seed_library_from_aevolve,
)
from .catalog import (
    get_builtin_swarms_patterns,
    seed_library_from_acg_survey,
    seed_library_from_deerflow,
    seed_library_from_ralph,
    seed_library_from_swarms,
)
from .deerflow_adapter import (
    deerflow_skills_to_stages,
    deerflow_to_template,
    parse_deerflow_session,
)
from .ralph_adapter import (
    parse_ralph_config,
    ralph_hats_to_stages,
    ralph_to_template,
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
from .codesign import (
    DesignProblem,
    compose_parallel,
    compose_series,
    feasibility_check,
    feedback_fixed_point,
)
from .prompt_optimization import (
    EvolutionaryOptimizer,
    NoOpOptimizer,
    PromptOptimizer,
    attach_optimizer,
)
from .workflow_generation import (
    HeuristicGenerator,
    ReasoningGenerator,
    WorkflowGenerator,
    generate_and_register,
)
from .deerflow_compiler import (
    deerflow_to_topology,
    managed_to_deerflow,
    organism_to_deerflow,
)
from .deerflow_executor import (
    DeerFlowResult,
    HAS_DEERFLOW,
    execute_deerflow,
)
from .distributed_watcher import (
    DistributedWatcher,
    HttpTransport,
    InMemoryTransport,
)
from .langgraph_watcher import (
    create_watcher_config,
    operon_watcher_node,
)
from .ralph_compiler import (
    managed_to_ralph,
    organism_to_ralph,
)
from .scion_adapter import (
    parse_scion_grove,
    scion_agents_to_stages,
    scion_to_template,
)
from .scion_compiler import (
    managed_to_scion,
    organism_to_scion,
)
from .swarms_compiler import (
    managed_to_swarms,
    organism_to_swarms,
    swarms_to_topology,
)
from .categorical import (
    Architecture,
    CompilationResult,
    CompilerFunctor,
    PreservationResult,
    extract_architecture,
    extract_compiled_architecture,
    swarms_functor,
    deerflow_functor,
    ralph_functor,
    scion_functor,
)
from .memory_bridge import (
    bridge_animaworks_memory,
    bridge_deerflow_memory,
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
    # Shared analysis (C1)
    "analyze_external_topology",
    "topology_to_template",
    # Swarms (C1)
    "parse_swarm_topology",
    "swarm_to_template",
    # AnimaWorks (C1)
    "parse_animaworks_org",
    "animaworks_roles_to_stages",
    "animaworks_to_template",
    # DeerFlow (C1)
    "parse_deerflow_session",
    "deerflow_skills_to_stages",
    "deerflow_to_template",
    # Ralph (C1)
    "parse_ralph_config",
    "ralph_hats_to_stages",
    "ralph_to_template",
    # A-Evolve (C1)
    "parse_aevolve_workspace",
    "aevolve_skills_to_stages",
    "aevolve_to_template",
    # Scion (C1)
    "parse_scion_grove",
    "scion_agents_to_stages",
    "scion_to_template",
    # Catalog (C2)
    "seed_library_from_swarms",
    "seed_library_from_deerflow",
    "seed_library_from_ralph",
    "seed_library_from_acg_survey",
    "get_builtin_swarms_patterns",
    # A-Evolve skill bridge (C2)
    "import_aevolve_skills",
    "seed_library_from_aevolve",
    # DeerFlow skill bridge (C2)
    "skill_to_template",
    "template_to_skill",
    "parse_skill_frontmatter",
    "extract_workflow_steps",
    # Hybrid assembly (C2)
    "hybrid_skill_organism",
    "default_template_generator",
    # AsyncThink Fork/Join (C3)
    "AsyncOrganizer",
    "AsyncThinkResult",
    "async_stage_handler",
    # Co-design (C4)
    "DesignProblem",
    "compose_series",
    "compose_parallel",
    "feedback_fixed_point",
    "feasibility_check",
    # Memory bridge (C3)
    "bridge_animaworks_memory",
    "bridge_deerflow_memory",
    # Compilers (C5)
    "organism_to_swarms",
    "managed_to_swarms",
    "organism_to_deerflow",
    "managed_to_deerflow",
    "organism_to_ralph",
    "managed_to_ralph",
    "organism_to_scion",
    "managed_to_scion",
    # Decompilers (C5)
    "deerflow_to_topology",
    "swarms_to_topology",
    # DeerFlow executor (C5)
    "DeerFlowResult",
    "HAS_DEERFLOW",
    "execute_deerflow",
    # Distributed watcher (C5)
    "DistributedWatcher",
    "InMemoryTransport",
    "HttpTransport",
    # LangGraph watcher (C5)
    "operon_watcher_node",
    "create_watcher_config",
    # Categorical formalization (C5+)
    "Architecture",
    "CompilationResult",
    "CompilerFunctor",
    "PreservationResult",
    "extract_architecture",
    "extract_compiled_architecture",
    "swarms_functor",
    "deerflow_functor",
    "ralph_functor",
    "scion_functor",
    # Prompt optimization (C7)
    "PromptOptimizer",
    "EvolutionaryOptimizer",
    "NoOpOptimizer",
    "attach_optimizer",
    # Workflow generation (C7)
    "WorkflowGenerator",
    "ReasoningGenerator",
    "HeuristicGenerator",
    "generate_and_register",
]
