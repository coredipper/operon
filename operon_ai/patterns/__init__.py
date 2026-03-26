"""Pattern-first wrappers for common Operon coordination motifs."""

from .adaptive import (
    AdaptiveRunResult,
    AdaptiveSkillOrganism,
    adaptive_skill_organism,
    assemble_pattern,
)
from .advisor import advise_topology
from .cli import CLIResult, cli_handler, cli_organism
from .managed import (
    ManagedOrganism,
    ManagedRunResult,
    managed_organism,
    consolidate,
)
from .organism import SkillOrganism, TelemetryProbe, skill_organism
from .repository import (
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    TaskFingerprint,
)
from .review import ReviewerGate, reviewer_gate
from .swarm import SpecialistSwarm, specialist_swarm
from .types import (
    CognitiveMode,
    InterventionKind,
    ReviewerGateConfig,
    ReviewerGateResult,
    SkillRunResult,
    SkillRuntimeComponent,
    SkillStage,
    SkillStageResult,
    SpecialistSwarmConfig,
    SpecialistSwarmResult,
    SubstrateView,
    TelemetryEvent,
    TopologyAdvice,
    WatcherIntervention,
    resolve_cognitive_mode,
)
from .watcher import (
    ExperienceRecord,
    SignalCategory,
    WatcherComponent,
    WatcherConfig,
    WatcherSignal,
)

__all__ = [
    "AdaptiveRunResult",
    "AdaptiveSkillOrganism",
    "CLIResult",
    "CognitiveMode",
    "ExperienceRecord",
    "InterventionKind",
    "ManagedOrganism",
    "ManagedRunResult",
    "PatternLibrary",
    "PatternRunRecord",
    "PatternTemplate",
    "ReviewerGate",
    "ReviewerGateConfig",
    "ReviewerGateResult",
    "SignalCategory",
    "SkillOrganism",
    "SkillRunResult",
    "SkillRuntimeComponent",
    "SkillStage",
    "SkillStageResult",
    "SubstrateView",
    "TaskFingerprint",
    "TelemetryEvent",
    "TelemetryProbe",
    "WatcherComponent",
    "WatcherConfig",
    "WatcherIntervention",
    "WatcherSignal",
    "adaptive_skill_organism",
    "advise_topology",
    "cli_handler",
    "cli_organism",
    "assemble_pattern",
    "consolidate",
    "managed_organism",
    "resolve_cognitive_mode",
    "reviewer_gate",
    "skill_organism",
    "SpecialistSwarm",
    "SpecialistSwarmConfig",
    "SpecialistSwarmResult",
    "specialist_swarm",
    "TopologyAdvice",
]
