"""Pattern-first wrappers for common Operon coordination motifs."""

from .advisor import advise_topology
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
)
from .watcher import (
    SignalCategory,
    WatcherComponent,
    WatcherConfig,
    WatcherSignal,
)

__all__ = [
    "InterventionKind",
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
    "advise_topology",
    "reviewer_gate",
    "skill_organism",
    "SpecialistSwarm",
    "SpecialistSwarmConfig",
    "SpecialistSwarmResult",
    "specialist_swarm",
    "TopologyAdvice",
]
