"""Pattern-first wrappers for common Operon coordination motifs."""

from .advisor import advise_topology
from .organism import SkillOrganism, TelemetryProbe, skill_organism
from .review import ReviewerGate, reviewer_gate
from .swarm import SpecialistSwarm, specialist_swarm
from .types import (
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
)

__all__ = [
    "ReviewerGate",
    "ReviewerGateConfig",
    "ReviewerGateResult",
    "reviewer_gate",
    "SkillOrganism",
    "SkillRunResult",
    "SkillRuntimeComponent",
    "SkillStage",
    "SkillStageResult",
    "SubstrateView",
    "TelemetryEvent",
    "TelemetryProbe",
    "skill_organism",
    "SpecialistSwarm",
    "SpecialistSwarmConfig",
    "SpecialistSwarmResult",
    "specialist_swarm",
    "TopologyAdvice",
    "advise_topology",
]
