"""Simple public result/config types for the pattern-first API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Protocol

from ..core.epistemic import EpistemicAnalysis, TopologyClass, TopologyRecommendation
from ..memory.bitemporal import BiTemporalFact, BiTemporalQuery
from ..core.types import ApprovalToken
from ..core.wagent import WiringDiagram
from ..core.wiring_runtime import ExecutionReport
from ..organelles.mitochondria import Mitochondria
from ..providers import ProviderConfig
from ..topology.loops import LoopResult


@dataclass(frozen=True)
class ReviewerGateConfig:
    """User-facing configuration for the reviewer-gate wrapper."""

    mode: str = "strict"
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    enable_cache: bool = True
    cache_ttl_seconds: float = 300.0
    timeout_seconds: float = 30.0
    detection_rate: float = 0.75
    comm_cost_ratio: float = 0.4


@dataclass(frozen=True)
class ReviewerGateResult:
    """Simplified result for reviewer-gated execution."""

    allowed: bool
    status: str
    output: Any | None
    reason: str
    approval_token: ApprovalToken | None
    raw: LoopResult


@dataclass(frozen=True)
class SpecialistSwarmConfig:
    """User-facing configuration for the specialist-swarm wrapper."""

    roles: tuple[str, ...]
    aggregation: str = "hub"
    detection_rate: float = 0.75
    comm_cost_ratio: float = 0.4


@dataclass(frozen=True)
class SpecialistSwarmResult:
    """Execution result for a specialist swarm."""

    outputs: dict[str, Any]
    aggregate: Any
    execution_report: ExecutionReport
    diagram: WiringDiagram
    analysis: EpistemicAnalysis


@dataclass(frozen=True)
class TopologyAdvice:
    """Pattern-first topology recommendation."""

    recommended_pattern: str
    suggested_api: str
    topology: TopologyClass
    rationale: str
    raw: TopologyRecommendation
    suggested_template: Any | None = None  # PatternTemplate if library provided


StageHandler = Callable[..., Any]


@dataclass
class SkillStage:
    """
    Declarative stage in a multi-agent skill organism.

    Stages can be deterministic handlers or provider-backed agents bound to a
    named nucleus alias such as ``fast`` or ``deep``.
    """

    name: str
    role: str
    instructions: str = ""
    mode: str = "fuzzy"
    nucleus: str | None = None
    handler: StageHandler | None = None
    tools: Mitochondria | None = None
    provider_config: ProviderConfig | None = None
    include_stage_outputs: bool = True
    include_shared_state: bool = True
    # --- Substrate hooks (Phase 2: bi-temporal integration) ---
    read_query: str | Callable[..., Any] | None = None
    fact_extractor: Callable[..., Any] | None = None
    emit_output_fact: bool = False
    fact_tags: tuple[str, ...] = ()
    # --- Cognitive mode (Phase 5: System A/B) ---
    cognitive_mode: CognitiveMode | None = None


def resolve_cognitive_mode(stage: SkillStage) -> CognitiveMode:
    """Resolve the cognitive mode of a stage, inferring from mode if not set."""
    if stage.cognitive_mode is not None:
        return stage.cognitive_mode
    mode = stage.mode.strip().lower().replace("-", "_")
    if mode in {"fixed", "fast", "deterministic"}:
        return CognitiveMode.OBSERVATIONAL
    return CognitiveMode.ACTION_ORIENTED


@dataclass(frozen=True)
class SubstrateView:
    """Read-only envelope of facts retrieved from the bi-temporal substrate
    before a stage runs.  Stage handlers receive this instead of the raw
    BiTemporalMemory instance, keeping them decoupled from memory internals."""

    facts: tuple[BiTemporalFact, ...]
    query: BiTemporalQuery | str | Callable[..., Any] | None
    record_time: datetime


@dataclass(frozen=True)
class SkillStageResult:
    """Result for a single stage execution."""

    stage_name: str
    role: str
    output: Any
    model_alias: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: float
    action_type: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SkillRunResult:
    """Result for a complete organism skill execution."""

    task: str
    final_output: Any
    stage_results: tuple[SkillStageResult, ...]
    shared_state: dict[str, Any]


@dataclass(frozen=True)
class TelemetryEvent:
    """Structured runtime event emitted by an organism component."""

    kind: str
    stage_name: str | None
    payload: dict[str, Any]


class SkillRuntimeComponent(Protocol):
    """Attachable component interface for skill-organism runtime hooks."""

    def on_run_start(self, task: str, shared_state: dict[str, Any]) -> None:
        """Called once before stage execution begins."""

    def on_stage_start(
        self,
        stage: SkillStage,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """Called immediately before a stage runs."""

    def on_stage_result(
        self,
        stage: SkillStage,
        result: SkillStageResult,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """Called after a stage produces an output."""

    def on_run_complete(self, result: SkillRunResult, shared_state: dict[str, Any]) -> None:
        """Called once after the full organism run completes."""


# --- Watcher intervention types (Phase 3: MASFly integration) ---

WATCHER_STATE_KEY = "_watcher_intervention"


class InterventionKind(Enum):
    """Phase 1 watcher intervention actions."""

    RETRY = "retry"
    ESCALATE = "escalate"
    HALT = "halt"


@dataclass(frozen=True)
class WatcherIntervention:
    """Intervention request produced by the watcher and consumed by the run loop."""

    kind: InterventionKind
    stage_name: str
    reason: str


# --- Cognitive mode types (Phase 5: System A/B) ---


class CognitiveMode(Enum):
    """Dual-process cognitive mode annotation (Dupoux/LeCun/Malik System A/B)."""

    OBSERVATIONAL = "observational"      # System A: passive sensing
    ACTION_ORIENTED = "action_oriented"  # System B: active decision-making
