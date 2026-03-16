"""Simple public result/config types for the pattern-first API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..core.epistemic import EpistemicAnalysis, TopologyClass, TopologyRecommendation
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
