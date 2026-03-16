"""Pattern-first wrapper for reviewer-gated execution."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from time import time
from typing import Any, Callable

from ..core.epistemic import EpistemicAnalysis, analyze as epistemic_analyze
from ..core.types import ActionProtein, DataType, IntegrityLabel, Signal
from ..core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram
from ..state.metabolism import ATP_Store
from ..topology.loops import (
    CoherentFeedForwardLoop,
    GateLogic,
    LoopResult,
)
from .types import ReviewerGateConfig, ReviewerGateResult

ExecutorFn = Callable[..., Any]
ReviewerFn = Callable[..., Any]


def _normalize_mode(mode: str) -> GateLogic:
    normalized = mode.strip().lower().replace("-", "_")
    mapping = {
        "strict": GateLogic.AND,
        "and": GateLogic.AND,
        "unanimous": GateLogic.UNANIMOUS,
        "permissive": GateLogic.OR,
        "or": GateLogic.OR,
        "executor_priority": GateLogic.EXECUTOR_PRIORITY,
        "reviewer_priority": GateLogic.ASSESSOR_PRIORITY,
        "assessor_priority": GateLogic.ASSESSOR_PRIORITY,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported reviewer gate mode: {mode}") from exc


def _call_arity(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        params = list(signature(fn).parameters.values())
    except (TypeError, ValueError):
        return fn(*args)
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return fn(*args)
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return fn(*args[:len(positional)])


def _coerce_executor_output(value: Any) -> ActionProtein:
    if isinstance(value, ActionProtein):
        return value
    if isinstance(value, dict) and "action_type" in value:
        return ActionProtein(
            action_type=str(value["action_type"]),
            payload=value.get("payload"),
            confidence=float(value.get("confidence", 1.0)),
            source_agent=value.get("source_agent"),
            metadata=dict(value.get("metadata", {})),
        )
    if isinstance(value, bool):
        if value:
            return ActionProtein("EXECUTE", "Approved by executor", 1.0)
        return ActionProtein("BLOCK", "Rejected by executor", 1.0)
    return ActionProtein("EXECUTE", value, 1.0)


def _coerce_reviewer_output(value: Any) -> ActionProtein:
    if isinstance(value, ActionProtein):
        return value
    if isinstance(value, dict) and "action_type" in value:
        return ActionProtein(
            action_type=str(value["action_type"]),
            payload=value.get("payload"),
            confidence=float(value.get("confidence", 1.0)),
            source_agent=value.get("source_agent"),
            metadata=dict(value.get("metadata", {})),
        )
    if isinstance(value, bool):
        if value:
            return ActionProtein("PERMIT", "Approved by reviewer", 1.0)
        return ActionProtein("BLOCK", "Blocked by reviewer", 1.0)
    if value is None:
        return ActionProtein("BLOCK", "Reviewer returned no decision", 0.0)
    return ActionProtein("PERMIT", value, 1.0)


def _build_diagram() -> WiringDiagram:
    diagram = WiringDiagram()
    diagram.add_module(ModuleSpec(
        "executor",
        outputs={"candidate": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        cost=ResourceCost(atp=10, latency_ms=25.0),
    ))
    diagram.add_module(ModuleSpec(
        "reviewer",
        outputs={"approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED)},
        cost=ResourceCost(atp=8, latency_ms=20.0),
    ))
    diagram.add_module(ModuleSpec(
        "sink",
        inputs={
            "candidate": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
            "approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED),
        },
        cost=ResourceCost(atp=2, latency_ms=5.0),
    ))
    diagram.connect("executor", "candidate", "sink", "candidate")
    diagram.connect("reviewer", "approval", "sink", "approval")
    return diagram


@dataclass
class ReviewerGate:
    """Simple reviewer-gate facade with a stable, engineer-facing API."""

    config: ReviewerGateConfig
    loop: CoherentFeedForwardLoop
    diagram: WiringDiagram
    analysis: EpistemicAnalysis
    _executor_fn: ExecutorFn | None = None
    _reviewer_fn: ReviewerFn | None = None

    def run(self, prompt: str) -> ReviewerGateResult:
        """Run prompt through executor + reviewer with default safe interlocks."""
        self.loop._total_requests += 1
        start_time = time()

        if self.loop.enable_circuit_breaker and not self.loop._check_circuit():
            raw = LoopResult(
                success=False,
                action="CIRCUIT_OPEN",
                blocked=True,
                block_reason="Circuit breaker is open",
                gate_logic=self.loop.gate_logic,
            )
            self.loop._record_result(raw)
            return _summarize_gate_result(raw)

        if self.loop.enable_cache:
            cached = self.loop._check_cache(prompt)
            if cached is not None:
                cached.cached = True
                return _summarize_gate_result(cached)

        signal = Signal(content=prompt)
        try:
            if self._executor_fn is None:
                executor_out = self.loop.executor.express(signal)
            else:
                executor_out = _coerce_executor_output(_call_arity(self._executor_fn, prompt))

            if self._reviewer_fn is None:
                reviewer_out = self.loop.assessor.express(signal)
            else:
                reviewer_out = _coerce_reviewer_output(
                    _call_arity(self._reviewer_fn, prompt, executor_out.payload)
                )
        except Exception as exc:
            self.loop._record_failure()
            raw = LoopResult(
                success=False,
                action="ERROR",
                blocked=True,
                block_reason=f"Agent error: {exc}",
                processing_time_ms=(time() - start_time) * 1000,
                gate_logic=self.loop.gate_logic,
            )
            self.loop._record_result(raw)
            return _summarize_gate_result(raw)

        raw = self.loop._apply_gate_logic(executor_out, reviewer_out, prompt)
        raw.processing_time_ms = (time() - start_time) * 1000

        if raw.success and not raw.blocked:
            self.loop._record_success()
        elif not raw.blocked:
            self.loop._record_failure()

        if self.loop.enable_cache:
            self.loop._cache_result(prompt, raw)

        self.loop._record_result(raw)
        if raw.blocked:
            self.loop._total_blocked += 1
            if self.loop.on_block:
                self.loop.on_block(raw)
        else:
            self.loop._total_permitted += 1
            if self.loop.on_permit:
                self.loop.on_permit(raw)

        if not self.loop.silent:
            self.loop._print_result(raw)

        return _summarize_gate_result(raw)


def _summarize_gate_result(raw: LoopResult) -> ReviewerGateResult:
    status = "allowed" if not raw.blocked else "blocked"
    if raw.action == "FAILURE":
        status = "failed"
    reason = raw.block_reason
    if not reason and raw.assessor_output is not None:
        reason = str(raw.assessor_output.payload)
    output = raw.executor_output.payload if raw.executor_output is not None else None
    return ReviewerGateResult(
        allowed=not raw.blocked,
        status=status,
        output=output,
        reason=reason,
        approval_token=raw.approval_token,
        raw=raw,
    )


def reviewer_gate(
    *,
    executor: ExecutorFn | None = None,
    reviewer: ReviewerFn | None = None,
    mode: str = "strict",
    budget: ATP_Store | None = None,
    enable_circuit_breaker: bool = True,
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 60.0,
    enable_cache: bool = True,
    cache_ttl_seconds: float = 300.0,
    timeout_seconds: float = 30.0,
    detection_rate: float = 0.75,
    comm_cost_ratio: float = 0.4,
    silent: bool = True,
) -> ReviewerGate:
    """Create a reviewer gate without exposing the formal substrate directly."""
    gate_logic = _normalize_mode(mode)
    config = ReviewerGateConfig(
        mode=mode,
        enable_circuit_breaker=enable_circuit_breaker,
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
        enable_cache=enable_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        timeout_seconds=timeout_seconds,
        detection_rate=detection_rate,
        comm_cost_ratio=comm_cost_ratio,
    )
    loop = CoherentFeedForwardLoop(
        budget=budget or ATP_Store(budget=500, silent=True),
        gate_logic=gate_logic,
        enable_circuit_breaker=enable_circuit_breaker,
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
        enable_cache=enable_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        timeout_seconds=timeout_seconds,
        silent=silent,
    )
    diagram = _build_diagram()
    analysis = epistemic_analyze(
        diagram,
        detection_rate=detection_rate,
        comm_cost_ratio=comm_cost_ratio,
    )
    return ReviewerGate(
        config=config,
        loop=loop,
        diagram=diagram,
        analysis=analysis,
        _executor_fn=executor,
        _reviewer_fn=reviewer,
    )
