"""Pattern-first multi-model skill runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from typing import Any

from ..core.agent import BioAgent
from ..core.types import ActionProtein, Signal
from ..organelles.nucleus import Nucleus
from ..state.metabolism import ATP_Store
from .types import (
    SkillRunResult,
    SkillRuntimeComponent,
    SkillStage,
    SkillStageResult,
    TelemetryEvent,
)


def _call_arity(fn, *args):
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
    return fn(*args[: len(positional)])


def _coerce_handler_output(value: Any, stage_name: str) -> ActionProtein:
    if isinstance(value, ActionProtein):
        value.source_agent = value.source_agent or stage_name
        return value
    return ActionProtein(
        action_type="EXECUTE",
        payload=value,
        confidence=1.0,
        source_agent=stage_name,
        metadata={"provider": "deterministic", "model": "deterministic-handler"},
    )


def _normalize_mode(mode: str) -> str:
    return mode.strip().lower().replace("-", "_")


@dataclass
class TelemetryProbe:
    """
    Built-in runtime component that records stage-level telemetry.

    By default it also writes the events into a namespaced shared-state key,
    so downstream stages can inspect what happened without colliding with
    application-owned fields.
    """

    state_key: str = "_operon_telemetry"
    events: list[TelemetryEvent] = field(default_factory=list)

    def _append(self, shared_state: dict[str, Any], event: TelemetryEvent) -> None:
        self.events.append(event)
        shared_state.setdefault(self.state_key, []).append(
            {
                "kind": event.kind,
                "stage_name": event.stage_name,
                **event.payload,
            }
        )

    def on_run_start(self, task: str, shared_state: dict[str, Any]) -> None:
        self._append(
            shared_state,
            TelemetryEvent(
                kind="run_start",
                stage_name=None,
                payload={"task": task},
            ),
        )

    def on_stage_start(
        self,
        stage: SkillStage,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        self._append(
            shared_state,
            TelemetryEvent(
                kind="stage_start",
                stage_name=stage.name,
                payload={
                    "role": stage.role,
                    "mode": stage.mode,
                    "known_outputs": sorted(stage_outputs.keys()),
                },
            ),
        )

    def on_stage_result(
        self,
        stage: SkillStage,
        result: SkillStageResult,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        self._append(
            shared_state,
            TelemetryEvent(
                kind="stage_result",
                stage_name=stage.name,
                payload={
                    "provider": result.provider,
                    "model": result.model,
                    "model_alias": result.model_alias,
                    "tokens_used": result.tokens_used,
                    "latency_ms": result.latency_ms,
                    "action_type": result.action_type,
                },
            ),
        )

    def on_run_complete(self, result: SkillRunResult, shared_state: dict[str, Any]) -> None:
        self._append(
            shared_state,
            TelemetryEvent(
                kind="run_complete",
                stage_name=None,
                payload={
                    "final_output": result.final_output,
                    "stages": [stage.stage_name for stage in result.stage_results],
                },
            ),
        )

    def summary(self) -> dict[str, Any]:
        stage_results = [e for e in self.events if e.kind == "stage_result"]
        return {
            "events": len(self.events),
            "stages": [e.stage_name for e in stage_results],
            "total_tokens": sum(e.payload.get("tokens_used", 0) for e in stage_results),
            "total_latency_ms": sum(e.payload.get("latency_ms", 0.0) for e in stage_results),
        }


@dataclass
class SkillOrganism:
    """Composable multi-stage runtime with model-tier routing."""

    stages: tuple[SkillStage, ...]
    nuclei: dict[str, Nucleus]
    budget: ATP_Store
    fast_alias: str = "fast"
    deep_alias: str = "deep"
    components: tuple[SkillRuntimeComponent, ...] = ()
    halt_on_block: bool = True
    _agents: dict[str, BioAgent] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for stage in self.stages:
            if stage.handler is not None:
                continue
            alias = self._resolve_alias(stage)
            nucleus = self.nuclei[alias]
            self._agents[stage.name] = BioAgent(
                name=stage.name,
                role=stage.role,
                atp_store=self.budget,
                nucleus=nucleus,
                instructions=stage.instructions,
                provider_config=stage.provider_config,
                tool_mitochondria=stage.tools,
                silent=True,
            )

    def run(
        self,
        task: str,
        shared_state: dict[str, Any] | None = None,
    ) -> SkillRunResult:
        state = dict(shared_state or {})
        stage_outputs: dict[str, Any] = {}
        stage_results: list[SkillStageResult] = []

        for component in self.components:
            component.on_run_start(task, state)

        for stage in self.stages:
            for component in self.components:
                component.on_stage_start(stage, state, stage_outputs)

            result = self._run_stage(stage, task, state, stage_outputs)
            stage_results.append(result)
            stage_outputs[stage.name] = result.output
            state[stage.name] = result.output
            state["last_stage"] = stage.name

            for component in self.components:
                component.on_stage_result(stage, result, state, stage_outputs)

            if self.halt_on_block and result.action_type in {"BLOCK", "FAILURE"}:
                break

        final_output = stage_results[-1].output if stage_results else None
        run_result = SkillRunResult(
            task=task,
            final_output=final_output,
            stage_results=tuple(stage_results),
            shared_state=dict(state),
        )
        for component in self.components:
            component.on_run_complete(run_result, state)
        return SkillRunResult(
            task=run_result.task,
            final_output=run_result.final_output,
            stage_results=run_result.stage_results,
            shared_state=dict(state),
        )

    def _run_stage(
        self,
        stage: SkillStage,
        task: str,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> SkillStageResult:
        if stage.handler is not None:
            protein = _coerce_handler_output(
                _call_arity(stage.handler, task, dict(shared_state), dict(stage_outputs), stage),
                stage.name,
            )
            return SkillStageResult(
                stage_name=stage.name,
                role=stage.role,
                output=protein.payload,
                model_alias="deterministic",
                provider=protein.metadata.get("provider", "deterministic"),
                model=protein.metadata.get("model", "deterministic-handler"),
                tokens_used=int(protein.metadata.get("tokens_used", 0)),
                latency_ms=float(protein.metadata.get("latency_ms", 0.0)),
                action_type=protein.action_type,
                metadata=dict(protein.metadata),
            )

        alias = self._resolve_alias(stage)
        agent = self._agents[stage.name]
        metadata: dict[str, Any] = {}
        if stage.include_shared_state and shared_state:
            metadata["shared_state"] = dict(shared_state)
        if stage.include_stage_outputs and stage_outputs:
            metadata["stage_outputs"] = dict(stage_outputs)

        protein = agent.express(Signal(content=task, source="SkillOrganism", metadata=metadata))
        return SkillStageResult(
            stage_name=stage.name,
            role=stage.role,
            output=protein.payload,
            model_alias=alias,
            provider=str(protein.metadata.get("provider", alias)),
            model=str(protein.metadata.get("model", alias)),
            tokens_used=int(protein.metadata.get("tokens_used", 0)),
            latency_ms=float(protein.metadata.get("latency_ms", 0.0)),
            action_type=protein.action_type,
            metadata=dict(protein.metadata),
        )

    def _resolve_alias(self, stage: SkillStage) -> str:
        if stage.nucleus is not None:
            alias = stage.nucleus
        else:
            mode = _normalize_mode(stage.mode)
            if mode in {"fixed", "fast", "deterministic"}:
                alias = self.fast_alias
            elif mode in {"fuzzy", "deep", "reasoning"}:
                alias = self.deep_alias
            else:
                raise ValueError(
                    f"Unsupported stage mode '{stage.mode}'. "
                    "Use fixed, fast, deterministic, fuzzy, deep, or reasoning."
                )

        if alias not in self.nuclei:
            raise ValueError(
                f"Stage '{stage.name}' resolved to unknown nucleus alias '{alias}'"
            )
        return alias


def skill_organism(
    *,
    stages: list[SkillStage] | tuple[SkillStage, ...],
    nuclei: dict[str, Nucleus] | None = None,
    fast_nucleus: Nucleus | None = None,
    deep_nucleus: Nucleus | None = None,
    fast_alias: str = "fast",
    deep_alias: str = "deep",
    components: list[SkillRuntimeComponent] | tuple[SkillRuntimeComponent, ...] = (),
    budget: ATP_Store | None = None,
    halt_on_block: bool = True,
) -> SkillOrganism:
    """
    Build a multi-stage organism that routes fixed vs fuzzy work by model tier.

    Fixed stages default to ``fast_alias`` and fuzzy stages default to
    ``deep_alias`` unless a stage pins itself to a specific nucleus alias.
    """

    stage_tuple = tuple(stages)
    if not stage_tuple:
        raise ValueError("skill_organism requires at least one stage")

    nuclei_map = dict(nuclei or {})
    if fast_nucleus is not None:
        nuclei_map[fast_alias] = fast_nucleus
    if deep_nucleus is not None:
        nuclei_map[deep_alias] = deep_nucleus

    for stage in stage_tuple:
        if stage.handler is not None:
            continue
        mode = _normalize_mode(stage.mode)
        if stage.nucleus is None and mode in {"fixed", "fast", "deterministic"} and fast_alias not in nuclei_map:
            raise ValueError(
                f"Stage '{stage.name}' needs a '{fast_alias}' nucleus for mode='{stage.mode}'"
            )
        if stage.nucleus is None and mode in {"fuzzy", "deep", "reasoning"} and deep_alias not in nuclei_map:
            raise ValueError(
                f"Stage '{stage.name}' needs a '{deep_alias}' nucleus for mode='{stage.mode}'"
            )
        if stage.nucleus is not None and stage.nucleus not in nuclei_map:
            raise ValueError(
                f"Stage '{stage.name}' refers to unknown nucleus alias '{stage.nucleus}'"
            )

    return SkillOrganism(
        stages=stage_tuple,
        nuclei=nuclei_map,
        budget=budget or ATP_Store(budget=1000, silent=True),
        fast_alias=fast_alias,
        deep_alias=deep_alias,
        components=tuple(components),
        halt_on_block=halt_on_block,
    )
