"""Adaptive assembly — dynamic organism construction from experience.

Closes the evo-devo loop: fingerprint a task, retrieve the best template
from the PatternLibrary, assemble it into a runnable topology, attach a
WatcherComponent, run, and record the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..memory.bitemporal import BiTemporalMemory
from ..organelles.nucleus import Nucleus
from ..state.metabolism import ATP_Store
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
    SkillRunResult,
    SkillRuntimeComponent,
    SkillStage,
    SkillStageResult,
)
from .watcher import WatcherComponent, WatcherConfig


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveRunResult:
    """Outcome of an adaptive assembly run."""

    template: PatternTemplate
    template_score: float
    run_result: SkillRunResult
    record: PatternRunRecord
    watcher_summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_fingerprint(task: str) -> TaskFingerprint:
    """Derive a rough fingerprint from a task description.

    Intentionally simple v1 heuristic: count sentences as subtasks,
    default to "sequential" shape, extract no roles or tags.
    """
    sentences = [
        s.strip()
        for s in task.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    return TaskFingerprint(
        task_shape="sequential",
        tool_count=0,
        subtask_count=max(1, len(sentences)),
        required_roles=(),
    )


def _spec_to_stage(
    spec: dict[str, Any],
    handlers: dict[str, Any] | None = None,
) -> SkillStage:
    """Convert a stage spec dict into a SkillStage instance."""
    handlers = handlers or {}
    name = spec["name"]
    return SkillStage(
        name=name,
        role=spec.get("role", name),
        instructions=spec.get("instructions", ""),
        mode=spec.get("mode", "fuzzy"),
        nucleus=spec.get("nucleus"),
        handler=handlers.get(name),
        read_query=spec.get("read_query"),
        emit_output_fact=spec.get("emit_output_fact", False),
        fact_tags=tuple(spec.get("fact_tags", ())),
    )


def _infer_success(run_result: SkillRunResult) -> bool:
    """Infer success from run result — no BLOCK/FAILURE in final stage."""
    if not run_result.stage_results:
        return False
    return run_result.stage_results[-1].action_type not in {"BLOCK", "FAILURE"}


def _gate_to_skill_result(task: str, gate_result: Any) -> SkillRunResult:
    """Wrap a ReviewerGateResult into SkillRunResult shape."""
    stage = SkillStageResult(
        stage_name="gate",
        role="reviewer_gate",
        output=gate_result.output,
        model_alias="gate",
        provider="reviewer_gate",
        model="reviewer_gate",
        tokens_used=0,
        latency_ms=getattr(gate_result.raw, "processing_time_ms", 0.0),
        action_type="EXECUTE" if gate_result.allowed else "BLOCK",
        metadata={"status": gate_result.status, "reason": gate_result.reason},
    )
    return SkillRunResult(
        task=task,
        final_output=gate_result.output,
        stage_results=(stage,),
        shared_state={},
    )


def _swarm_to_skill_result(task: str, swarm_result: Any) -> SkillRunResult:
    """Wrap a SpecialistSwarmResult into SkillRunResult shape."""
    stages = tuple(
        SkillStageResult(
            stage_name=role,
            role=role,
            output=output,
            model_alias="swarm",
            provider="specialist_swarm",
            model="specialist_swarm",
            tokens_used=0,
            latency_ms=0.0,
            action_type="EXECUTE",
            metadata={},
        )
        for role, output in swarm_result.outputs.items()
    )
    return SkillRunResult(
        task=task,
        final_output=swarm_result.aggregate,
        stage_results=stages,
        shared_state={},
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_pattern(
    template: PatternTemplate,
    *,
    nuclei: dict[str, Nucleus] | None = None,
    fast_nucleus: Nucleus | None = None,
    deep_nucleus: Nucleus | None = None,
    fast_alias: str = "fast",
    deep_alias: str = "deep",
    components: list[SkillRuntimeComponent] | None = None,
    budget: ATP_Store | None = None,
    substrate: BiTemporalMemory | None = None,
    handlers: dict[str, Any] | None = None,
    workers: dict[str, Any] | None = None,
    aggregator: Any | None = None,
    executor: Any | None = None,
    reviewer: Any | None = None,
) -> SkillOrganism | ReviewerGate | SpecialistSwarm:
    """Assemble a runnable topology from a PatternTemplate.

    Dispatches on ``template.topology``:
      - ``"skill_organism"`` / ``"single_worker"`` → ``skill_organism(...)``
      - ``"reviewer_gate"`` → ``reviewer_gate(...)``
      - ``"specialist_swarm"`` → ``specialist_swarm(...)``
    """
    topology = template.topology

    if topology in ("skill_organism", "single_worker"):
        stages = [_spec_to_stage(s, handlers) for s in template.stage_specs]
        if topology == "single_worker" and len(stages) != 1:
            raise ValueError(
                f"single_worker topology requires exactly 1 stage spec, got {len(stages)}"
            )
        return skill_organism(
            stages=stages,
            nuclei=nuclei,
            fast_nucleus=fast_nucleus,
            deep_nucleus=deep_nucleus,
            fast_alias=fast_alias,
            deep_alias=deep_alias,
            components=components or [],
            budget=budget,
            substrate=substrate,
        )

    if topology == "reviewer_gate":
        h = handlers or {}
        specs = template.stage_specs
        exec_fn = executor or (h.get(specs[0]["name"]) if specs else None)
        review_fn = reviewer or (h.get(specs[1]["name"]) if len(specs) > 1 else None)
        return reviewer_gate(executor=exec_fn, reviewer=review_fn)

    if topology == "specialist_swarm":
        roles = [s.get("role", s["name"]) for s in template.stage_specs]
        return specialist_swarm(
            roles=roles,
            workers=workers,
            aggregator=aggregator,
        )

    raise ValueError(f"Unknown topology: {topology!r}")


# ---------------------------------------------------------------------------
# Adaptive wrapper
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSkillOrganism:
    """Wraps the compose → run → record lifecycle for adaptive assembly."""

    library: PatternLibrary
    template: PatternTemplate
    template_score: float
    fingerprint: TaskFingerprint
    _organism: SkillOrganism | ReviewerGate | SpecialistSwarm
    _watcher: WatcherComponent
    _telemetry: TelemetryProbe

    def run(self, task: str) -> AdaptiveRunResult:
        """Execute the assembled organism and record the outcome."""
        self._watcher.set_fingerprint(self.fingerprint)
        start = datetime.now()

        # Run the appropriate topology
        if isinstance(self._organism, SkillOrganism):
            run_result = self._organism.run(task)
        elif isinstance(self._organism, ReviewerGate):
            run_result = _gate_to_skill_result(task, self._organism.run(task))
        elif isinstance(self._organism, SpecialistSwarm):
            run_result = _swarm_to_skill_result(task, self._organism.run(task))
        else:
            raise TypeError(f"Unsupported organism type: {type(self._organism)}")

        latency_ms = (datetime.now() - start).total_seconds() * 1000
        success = _infer_success(run_result)
        total_tokens = sum(sr.tokens_used for sr in run_result.stage_results)

        # Build intervention records from watcher
        intervention_dicts = tuple(
            {"kind": i.kind.value, "stage_name": i.stage_name, "reason": i.reason}
            for i in self._watcher.interventions
        )

        # Record run outcome in library
        record = PatternRunRecord(
            record_id=self.library.make_id(),
            template_id=self.template.template_id,
            fingerprint=self.fingerprint,
            success=success,
            latency_ms=latency_ms,
            tokens_used=total_tokens,
            interventions=intervention_dicts,
        )
        self.library.record_run(record)

        # Record experiences in watcher pool
        for intv in self._watcher.interventions:
            # Find the dominant signal for this intervention's stage
            stage_signals = [
                s for s in self._watcher.signals
                if s.stage_name == intv.stage_name
            ]
            dominant = stage_signals[0].category.value if stage_signals else "unknown"
            detail = stage_signals[0].detail if stage_signals else {}
            self._watcher.record_experience(
                fingerprint=self.fingerprint,
                stage_name=intv.stage_name,
                signal_category=dominant,
                signal_detail=detail,
                intervention_kind=intv.kind.value,
                intervention_reason=intv.reason,
                outcome_success=success,
            )

        return AdaptiveRunResult(
            template=self.template,
            template_score=self.template_score,
            run_result=run_result,
            record=record,
            watcher_summary=self._watcher.summary(),
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def adaptive_skill_organism(
    task: str,
    *,
    fingerprint: TaskFingerprint | None = None,
    library: PatternLibrary,
    nuclei: dict[str, Nucleus] | None = None,
    fast_nucleus: Nucleus | None = None,
    deep_nucleus: Nucleus | None = None,
    fast_alias: str = "fast",
    deep_alias: str = "deep",
    components: list[SkillRuntimeComponent] | None = None,
    watcher_config: WatcherConfig | None = None,
    budget: ATP_Store | None = None,
    substrate: BiTemporalMemory | None = None,
    handlers: dict[str, Any] | None = None,
    workers: dict[str, Any] | None = None,
    aggregator: Any | None = None,
    executor: Any | None = None,
    reviewer: Any | None = None,
    auto_fingerprint: bool = True,
) -> AdaptiveSkillOrganism:
    """Build an adaptive organism from library templates.

    1. Auto-fingerprint the task if no fingerprint provided
    2. Retrieve the best template from the library
    3. Assemble the pattern into a runnable topology
    4. Attach a watcher and telemetry probe
    5. Return an AdaptiveSkillOrganism ready to .run()
    """
    # 1. Fingerprint
    if fingerprint is None:
        if not auto_fingerprint:
            raise ValueError("No fingerprint provided and auto_fingerprint is False")
        fingerprint = _auto_fingerprint(task)

    # 2. Retrieve best template
    ranked = library.top_templates_for(fingerprint)
    if not ranked:
        raise ValueError("No templates in library")
    template, score = ranked[0]

    # 3. Build watcher and telemetry
    watcher = WatcherComponent(config=watcher_config or WatcherConfig())
    telemetry = TelemetryProbe()
    all_components: list[SkillRuntimeComponent] = list(components or [])
    all_components.extend([watcher, telemetry])

    # 4. Assemble pattern
    organism = assemble_pattern(
        template,
        nuclei=nuclei,
        fast_nucleus=fast_nucleus,
        deep_nucleus=deep_nucleus,
        fast_alias=fast_alias,
        deep_alias=deep_alias,
        components=all_components,
        budget=budget,
        substrate=substrate,
        handlers=handlers,
        workers=workers,
        aggregator=aggregator,
        executor=executor,
        reviewer=reviewer,
    )

    return AdaptiveSkillOrganism(
        library=library,
        template=template,
        template_score=score,
        fingerprint=fingerprint,
        _organism=organism,
        _watcher=watcher,
        _telemetry=telemetry,
    )
