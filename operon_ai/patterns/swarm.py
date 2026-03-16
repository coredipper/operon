"""Pattern-first specialist swarm wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable

from ..core.epistemic import EpistemicAnalysis, analyze as epistemic_analyze
from ..core.types import DataType, IntegrityLabel
from ..core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram
from ..core.wiring_runtime import DiagramExecutor, ExecutionReport
from .types import SpecialistSwarmConfig, SpecialistSwarmResult

WorkerFn = Callable[..., Any]
AggregatorFn = Callable[..., Any]


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


def _call_aggregate(fn: Callable[..., Any], task: str, outputs: dict[str, Any]) -> Any:
    try:
        params = list(signature(fn).parameters.values())
    except (TypeError, ValueError):
        return fn(task, outputs)
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return fn(task, outputs)
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 0:
        return fn()
    if len(positional) == 1:
        return fn(outputs)
    return fn(task, outputs)


def _default_worker(role: str, task: str) -> dict[str, str]:
    return {
        "role": role,
        "summary": f"{role} reviewed the task",
        "task": task,
    }


def _default_aggregate(outputs: dict[str, Any]) -> Any:
    if outputs and all(isinstance(value, str) for value in outputs.values()):
        return "\n\n".join(
            f"[{role}] {value}" for role, value in outputs.items()
        )
    return outputs


def _build_diagram(roles: tuple[str, ...]) -> WiringDiagram:
    diagram = WiringDiagram()
    for role in roles:
        diagram.add_module(ModuleSpec(
            role,
            inputs={"task": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"result": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            cost=ResourceCost(atp=10, latency_ms=20.0),
        ))
    diagram.add_module(ModuleSpec(
        "coordinator",
        inputs={
            f"in_{idx}": PortType(DataType.JSON, IntegrityLabel.VALIDATED)
            for idx, _ in enumerate(roles)
        },
        outputs={"result": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        cost=ResourceCost(atp=6, latency_ms=10.0),
    ))
    for idx, role in enumerate(roles):
        diagram.connect(role, "result", "coordinator", f"in_{idx}")
    return diagram


@dataclass
class SpecialistSwarm:
    """Engineer-facing facade for centralized specialist coordination."""

    config: SpecialistSwarmConfig
    diagram: WiringDiagram
    analysis: EpistemicAnalysis
    _workers: dict[str, WorkerFn]
    _aggregator: AggregatorFn | None = None

    def run(self, task: str) -> SpecialistSwarmResult:
        """Execute all specialists and aggregate their outputs through a hub."""
        executor = DiagramExecutor(self.diagram)

        for role in self.config.roles:
            worker_fn = self._workers.get(role)

            def handler(inputs, *, _role=role, _worker_fn=worker_fn):
                prompt = str(inputs["task"].value)
                if _worker_fn is None:
                    result = _default_worker(_role, prompt)
                else:
                    result = _call_arity(_worker_fn, prompt, _role)
                return {"result": result}

            executor.register_module(role, handler)

        def coordinator_handler(inputs):
            ordered = {
                role: inputs[f"in_{idx}"].value
                for idx, role in enumerate(self.config.roles)
            }
            if self._aggregator is None:
                aggregate = _default_aggregate(ordered)
            else:
                aggregate = _call_aggregate(self._aggregator, task, ordered)
            return {"result": aggregate}

        executor.register_module("coordinator", coordinator_handler)
        report = executor.execute(
            external_inputs={
                role: {"task": task}
                for role in self.config.roles
            }
        )
        outputs = {
            role: report.modules[role].outputs["result"].value
            for role in self.config.roles
        }
        aggregate = report.modules["coordinator"].outputs["result"].value
        return SpecialistSwarmResult(
            outputs=outputs,
            aggregate=aggregate,
            execution_report=report,
            diagram=self.diagram,
            analysis=self.analysis,
        )


def specialist_swarm(
    *,
    roles: list[str] | tuple[str, ...],
    workers: dict[str, WorkerFn] | None = None,
    aggregator: AggregatorFn | None = None,
    aggregation: str = "hub",
    detection_rate: float = 0.75,
    comm_cost_ratio: float = 0.4,
) -> SpecialistSwarm:
    """Create a centralized specialist swarm with a simple default hub."""
    roles_tuple = tuple(roles)
    if not roles_tuple:
        raise ValueError("specialist_swarm requires at least one role")
    if len(set(roles_tuple)) != len(roles_tuple):
        raise ValueError("specialist_swarm roles must be unique")
    if aggregation != "hub":
        raise ValueError(
            "specialist_swarm currently supports aggregation='hub' only"
        )

    config = SpecialistSwarmConfig(
        roles=roles_tuple,
        aggregation=aggregation,
        detection_rate=detection_rate,
        comm_cost_ratio=comm_cost_ratio,
    )
    diagram = _build_diagram(roles_tuple)
    analysis = epistemic_analyze(
        diagram,
        detection_rate=detection_rate,
        comm_cost_ratio=comm_cost_ratio,
    )
    return SpecialistSwarm(
        config=config,
        diagram=diagram,
        analysis=analysis,
        _workers=dict(workers or {}),
        _aggregator=aggregator,
    )
