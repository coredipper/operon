"""LangGraph compiler — per-stage nodes calling organism.run_single_stage().

Each organism stage becomes a LangGraph node that calls
``organism.run_single_stage()``. Conditional edges route based on
the return value: ``"continue"`` → next stage, ``"halt"``/``"blocked"``
→ END. All structural guarantees (CertificateGate, VerifierComponent,
WatcherComponent, halt_on_block, retry, escalation) are handled by
the organism's own per-stage logic — no reimplementation.

LangGraph provides the execution host, graph topology, observability,
and checkpointing. The organism provides the structural guarantees.

Requires ``langgraph``::

    pip install operon-ai[langgraph]

All external imports are lazy.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from ..patterns.types import (
    RunContext,
    SkillStageResult,
    WATCHER_STATE_KEY,
)

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

try:
    from langgraph.graph import StateGraph  # noqa: F401

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class LangGraphState(TypedDict):
    """LangGraph state flowing through the per-stage graph."""

    messages: Annotated[list, operator.add]
    task: str
    stage_outputs: dict[str, str]
    stage_results: Annotated[list, operator.add]
    shared_state: dict[str, Any]
    halted: bool
    _routing: str  # "continue" / "halt" / "blocked"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LangGraphResult:
    """Result of executing an organism in LangGraph."""

    output: str
    stage_outputs: dict[str, str]
    interventions: list[dict[str, Any]]
    timing_ms: float
    certificates_verified: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def organism_to_langgraph(organism: Any) -> Any:
    """Compile a SkillOrganism into a per-stage LangGraph StateGraph.

    Each stage becomes a LangGraph node that calls
    ``organism.run_single_stage()``. Conditional edges route based
    on the return value.

    The compiled graph requires ``task`` and ``shared_state`` in its
    input state. Use :func:`run_organism_langgraph` for the full
    lifecycle (``on_run_start``/``on_run_complete`` + certificate
    verification).

    Note: LangGraph checkpointing between stages does not persist
    component instance state (watcher counters, telemetry buffers).
    For resumable execution, use ``organism.run()`` directly.

    Parameters
    ----------
    organism:
        A ``SkillOrganism`` with stages and components attached.

    Returns
    -------
    CompiledStateGraph
        A LangGraph graph with one node per stage.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[langgraph]"
        ) from e

    stages = organism.stages
    components = organism.components

    # Resolve watcher key
    watcher_key = WATCHER_STATE_KEY
    for component in components:
        cfg = getattr(component, "config", None)
        sk = getattr(cfg, "state_key", None)
        if sk is not None:
            watcher_key = sk
            break

    def make_stage_node(stage):
        """Create a LangGraph node that calls organism.run_single_stage()."""
        def node(state: LangGraphState) -> dict:
            ctx = RunContext(state.get("shared_state", {}), watcher_key=watcher_key)
            outputs = state.get("stage_outputs", {})
            results_dicts = state.get("stage_results", [])

            # Reconstruct mutable lists for run_single_stage
            results: list[SkillStageResult] = []

            try:
                decision = organism.run_single_stage(
                    stage, state["task"], ctx, outputs, results,
                )
            except Exception as exc:
                return {
                    "halted": True,
                    "_routing": "halt",
                    "shared_state": dict(ctx),
                    "stage_results": [{
                        "stage": stage.name, "output": f"Error: {exc}",
                        "error": str(exc),
                    }],
                }

            # Serialize the new stage result
            new_results = []
            for sr in results:
                out = sr.output if isinstance(sr.output, str) else str(sr.output)
                new_results.append({
                    "stage": sr.stage_name,
                    "model": sr.model_alias,
                    "action": sr.action_type,
                    "output": out,
                })
                outputs[sr.stage_name] = out

            return {
                "stage_outputs": outputs,
                "stage_results": new_results,
                "shared_state": dict(ctx),
                "halted": decision != "continue",
                "_routing": decision,
            }

        return node

    # Build the graph
    builder = StateGraph(LangGraphState)

    for stage in stages:
        builder.add_node(stage.name, make_stage_node(stage))

    # Routing: each stage → next or END
    def route(state: LangGraphState) -> str:
        return "halt" if state.get("_routing") != "continue" else "next"

    for i, stage in enumerate(stages):
        if i < len(stages) - 1:
            builder.add_conditional_edges(
                stage.name, route,
                {"next": stages[i + 1].name, "halt": END},
            )
        else:
            builder.add_edge(stage.name, END)

    builder.add_edge(START, stages[0].name)
    return builder.compile()


def run_organism_langgraph(
    organism: Any,
    *,
    task: str,
    verify_certificates: bool = True,
) -> LangGraphResult:
    """Compile and execute an organism in LangGraph in one call."""
    try:
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[langgraph]"
        ) from e

    # Call on_run_start for all components
    watcher_key = WATCHER_STATE_KEY
    for component in organism.components:
        cfg = getattr(component, "config", None)
        sk = getattr(cfg, "state_key", None)
        if sk is not None:
            watcher_key = sk
            break

    ctx = RunContext({}, watcher_key=watcher_key)
    for component in organism.components:
        component.on_run_start(task, ctx)

    graph = organism_to_langgraph(organism)

    t0 = time.monotonic()
    result = graph.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "stage_outputs": {},
        "stage_results": [],
        "shared_state": dict(ctx),
        "halted": False,
        "_routing": "continue",
    })
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Call on_run_complete for all components
    final_ctx = RunContext(result.get("shared_state", {}), watcher_key=watcher_key)
    stage_outputs = result.get("stage_outputs", {})
    stage_results_dicts = result.get("stage_results", [])

    final_output = ""
    if stage_results_dicts:
        last = stage_results_dicts[-1]
        final_output = last.get("output", "")

    from ..patterns.types import SkillRunResult
    run_result = SkillRunResult(
        task=task,
        final_output=final_output,
        stage_results=(),
        shared_state=dict(final_ctx),
    )
    for component in organism.components:
        component.on_run_complete(run_result, final_ctx)

    # Extract interventions
    interventions = []
    for component in organism.components:
        if hasattr(component, "interventions"):
            for intv in component.interventions:
                interventions.append({
                    "stage": intv.stage_name,
                    "kind": intv.kind.value,
                    "reason": intv.reason,
                })

    # Certificate verification
    cert_results: list[dict[str, Any]] = []
    if verify_certificates:
        for cert in organism.collect_certificates():
            verification = cert.verify()
            cert_results.append({
                "theorem": cert.theorem,
                "holds": verification.holds,
            })

    stages_completed = [d.get("stage", "") for d in stage_results_dicts]

    return LangGraphResult(
        output=final_output,
        stage_outputs=stage_outputs,
        interventions=interventions,
        timing_ms=elapsed_ms,
        certificates_verified=cert_results,
        metadata={
            "halted": result.get("halted", False),
            "stages_completed": stages_completed,
            "run_complete": True,
        },
    )
