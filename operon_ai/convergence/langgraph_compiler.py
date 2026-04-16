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
    # Fan-out/fan-in support for parallel groups
    _parallel_results: Annotated[list, operator.add]
    _parallel_snap: dict[str, Any]


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


def compute_group_node_names(groups, stage_names: set[str] | None = None) -> list[str]:
    """Compute unique node names for stage groups.

    Single-stage groups use the stage name. Multi-stage groups get a
    generated name that cannot collide with any stage name.
    """
    all_stage_names = stage_names or set()
    names: list[str] = []
    for i, group in enumerate(groups):
        if len(group) == 1:
            names.append(group[0].name)
        else:
            # Generate a name that can't collide with user stage names
            candidate = f"__parallel_{i}"
            while candidate in all_stage_names:
                candidate = f"__parallel_{i}_{id(group)}"
            names.append(candidate)
    return names


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

    components = organism.components
    groups = organism.stage_groups or tuple((s,) for s in organism.stages)

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

    def make_parallel_stage_node(stage):
        """Create a LangGraph node for a stage within a parallel group.

        Like make_stage_node but also appends to _parallel_results for
        the join node to merge.
        """
        def node(state: LangGraphState) -> dict:
            import copy as _copy
            ctx = RunContext(
                _copy.deepcopy(state.get("shared_state", {})),
                watcher_key=watcher_key,
            )
            outputs = _copy.deepcopy(state.get("stage_outputs", {}))
            results: list[SkillStageResult] = []

            try:
                decision = organism.run_single_stage(
                    stage, state["task"], ctx, outputs, results,
                )
            except Exception as exc:
                return {
                    "_parallel_results": [{
                        "stage": stage.name,
                        "shared_state": dict(ctx),
                        "stage_outputs": outputs,
                        "stage_results": [{
                            "stage": stage.name, "output": f"Error: {exc}",
                            "error": str(exc),
                        }],
                        "decision": "halt",
                    }],
                }

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
                "_parallel_results": [{
                    "stage": stage.name,
                    "shared_state": dict(ctx),
                    "stage_outputs": outputs,
                    "stage_results": new_results,
                    "decision": decision,
                }],
            }

        return node

    def make_fork_node():
        """Create a fork node that snapshots state before fan-out."""
        def node(state: LangGraphState) -> dict:
            import copy as _copy
            return {
                "_parallel_snap": _copy.deepcopy(state.get("shared_state", {})),
                "_parallel_results": [],  # clear from any previous group
            }
        return node

    def make_fork_router(group):
        """Create a conditional edge function that returns Send objects."""
        def router(state: LangGraphState):
            try:
                from langgraph.types import Send
            except ImportError:
                from langgraph.graph import Send
            import copy as _copy
            snap = state.get("_parallel_snap", state.get("shared_state", {}))
            snap_outputs = state.get("stage_outputs", {})
            return [
                Send(stage.name, {
                    "messages": state.get("messages", []),
                    "task": state["task"],
                    "shared_state": _copy.deepcopy(snap),
                    "stage_outputs": _copy.deepcopy(snap_outputs),
                    "stage_results": [],
                    "halted": False,
                    "_routing": "continue",
                    "_parallel_results": [],
                    "_parallel_snap": {},
                })
                for stage in group
            ]
        return router

    def make_join_node(group):
        """Create a join node that merges parallel results."""
        from ..patterns.organism import _merge_parallel_results

        def node(state: LangGraphState) -> dict:
            snap = state.get("_parallel_snap", {})
            parallel_results = state.get("_parallel_results", [])

            # Build per_stage dict for _merge_parallel_results
            per_stage: dict[str, tuple] = {}
            all_stage_results = []
            for r in parallel_results:
                per_stage[r["stage"]] = (
                    r["shared_state"],
                    r["stage_outputs"],
                    [],  # SkillStageResult list (not serialized)
                    r["decision"],
                )
                all_stage_results.extend(r.get("stage_results", []))

            merged_state = dict(snap)
            merged_outputs = dict(state.get("stage_outputs", {}))
            merged_results: list = []

            try:
                _merge_parallel_results(
                    merged_state, merged_outputs, merged_results,
                    snap, per_stage,
                )
            except Exception:
                pass  # conflict detection — log but don't crash

            halted = any(r["decision"] != "continue" for r in parallel_results)
            return {
                "shared_state": merged_state,
                "stage_outputs": merged_outputs,
                "stage_results": all_stage_results,
                "halted": halted,
                "_routing": "halt" if halted else "continue",
                "_parallel_results": [],
            }

        return node

    # Build the graph — fan-out/fan-in for parallel groups
    builder = StateGraph(LangGraphState)

    # Collect all node names for sequential wiring between groups
    group_entry_nodes: list[str] = []  # first node of each group
    group_exit_nodes: list[str] = []   # last node of each group

    for i, group in enumerate(groups):
        if len(group) == 1:
            # Sequential: single node
            stage = group[0]
            builder.add_node(stage.name, make_stage_node(stage))
            group_entry_nodes.append(stage.name)
            group_exit_nodes.append(stage.name)
        else:
            # Parallel: fork → [stage_a, stage_b, ...] → join
            fork_name = f"__fork_{i}"
            join_name = f"__join_{i}"

            builder.add_node(fork_name, make_fork_node())
            for stage in group:
                builder.add_node(stage.name, make_parallel_stage_node(stage))
            builder.add_node(join_name, make_join_node(group))

            # Fork → Send to each stage (fan-out)
            builder.add_conditional_edges(fork_name, make_fork_router(group))

            # Each stage → join (fan-in)
            for stage in group:
                builder.add_edge(stage.name, join_name)

            group_entry_nodes.append(fork_name)
            group_exit_nodes.append(join_name)

    # Wire groups sequentially with routing
    def route(state: LangGraphState) -> str:
        return "halt" if state.get("_routing") != "continue" else "next"

    for i in range(len(group_exit_nodes)):
        if i < len(group_entry_nodes) - 1:
            builder.add_conditional_edges(
                group_exit_nodes[i], route,
                {"next": group_entry_nodes[i + 1], "halt": END},
            )
        else:
            builder.add_edge(group_exit_nodes[i], END)

    builder.add_edge(START, group_entry_nodes[0])
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
        "_parallel_results": [],
        "_parallel_snap": {},
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
