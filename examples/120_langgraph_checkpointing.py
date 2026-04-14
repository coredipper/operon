"""
Example 120: LangGraph Checkpointing Between Stages
=====================================================

Demonstrates that LangGraph's per-stage graph topology enables
checkpointing between stages — run some stages, capture state,
and resume later.

1. Build a 4-stage organism
2. Compile to LangGraph with a MemorySaver checkpointer
3. Execute and inspect checkpoint state
4. Document the limitation: component instance state does not persist

Limitation (documented honestly): LangGraph checkpointing saves
the graph state (stage_outputs, shared_state, messages) but NOT
component instance state (watcher counters, telemetry buffers,
verifier quality scores). For fully resumable execution with
component state, use ``organism.run()`` directly.

Usage: python examples/120_langgraph_checkpointing.py
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, Any, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.types import RunContext, SkillStageResult, WATCHER_STATE_KEY

# LangGraph imports (lazy)
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langchain_core.messages import HumanMessage
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


# State schema — defined at module level so type hints resolve
class CheckpointState(TypedDict):
    messages: Annotated[list, operator.add]
    task: str
    stage_outputs: dict[str, str]
    stage_results: Annotated[list, operator.add]
    shared_state: dict[str, Any]
    halted: bool
    _routing: str


def build_organism():
    """Build a 4-stage pipeline."""
    fast = Nucleus(provider=MockProvider(responses={
        "triage": "Priority: P1 (production crash)",
        "classify": "Category: authentication",
        "investig": "Root cause: expired JWT not caught by middleware",
        "implement": "Fix: added JWT expiry check in auth middleware with 5-min grace period",
    }))
    deep = Nucleus(provider=MockProvider(responses={}))

    return skill_organism(
        stages=[
            SkillStage(name="triage", role="Triager",
                       instructions="Triage the incident.",
                       mode="fixed"),
            SkillStage(name="classify", role="Classifier",
                       instructions="Classify the issue type.",
                       mode="fixed"),
            SkillStage(name="investigate", role="Investigator",
                       instructions="Investigate root cause.",
                       mode="fixed"),
            SkillStage(name="fix", role="Engineer",
                       instructions="Implement the fix.",
                       mode="fuzzy"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=2000, silent=True),
    )


def _build_checkpointed_graph(org):
    """Build a per-stage LangGraph with MemorySaver checkpointer."""
    stages = org.stages
    watcher_key = WATCHER_STATE_KEY

    def make_stage_node(stage):
        def node(state: CheckpointState) -> dict:
            ctx = RunContext(state.get("shared_state", {}), watcher_key=watcher_key)
            outputs = state.get("stage_outputs", {})
            results: list[SkillStageResult] = []

            decision = org.run_single_stage(
                stage, state["task"], ctx, outputs, results,
            )

            new_results = []
            for sr in results:
                out = sr.output if isinstance(sr.output, str) else str(sr.output)
                new_results.append({"stage": sr.stage_name, "output": out})
                outputs[sr.stage_name] = out

            return {
                "stage_outputs": outputs,
                "stage_results": new_results,
                "shared_state": dict(ctx),
                "halted": decision != "continue",
                "_routing": decision,
            }
        return node

    def route(state: CheckpointState) -> str:
        return "halt" if state.get("_routing") != "continue" else "next"

    builder = StateGraph(CheckpointState)
    for stage in stages:
        builder.add_node(stage.name, make_stage_node(stage))

    for i, stage in enumerate(stages):
        if i < len(stages) - 1:
            builder.add_conditional_edges(
                stage.name, route,
                {"next": stages[i + 1].name, "halt": END},
            )
        else:
            builder.add_edge(stage.name, END)

    builder.add_edge(START, stages[0].name)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer), checkpointer


def main():
    print("=" * 60)
    print("LangGraph Checkpointing Between Stages")
    print("=" * 60)

    if not HAS_LANGGRAPH:
        print("\n  LangGraph not installed. Install with:")
        print("  pip install operon-ai[langgraph]")
        return

    org = build_organism()

    # ------------------------------------------------------------------
    # 1. Compile with checkpointer
    # ------------------------------------------------------------------
    print("\n--- Compile with Checkpointer ---")
    graph, checkpointer = _build_checkpointed_graph(org)
    print(f"  stages: {[s.name for s in org.stages]}")
    print(f"  checkpointer: MemorySaver")

    # ------------------------------------------------------------------
    # 2. Execute with thread_id
    # ------------------------------------------------------------------
    print("\n--- Execute ---")
    task = "Production auth failures after JWT migration"
    config = {"configurable": {"thread_id": "incident-001"}}

    # Initialize components
    for component in org.components:
        component.on_run_start(task, {})

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "stage_outputs": {},
            "stage_results": [],
            "shared_state": {},
            "halted": False,
            "_routing": "continue",
        },
        config=config,
    )

    # ------------------------------------------------------------------
    # 3. Inspect results and checkpoint
    # ------------------------------------------------------------------
    print("\n--- Results ---")
    stage_results = result.get("stage_results", [])
    stage_outputs = result.get("stage_outputs", {})

    print(f"  stages completed: {len(stage_results)}")
    for sr in stage_results:
        output_preview = str(sr.get("output", ""))[:55]
        print(f"    {sr.get('stage', '?'):15s} {output_preview}")

    print("\n--- Checkpoint ---")
    checkpoint = checkpointer.get(config)
    print(f"  checkpoint exists: {checkpoint is not None}")
    if checkpoint:
        print(f"  checkpoint id:    {checkpoint.get('id', 'N/A')}")

    # ------------------------------------------------------------------
    # 4. Limitation: component state is NOT checkpointed
    # ------------------------------------------------------------------
    print("\n--- Limitation: Component State ---")
    print("  LangGraph checkpoints save:")
    print("    + stage_outputs (dict[str, str])")
    print("    + shared_state (dict[str, Any])")
    print("    + messages (list)")
    print("    + halted flag")
    print("  LangGraph checkpoints do NOT save:")
    print("    - WatcherComponent.signals (list)")
    print("    - WatcherComponent.interventions (list)")
    print("    - VerifierComponent.quality_scores (list)")
    print("    - ATP_Store.consumed (counter)")
    print()
    print("  For fully resumable execution with component state,")
    print("  use organism.run() directly.")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    assert len(stage_results) == 4, f"Expected 4, got {len(stage_results)}"
    assert checkpoint is not None, "Checkpoint should exist"
    assert len(stage_outputs) == 4
    assert result.get("halted") is False

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
