"""LangGraph compiler — compile SkillOrganism directly to a guarded StateGraph.

Unlike ``guarded_graph.py`` which takes a compiled DeerFlow dict and
reimplements component logic, this compiler takes a ``SkillOrganism``
directly and **reuses** its attached components (WatcherComponent,
VerifierComponent, CertificateGateComponent) via their protocol hooks.

The result is a LangGraph ``CompiledStateGraph`` where Operon's structural
guarantees are enforced by LangGraph's conditional edges — not by
observation from outside.

Requires ``langgraph`` and ``langchain-openai``::

    pip install operon-ai[deerflow]

All external imports are lazy.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from ..patterns.types import (
    InterventionKind,
    RunContext,
    SkillStageResult,
    WatcherIntervention,
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
    """LangGraph state flowing through the compiled organism graph."""

    messages: Annotated[list, operator.add]
    stage_outputs: dict[str, str]
    shared_state: dict[str, Any]  # Operon RunContext (serialized)
    current_stage: str
    use_deep: bool
    halted: bool
    intervention_log: Annotated[list, operator.add]
    _routing: str  # Current routing decision: "" / "retry" / "escalate"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LangGraphResult:
    """Result of executing a compiled organism in LangGraph."""

    output: str
    stage_outputs: dict[str, str]
    interventions: list[dict[str, Any]]
    timing_ms: float
    certificates_verified: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def organism_to_langgraph(
    organism: Any,
    *,
    fast_model: Any | None = None,
    deep_model: Any | None = None,
    fast_model_name: str = "gemma4:latest",
    deep_model_name: str = "gemma4:latest",
    ollama_base_url: str = "http://localhost:11434/v1",
) -> Any:
    """Compile a SkillOrganism into a guarded LangGraph StateGraph.

    Each organism stage becomes a triple of LangGraph nodes:

    1. **pre_guard** — calls ``component.on_stage_start()`` for all
       attached components. Routes to END on HALT (CertificateGate).
    2. **agent** — calls the LLM with the stage's instructions.
    3. **post_guard** — calls ``component.on_stage_result()`` for all
       attached components, reads ``RunContext.watcher_intervention``,
       and routes to retry/escalate/halt/next via conditional edges.

    This reuses the organism's actual components — no reimplementation
    of watcher/verifier/gate logic.

    Parameters
    ----------
    organism:
        A ``SkillOrganism`` with stages and components attached.
    fast_model / deep_model:
        Pre-built ``BaseChatModel`` instances.  If ``None``, created
        from model names via ``ChatOpenAI`` pointed at Ollama.

    Returns
    -------
    CompiledStateGraph
        A LangGraph graph that can be invoked or streamed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[deerflow]"
        ) from e

    # --- Model resolution ---
    if fast_model is None or deep_model is None:
        from langchain_openai import ChatOpenAI

        if fast_model is None:
            fast_model = ChatOpenAI(
                base_url=ollama_base_url, model=fast_model_name,
                api_key="ollama", temperature=0.0, max_tokens=1024,
            )
        if deep_model is None:
            deep_model = ChatOpenAI(
                base_url=ollama_base_url, model=deep_model_name,
                api_key="ollama", temperature=0.0, max_tokens=2048,
            )

    stages = organism.stages
    components = organism.components

    # Resolve watcher key from components (same logic as SkillOrganism.run)
    watcher_key = WATCHER_STATE_KEY
    for component in components:
        cfg = getattr(component, "config", None)
        sk = getattr(cfg, "state_key", None)
        if sk is not None:
            watcher_key = sk
            break

    # --- Node factories ---

    def make_pre_guard(stage):
        """Call all component.on_stage_start hooks. Check for HALT."""
        def pre_guard(state: LangGraphState) -> dict:
            ctx = RunContext(state.get("shared_state", {}), watcher_key=watcher_key)
            outputs = state.get("stage_outputs", {})

            for component in components:
                component.on_stage_start(stage, ctx, outputs)

            # Check for pre-stage HALT (e.g., CertificateGate)
            intervention = ctx.pop(watcher_key, None)
            if isinstance(intervention, WatcherIntervention):
                if intervention.kind == InterventionKind.HALT:
                    return {
                        "halted": True,
                        "shared_state": dict(ctx),
                        "intervention_log": [{
                            "stage": stage.name, "kind": "halt",
                            "reason": intervention.reason,
                        }],
                    }

            return {
                "current_stage": stage.name,
                "shared_state": dict(ctx),
            }
        return pre_guard

    def make_agent(stage):
        """Call the LLM with stage instructions."""
        def agent(state: LangGraphState) -> dict:
            model = deep_model if state.get("use_deep") else fast_model
            sys_msg = SystemMessage(
                content=f"Role: {stage.role}. {stage.instructions}"
            )

            input_msgs = [sys_msg]
            for m in state.get("messages", []):
                if isinstance(m, HumanMessage):
                    input_msgs.append(m)
                    break
            # Prior stage outputs as context
            for prev_name, prev_output in state.get("stage_outputs", {}).items():
                input_msgs.append(AIMessage(content=f"[{prev_name}]: {prev_output}"))

            try:
                response = model.invoke(input_msgs)
                output_text = response.content if hasattr(response, "content") else str(response)
                action_type = "EXECUTE"
            except Exception as exc:
                output_text = f"Error: {exc}"
                action_type = "FAILURE"
                response = AIMessage(content=output_text)

            # Build a SkillStageResult for the components
            model_alias = "deep" if state.get("use_deep") else "fast"
            result = SkillStageResult(
                stage_name=stage.name,
                role=stage.role,
                output=output_text,
                model_alias=model_alias,
                provider="langgraph",
                model=getattr(model, "model_name", model_alias),
                tokens_used=0,
                latency_ms=0.0,
                action_type=action_type,
                metadata={},
            )

            # Store result in shared_state for post_guard to use
            ctx = RunContext(state.get("shared_state", {}), watcher_key=watcher_key)
            ctx["_lg_stage_result"] = result
            ctx["_lg_response"] = response

            return {
                "messages": [response],
                "shared_state": dict(ctx),
                "use_deep": False,  # Reset
            }
        return agent

    def make_post_guard(stage):
        """Call all component.on_stage_result hooks. Read intervention."""
        def post_guard(state: LangGraphState) -> dict:
            ctx = RunContext(state.get("shared_state", {}), watcher_key=watcher_key)
            outputs = state.get("stage_outputs", {})
            result = ctx.pop("_lg_stage_result", None)
            ctx.pop("_lg_response", None)

            if result is None:
                return {"shared_state": dict(ctx)}

            # Call components in the same order as SkillOrganism.run():
            # non-watcher first, then watcher last
            for component in components:
                if not hasattr(component, "_decide_intervention"):
                    component.on_stage_result(stage, result, ctx, outputs)
            for component in components:
                if hasattr(component, "_decide_intervention"):
                    component.on_stage_result(stage, result, ctx, outputs)

            # Read intervention (set by WatcherComponent)
            intervention = ctx.pop(watcher_key, None)

            if isinstance(intervention, WatcherIntervention):
                kind = intervention.kind.value
                return {
                    "halted": kind == "halt",
                    "use_deep": kind == "escalate",
                    "_routing": kind,  # "halt" / "escalate" / "retry"
                    "shared_state": dict(ctx),
                    "intervention_log": [{
                        "stage": stage.name, "kind": kind,
                        "reason": intervention.reason,
                    }],
                }

            # No intervention — accept output
            new_outputs = dict(outputs)
            new_outputs[stage.name] = result.output
            return {
                "stage_outputs": new_outputs,
                "_routing": "",  # Clear — no intervention
                "shared_state": dict(ctx),
                "intervention_log": [],
            }

        return post_guard

    # --- Build the graph ---
    builder = StateGraph(LangGraphState)

    for i, stage in enumerate(stages):
        pre = f"pre_{stage.name}"
        ag = f"agent_{stage.name}"
        post = f"post_{stage.name}"

        builder.add_node(pre, make_pre_guard(stage))
        builder.add_node(ag, make_agent(stage))
        builder.add_node(post, make_post_guard(stage))

        # pre_guard → halt or agent
        builder.add_conditional_edges(
            pre,
            lambda s: "halt" if s.get("halted") else "proceed",
            {"proceed": ag, "halt": END},
        )

        # agent → post_guard
        builder.add_edge(ag, post)

        # post_guard → halt / retry / next
        # Uses _routing (non-accumulated) to avoid stale intervention_log bug
        def _post_route(s: LangGraphState) -> str:
            if s.get("halted"):
                return "halt"
            if s.get("_routing") in ("retry", "escalate"):
                return "retry"
            return "next"

        if i < len(stages) - 1:
            next_pre = f"pre_{stages[i + 1].name}"
            builder.add_conditional_edges(
                post, _post_route,
                {"next": next_pre, "retry": pre, "halt": END},
            )
        else:
            builder.add_conditional_edges(
                post, _post_route,
                {"next": END, "retry": pre, "halt": END},
            )

    builder.add_edge(START, f"pre_{stages[0].name}")
    return builder.compile()


def run_organism_langgraph(
    organism: Any,
    *,
    task: str,
    fast_model: Any | None = None,
    deep_model: Any | None = None,
    fast_model_name: str = "gemma4:latest",
    deep_model_name: str = "gemma4:latest",
    ollama_base_url: str = "http://localhost:11434/v1",
    verify_certificates: bool = True,
) -> LangGraphResult:
    """Compile and execute an organism in LangGraph in one call."""
    try:
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[deerflow]"
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

    graph = organism_to_langgraph(
        organism,
        fast_model=fast_model,
        deep_model=deep_model,
        fast_model_name=fast_model_name,
        deep_model_name=deep_model_name,
        ollama_base_url=ollama_base_url,
    )

    t0 = time.monotonic()
    result = graph.invoke({
        "messages": [HumanMessage(content=task)],
        "stage_outputs": {},
        "shared_state": dict(ctx),
        "current_stage": "",
        "use_deep": False,
        "halted": False,
        "intervention_log": [],
        "_routing": "",
    })
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Call on_run_complete for all components
    final_ctx = RunContext(result.get("shared_state", {}), watcher_key=watcher_key)
    stage_outputs = result.get("stage_outputs", {})
    final_output = list(stage_outputs.values())[-1] if stage_outputs else ""

    # Build a minimal SkillRunResult for on_run_complete
    from ..patterns.types import SkillRunResult
    run_result = SkillRunResult(
        task=task,
        final_output=final_output,
        stage_results=(),
        shared_state=dict(final_ctx),
    )
    for component in organism.components:
        component.on_run_complete(run_result, final_ctx)

    # Certificate verification
    cert_results: list[dict[str, Any]] = []
    if verify_certificates:
        for cert in organism.collect_certificates():
            verification = cert.verify()
            cert_results.append({
                "theorem": cert.theorem,
                "holds": verification.holds,
            })

    return LangGraphResult(
        output=final_output,
        stage_outputs=stage_outputs,
        interventions=result.get("intervention_log", []),
        timing_ms=elapsed_ms,
        certificates_verified=cert_results,
        metadata={
            "halted": result.get("halted", False),
            "stages_completed": list(stage_outputs.keys()),
        },
    )
