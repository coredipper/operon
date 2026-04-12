"""Guarded LangGraph compiler — Operon structural guarantees as graph topology.

Compiles a :class:`SkillOrganism` into a LangGraph ``StateGraph`` where each
Operon stage becomes a triple:

1. **Pre-guard node** — runs ``CertificateGateComponent.on_stage_start()``
   to check genome integrity before the LLM call.
2. **Agent node** — calls the LLM with the stage's system prompt.
3. **Post-guard node** — runs ``VerifierComponent.on_stage_result()`` +
   ``WatcherComponent._decide_intervention()``, returns a routing decision.

Conditional edges enforce interventions natively in LangGraph:

- **HALT** → ``END`` (LangGraph stops the graph)
- **ESCALATE** → re-run the agent node with the deep model
- **RETRY** → re-run the agent node with the same model
- **OK** → proceed to the next stage's pre-guard

This is the real Prop 5.1 claim: structural properties are preserved under
compilation because they ARE the compiled graph topology, not annotations
on an opaque execution.

Requires ``langgraph`` and ``langchain-openai`` (installed with DeerFlow)::

    pip install operon-ai[deerflow]

All external imports are lazy — the module is importable without LangGraph.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

try:
    from langgraph.graph import StateGraph  # noqa: F401

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# State schema (must be at module level for LangGraph's type hint resolution)
# ---------------------------------------------------------------------------


class GuardedState(TypedDict):
    """LangGraph state flowing through the guarded graph."""

    messages: Annotated[list, operator.add]
    stage_outputs: dict[str, str]
    current_stage: str
    current_stage_idx: int
    use_deep: bool
    halted: bool
    intervention_log: Annotated[list, operator.add]
    watcher_state: dict[str, Any]
    retry_count: int
    _pending_output: str
    _pending_action: str


@dataclass(frozen=True)
class GuardedGraphResult:
    """Result of executing a guarded LangGraph."""

    output: str
    stage_outputs: dict[str, str]
    interventions: list[dict[str, Any]]
    watcher_summary: dict[str, Any]
    certificates_verified: list[dict[str, Any]]
    timing_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_guarded_graph(
    compiled: dict[str, Any],
    *,
    fast_model: Any | None = None,
    deep_model: Any | None = None,
    fast_model_name: str = "gemma4:latest",
    deep_model_name: str = "gemma4:latest",
    ollama_base_url: str = "http://localhost:11434/v1",
    watcher_config: dict[str, Any] | None = None,
    genome: Any | None = None,
    repair: Any | None = None,
    checkpoint: Any | None = None,
    rubric: Any | None = None,
) -> Any:
    """Compile an Operon organism dict into a guarded LangGraph.

    Each stage becomes a guarded triple: pre-check → LLM → post-check,
    with conditional edges enforcing HALT/ESCALATE/RETRY.

    Parameters
    ----------
    compiled:
        Dict produced by :func:`organism_to_deerflow`.
    fast_model / deep_model:
        Pre-built ``BaseChatModel`` instances.  If ``None``, created from
        model names via ``ChatOpenAI`` pointed at Ollama.
    genome / repair / checkpoint:
        If all three are provided, ``CertificateGateComponent`` runs as
        a pre-guard on each stage.
    rubric:
        If provided, ``VerifierComponent`` runs as a post-guard evaluating
        output quality.  Callable: ``(output: str, stage_name: str) -> float``.
    watcher_config:
        Config dict for convergence detection thresholds.

    Returns
    -------
    CompiledStateGraph
        A LangGraph graph that can be invoked or streamed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[deerflow]"
        ) from e

    from .langgraph_watcher import operon_watcher_node

    # --- Model resolution ---
    if fast_model is None:
        fast_model = ChatOpenAI(
            base_url=ollama_base_url,
            model=fast_model_name,
            api_key="ollama",
            temperature=0.0,
            max_tokens=1024,
        )
    if deep_model is None:
        deep_model = ChatOpenAI(
            base_url=ollama_base_url,
            model=deep_model_name,
            api_key="ollama",
            temperature=0.0,
            max_tokens=2048,
        )

    # --- Extract stages from compiled dict ---
    lead_name = compiled.get("assistant_id", "agent")
    lead_skills = compiled.get("skills", [])
    sub_agents = compiled.get("sub_agents", [])

    stages: list[dict[str, Any]] = [
        {"name": lead_name, "role": "lead", "instructions": " ".join(lead_skills)},
    ]
    for sa in sub_agents:
        stages.append({
            "name": sa["name"],
            "role": sa.get("role", "worker"),
            "instructions": " ".join(sa.get("skills", [])),
        })

    # --- Node factories ---
    max_retries = 2
    if watcher_config:
        max_retries = watcher_config.get("max_retries_per_stage", 2)

    def make_pre_guard(stage_info: dict) -> Any:
        """Factory for pre-guard nodes (CertificateGate check)."""
        stage_name = stage_info["name"]

        def pre_guard(state: GuardedState) -> dict:
            if genome is not None and repair is not None and checkpoint is not None:
                damage = repair.scan(genome, checkpoint)
                if damage:
                    return {
                        "halted": True,
                        "messages": [f"[HALT] genome corruption before {stage_name}: {len(damage)} damage(s)"],
                        "intervention_log": [{
                            "stage": stage_name,
                            "kind": "halt",
                            "reason": f"genome corruption: {len(damage)} damage(s)",
                            "phase": "pre_guard",
                        }],
                    }
            return {"current_stage": stage_name}

        return pre_guard

    def make_agent(stage_info: dict) -> Any:
        """Factory for agent nodes (LLM call)."""
        stage_name = stage_info["name"]
        instructions = stage_info["instructions"]

        def agent(state: GuardedState) -> dict:
            model = deep_model if state.get("use_deep") else fast_model
            sys_msg = SystemMessage(content=f"Role: {stage_info['role']}. {instructions}")

            # Build input: system + task + previous stage outputs
            input_msgs = [sys_msg]
            # Add original task (first human message)
            for m in state.get("messages", []):
                if isinstance(m, HumanMessage):
                    input_msgs.append(m)
                    break
            # Add previous ACCEPTED stage outputs as context (not pending)
            for prev_name, prev_output in state.get("stage_outputs", {}).items():
                if prev_name != stage_name:  # Exclude own prior rejected output
                    input_msgs.append(AIMessage(content=f"[{prev_name}]: {prev_output}"))

            was_escalated = state.get("use_deep", False)
            action_type = "EXECUTE"
            try:
                response = model.invoke(input_msgs)
                output_text = response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                output_text = f"Error: {exc}"
                action_type = "FAILURE"
                response = AIMessage(content=output_text)

            # Write to a pending key — post_guard decides whether to accept
            return {
                "messages": [response],
                "_pending_output": output_text,
                "_pending_action": action_type,
                "use_deep": False,  # Reset flag
                "retry_count": 1 if was_escalated else 0,
            }

        return agent

    def make_post_guard(stage_info: dict) -> Any:
        """Factory for post-guard nodes (Verifier + Watcher check)."""
        stage_name = stage_info["name"]

        def post_guard(state: GuardedState) -> dict:
            output = state.get("_pending_output", "")
            action_type = state.get("_pending_action", "EXECUTE")
            interventions: list[dict] = []

            # --- Verifier quality check ---
            quality = 1.0
            if rubric is not None and action_type != "FAILURE":
                try:
                    quality = rubric(output, stage_name)
                    quality = max(0.0, min(1.0, quality))
                except Exception:
                    quality = 0.5

            # --- Feed watcher with real action type ---
            ws = dict(state.get("watcher_state", {
                "stage_results": [],
                "watcher_signals": [],
                "watcher_interventions": [],
                "_watcher_cursor": 0,
            }))
            ws["stage_results"] = list(ws.get("stage_results", []))
            ws["stage_results"].append({
                "stage_name": stage_name,
                "action_type": action_type,
                "quality": quality,
            })
            watcher_update = operon_watcher_node(ws, watcher_config=watcher_config)
            ws.update(watcher_update)

            # --- Decide intervention ---
            model_alias = "deep" if state.get("use_deep") else "fast"
            already_escalated = state.get("retry_count", 0) > 0

            # Quality-based escalation (adaptive immune).
            # Only escalate once per stage.
            if quality < 0.5 and model_alias == "fast" and not already_escalated:
                interventions.append({
                    "stage": stage_name,
                    "kind": "escalate",
                    "reason": f"low quality ({quality:.2f}) on fast model",
                    "phase": "post_guard",
                })
                return {
                    "use_deep": True,
                    "intervention_log": interventions,
                    "watcher_state": ws,
                }

            # Watcher-based halt
            if watcher_update.get("should_halt"):
                interventions.append({
                    "stage": stage_name,
                    "kind": "halt",
                    "reason": "watcher convergence failure",
                    "phase": "post_guard",
                })
                return {
                    "halted": True,
                    "intervention_log": interventions,
                    "watcher_state": ws,
                }

            # Read NEW watcher interventions for THIS stage only.
            # Compare against pre-call count to avoid reacting to prior stages.
            old_interventions = state.get("watcher_state", {}).get(
                "watcher_interventions", []
            )
            new_watcher_interventions = ws.get("watcher_interventions", [])[
                len(old_interventions):
            ]
            for wi in new_watcher_interventions:
                if wi.get("stage_name") != stage_name:
                    continue
                action = wi.get("action", "")
                if action in ("escalate", "retry") and not already_escalated:
                    interventions.append({
                        "stage": stage_name,
                        "kind": "escalate",
                        "reason": wi.get("reason", "watcher escalation"),
                        "phase": "post_guard",
                    })
                    return {
                        "use_deep": True,
                        "intervention_log": interventions,
                        "watcher_state": ws,
                    }

            # If already escalated and still failing, halt — don't accept
            # error output into the pipeline.
            if already_escalated and action_type == "FAILURE":
                interventions.append({
                    "stage": stage_name,
                    "kind": "halt",
                    "reason": "stage failed on both fast and deep models",
                    "phase": "post_guard",
                })
                return {
                    "halted": True,
                    "intervention_log": interventions,
                    "watcher_state": ws,
                }

            # --- Accept output: move from pending to stage_outputs ---
            new_outputs = dict(state.get("stage_outputs", {}))
            new_outputs[stage_name] = output

            return {
                "stage_outputs": new_outputs,
                "watcher_state": ws,
                "intervention_log": interventions,
            }

        return post_guard

    # --- Build the graph ---
    builder = StateGraph(GuardedState)

    for i, stage in enumerate(stages):
        name = stage["name"]
        pre_name = f"pre_guard_{name}"
        agent_name = f"agent_{name}"
        post_name = f"post_guard_{name}"

        builder.add_node(pre_name, make_pre_guard(stage))
        builder.add_node(agent_name, make_agent(stage))
        builder.add_node(post_name, make_post_guard(stage))

        # Wire: pre_guard → conditional (halt or proceed to agent)
        def make_pre_route(an: str):
            def route(state: GuardedState) -> str:
                return "halt" if state.get("halted") else "proceed"
            return route

        builder.add_conditional_edges(
            pre_name,
            make_pre_route(agent_name),
            {"proceed": agent_name, "halt": END},
        )

        # Wire: agent → post_guard
        builder.add_edge(agent_name, post_name)

        # Wire: post_guard → conditional (halt/escalate/next)
        # Escalation routes back through pre_guard so the certificate
        # gate runs before every LLM call, not just the first attempt.
        if i < len(stages) - 1:
            next_pre = f"pre_guard_{stages[i + 1]['name']}"

            def make_post_route(pn: str, npre: str):
                def route(state: GuardedState) -> str:
                    if state.get("halted"):
                        return "halt"
                    if state.get("use_deep"):
                        return "escalate"
                    return "next"
                return route

            builder.add_conditional_edges(
                post_name,
                make_post_route(pre_name, next_pre),
                {"next": next_pre, "escalate": pre_name, "halt": END},
            )
        else:
            # Last stage: halt or done
            def make_final_route(pn: str):
                def route(state: GuardedState) -> str:
                    if state.get("halted"):
                        return "halt"
                    if state.get("use_deep"):
                        return "escalate"
                    return "done"
                return route

            builder.add_conditional_edges(
                post_name,
                make_final_route(pre_name),
                {"done": END, "escalate": pre_name, "halt": END},
            )

    # Entry edge: START → first pre_guard
    builder.add_edge(START, f"pre_guard_{stages[0]['name']}")

    return builder.compile()


def run_guarded_graph(
    compiled: dict[str, Any],
    *,
    task: str,
    fast_model: Any | None = None,
    deep_model: Any | None = None,
    fast_model_name: str = "gemma4:latest",
    deep_model_name: str = "gemma4:latest",
    ollama_base_url: str = "http://localhost:11434/v1",
    watcher_config: dict[str, Any] | None = None,
    genome: Any | None = None,
    repair: Any | None = None,
    checkpoint: Any | None = None,
    rubric: Any | None = None,
    verify_certificates: bool = True,
) -> GuardedGraphResult:
    """Compile and execute a guarded LangGraph in one call.

    Convenience wrapper around :func:`compile_guarded_graph` +
    ``graph.invoke()``.
    """
    try:
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[deerflow]"
        ) from e

    graph = compile_guarded_graph(
        compiled,
        fast_model=fast_model,
        deep_model=deep_model,
        fast_model_name=fast_model_name,
        deep_model_name=deep_model_name,
        ollama_base_url=ollama_base_url,
        watcher_config=watcher_config,
        genome=genome,
        repair=repair,
        checkpoint=checkpoint,
        rubric=rubric,
    )

    t0 = time.monotonic()
    result = graph.invoke({
        "messages": [HumanMessage(content=task)],
        "stage_outputs": {},
        "current_stage": "",
        "current_stage_idx": 0,
        "use_deep": False,
        "halted": False,
        "intervention_log": [],
        "watcher_state": {},
        "retry_count": 0,
        "_pending_output": "",
        "_pending_action": "",
    })
    elapsed_ms = (time.monotonic() - t0) * 1000

    # --- Certificate verification ---
    cert_results: list[dict[str, Any]] = []
    if verify_certificates:
        from ..core.certificate import certificate_from_dict

        for cd in compiled.get("certificates", []):
            try:
                cert = certificate_from_dict(cd)
                verification = cert.verify()
                cert_results.append({
                    "theorem": cert.theorem,
                    "holds": verification.holds,
                })
            except Exception as exc:
                cert_results.append({
                    "theorem": cd.get("theorem", "unknown"),
                    "holds": False,
                    "error": str(exc),
                })

    # Extract final output
    stage_outputs = result.get("stage_outputs", {})
    final_output = ""
    if stage_outputs:
        last_key = list(stage_outputs.keys())[-1]
        final_output = stage_outputs[last_key]

    return GuardedGraphResult(
        output=final_output,
        stage_outputs=stage_outputs,
        interventions=result.get("intervention_log", []),
        watcher_summary=result.get("watcher_state", {}).get("watcher_summary", {}),
        certificates_verified=cert_results,
        timing_ms=elapsed_ms,
        metadata={
            "halted": result.get("halted", False),
            "stages_completed": list(stage_outputs.keys()),
        },
    )
