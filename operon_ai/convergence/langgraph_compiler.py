"""LangGraph compiler — wrap SkillOrganism.run() as a LangGraph node.

Instead of reimplementing the organism's execution logic as LangGraph
nodes, this compiler wraps ``organism.run()`` directly. All structural
guarantees (CertificateGate, VerifierComponent, WatcherComponent,
halt_on_block, retry, escalation) are handled by the organism's own
run loop — the same code path tested by 1856+ unit tests.

LangGraph provides the execution host and graph topology. The organism
provides the structural guarantees.

Requires ``langgraph`` and ``langchain-openai``::

    pip install operon-ai[deerflow]

All external imports are lazy.
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
# State schema
# ---------------------------------------------------------------------------


class LangGraphState(TypedDict):
    """LangGraph state for the organism wrapper."""

    messages: Annotated[list, operator.add]
    output: str
    stage_outputs: dict[str, str]
    halted: bool
    intervention_log: list[dict[str, Any]]
    run_complete: bool


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
    """Compile a SkillOrganism into a LangGraph StateGraph.

    The graph has a single node that calls ``organism.run()``.
    All structural guarantees run inside the organism's own run loop.

    Parameters
    ----------
    organism:
        A ``SkillOrganism`` with stages and components attached.

    Returns
    -------
    CompiledStateGraph
        A LangGraph graph that can be invoked or streamed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[deerflow]"
        ) from e

    def run_organism(state: LangGraphState) -> dict:
        """Execute the full organism pipeline."""
        # Extract task from messages
        task = ""
        for m in state.get("messages", []):
            if isinstance(m, HumanMessage):
                task = m.content
                break

        # Run the organism — all guarantees enforced internally
        result = organism.run(task)

        # Extract outputs
        stage_outputs = {
            sr.stage_name: (sr.output if isinstance(sr.output, str) else str(sr.output))
            for sr in result.stage_results
        }
        final_output = result.final_output
        if not isinstance(final_output, str):
            final_output = str(final_output) if final_output is not None else ""

        # Extract intervention history from watcher (if present)
        interventions = []
        for component in organism.components:
            if hasattr(component, "interventions"):
                for intv in component.interventions:
                    interventions.append({
                        "stage": intv.stage_name,
                        "kind": intv.kind.value,
                        "reason": intv.reason,
                    })

        halted = bool(result.shared_state.get("_blocked_by"))

        return {
            "output": final_output,
            "stage_outputs": stage_outputs,
            "halted": halted,
            "intervention_log": interventions,
            "run_complete": True,
        }

    builder = StateGraph(LangGraphState)
    builder.add_node("organism", run_organism)
    builder.add_edge(START, "organism")
    builder.add_edge("organism", END)
    return builder.compile()


def run_organism_langgraph(
    organism: Any,
    *,
    task: str,
    verify_certificates: bool = True,
) -> LangGraphResult:
    """Compile and execute an organism in LangGraph in one call.

    Parameters
    ----------
    organism:
        A ``SkillOrganism`` with stages and components attached.
    task:
        The user task / prompt to execute.
    verify_certificates:
        If ``True``, verify certificates post-execution.
    """
    try:
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install operon-ai[deerflow]"
        ) from e

    graph = organism_to_langgraph(organism)

    t0 = time.monotonic()
    result = graph.invoke({
        "messages": [HumanMessage(content=task)],
        "output": "",
        "stage_outputs": {},
        "halted": False,
        "intervention_log": [],
        "run_complete": False,
    })
    elapsed_ms = (time.monotonic() - t0) * 1000

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
        output=result.get("output", ""),
        stage_outputs=result.get("stage_outputs", {}),
        interventions=result.get("intervention_log", []),
        timing_ms=elapsed_ms,
        certificates_verified=cert_results,
        metadata={
            "halted": result.get("halted", False),
            "stages_completed": list(result.get("stage_outputs", {}).keys()),
            "run_complete": result.get("run_complete", False),
        },
    )
