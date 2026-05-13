"""DeerFlow executor — run compiled organisms in DeerFlow's LangGraph runtime.

Executes compiled dicts from :func:`organism_to_deerflow` against the
actual DeerFlow framework, with Operon's watcher monitoring the execution
stream and certificate verification post-run.

Requires ``deerflow-harness`` (Python >=3.12)::

    pip install "git+https://github.com/bytedance/deer-flow.git#subdirectory=backend/packages/harness"

All DeerFlow imports are lazy — the module is importable without DeerFlow
installed.  Use :data:`HAS_DEERFLOW` to check availability at runtime.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .langgraph_watcher import operon_watcher_node

# ---------------------------------------------------------------------------
# Availability guard (lightweight — no heavy imports at module level)
# ---------------------------------------------------------------------------

try:
    import deerflow  # noqa: F401

    HAS_DEERFLOW = True
except ImportError:
    HAS_DEERFLOW = False


# ---------------------------------------------------------------------------
# Result type and Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeerFlowExecutionConfig:
    """Configuration for executing a compiled organism in DeerFlow."""

    model: Any | None = None
    """Pre-built ``BaseChatModel`` instance. If ``None``, one is created from *model_name* via ``ChatOpenAI`` pointed at Ollama."""

    model_name: str = "gemma4:latest"
    """Ollama model name (used when *model* is ``None``)."""

    ollama_base_url: str = "http://localhost:11434/v1"
    """Ollama OpenAI-compatible endpoint."""

    enable_watcher: bool = True
    """If ``True``, run ``operon_watcher_node`` on each stream chunk."""

    watcher_config: dict[str, Any] | None = None
    """Config dict for the watcher (max_intervention_rate, max_retries)."""

    verify_certificates: bool = True
    """If ``True``, verify certificates from the compiled dict post-run."""


@dataclass(frozen=True)
class DeerFlowResult:
    """Result of executing a compiled organism in DeerFlow."""

    output: str
    """Final text output from the agent."""

    messages: tuple[dict[str, Any], ...]
    """Raw LangGraph messages from the execution stream."""

    timing_ms: float
    """Wall-clock execution time in milliseconds."""

    watcher_summary: dict[str, Any]
    """Watcher convergence summary (signals, interventions, rates)."""

    certificates_verified: tuple[dict[str, Any], ...]
    """Certificate verification results post-execution."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary execution metadata."""


# ---------------------------------------------------------------------------
# Schema transformation
# ---------------------------------------------------------------------------


def _compiled_to_agent_kwargs(compiled: dict[str, Any]) -> dict[str, Any]:
    """Transform Operon's compiled dict to ``create_deerflow_agent`` kwargs.

    Mapping:
    - ``assistant_id`` → ``name``
    - ``skills`` → system prompt content
    - ``sub_agents`` → system prompt team description
    - ``sandbox`` → ``RuntimeFeatures(sandbox=True/False)``
    """
    skills = compiled.get("skills", [])
    sub_agents = compiled.get("sub_agents", [])
    sandbox = compiled.get("sandbox", "none")

    # Build system prompt from skills and sub-agent descriptions.
    parts = ["You are an AI assistant."]
    if skills:
        parts.append("\nYour capabilities:")
        for skill in skills:
            parts.append(f"- {skill}")
    if sub_agents:
        parts.append("\nYou coordinate the following team members:")
        for sa in sub_agents:
            sa_skills = ", ".join(sa.get("skills", []))
            parts.append(f"- {sa['name']} ({sa['role']}): {sa_skills}")
    system_prompt = "\n".join(parts)

    return {
        "name": compiled.get("assistant_id", "operon_agent"),
        "system_prompt": system_prompt,
        "sandbox": sandbox != "none",
        "subagent": False,  # Phase 1: single-agent only
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_deerflow(
    compiled: dict[str, Any],
    *,
    task: str,
    exec_config: DeerFlowExecutionConfig | None = None,
    # Legacy kwargs for backward compatibility
    model: Any | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    enable_watcher: bool | None = None,
    watcher_config: dict[str, Any] | None = None,
    verify_certificates: bool | None = None,
) -> DeerFlowResult:
    """Execute a compiled organism dict in DeerFlow's LangGraph runtime.

    Note: execution time is bounded by the LangGraph ``recursion_limit``
    derived from the compiled dict's timeout, not by a wall-clock timer.

    Parameters
    ----------
    compiled:
        Dict produced by :func:`organism_to_deerflow`.
    task:
        The user task / prompt to execute.
    exec_config:
        Execution configuration for the DeerFlow run.
    model, model_name, ollama_base_url, enable_watcher, watcher_config, verify_certificates:
        Legacy kwargs for backward compatibility. Prefer `exec_config`.

    Returns
    -------
    DeerFlowResult
        Execution output, watcher summary, and certificate verifications.

    Raises
    ------
    ImportError
        If DeerFlow is not installed.
    """
    if exec_config is None:
        import warnings

        # Check if any legacy kwargs were provided
        legacy_kwargs_used = any(
            v is not None for v in [model, model_name, ollama_base_url, enable_watcher, watcher_config, verify_certificates]
        )
        if legacy_kwargs_used:
            warnings.warn(
                "Passing configuration directly as keyword arguments to execute_deerflow is deprecated. "
                "Use the 'exec_config' parameter with a DeerFlowExecutionConfig object instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Build config from legacy kwargs, falling back to defaults if not provided
        kwargs_to_pass = {}
        if model is not None:
            kwargs_to_pass["model"] = model
        if model_name is not None:
            kwargs_to_pass["model_name"] = model_name
        if ollama_base_url is not None:
            kwargs_to_pass["ollama_base_url"] = ollama_base_url
        if enable_watcher is not None:
            kwargs_to_pass["enable_watcher"] = enable_watcher
        if watcher_config is not None:
            kwargs_to_pass["watcher_config"] = watcher_config
        if verify_certificates is not None:
            kwargs_to_pass["verify_certificates"] = verify_certificates

        exec_config = DeerFlowExecutionConfig(**kwargs_to_pass)

    # --- Reject unsupported multi-agent configs ---
    sub_agents = compiled.get("sub_agents", [])
    if sub_agents:
        raise ValueError(
            f"execute_deerflow() does not support multi-agent execution "
            f"({len(sub_agents)} sub-agents). Use compile_guarded_graph() "
            f"from guarded_graph.py for multi-stage pipelines with "
            f"structural guarantees."
        )

    # --- Lazy imports (all DeerFlow + LangChain deps) ---
    try:
        from deerflow.agents.factory import create_deerflow_agent
        from deerflow.agents.features import RuntimeFeatures
    except ImportError as e:
        raise ImportError(
            "DeerFlow is required for execution but not installed. "
            "Install with: pip install "
            '"git+https://github.com/bytedance/deer-flow.git'
            '#subdirectory=backend/packages/harness"'
        ) from e

    # --- Model resolution ---
    if exec_config.model is None:
        from langchain_openai import ChatOpenAI

        thinking = compiled.get("config", {}).get("thinking_enabled", False)
        model = ChatOpenAI(
            base_url=exec_config.ollama_base_url,
            model=exec_config.model_name,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            temperature=0.0,
            max_tokens=2048 if thinking else 500,
        )
    else:
        model = exec_config.model

    # --- Schema transformation ---
    kwargs = _compiled_to_agent_kwargs(compiled)
    features = RuntimeFeatures(
        sandbox=kwargs.pop("sandbox"),
        subagent=kwargs.pop("subagent"),
    )

    # --- Create DeerFlow agent ---
    agent = create_deerflow_agent(
        model=model,
        system_prompt=kwargs["system_prompt"],
        features=features,
        name=kwargs["name"],
    )

    # --- Single-call execution with watcher ---
    from langchain_core.messages import HumanMessage

    state = {"messages": [HumanMessage(content=task)]}
    config = {"recursion_limit": compiled.get("recursion_limit", 100)}

    watcher_state: dict[str, Any] = {
        "stage_results": [],
        "watcher_signals": [],
        "watcher_interventions": [],
        "_watcher_cursor": 0,
    }

    collected_messages: list[dict[str, Any]] = []
    final_text = ""
    action_type = "EXECUTE"
    t0 = time.monotonic()

    try:
        result_state = agent.invoke(state, config=config)
        messages = result_state.get("messages", [])
        for msg in messages:
            msg_dict = {
                "type": getattr(msg, "type", "unknown"),
                "content": getattr(msg, "content", ""),
            }
            collected_messages.append(msg_dict)
        if messages:
            final_text = getattr(messages[-1], "content", "") or ""
    except Exception as exc:
        action_type = "FAILURE"
        final_text = f"Execution failed: {exc}"
        collected_messages.append({"type": "error", "content": final_text})

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Feed one stage result to watcher (one agent = one stage)
    if exec_config.enable_watcher:
        watcher_state["stage_results"].append({
            "stage_name": kwargs["name"],
            "action_type": action_type,
        })
        watcher_update = operon_watcher_node(
            watcher_state, watcher_config=exec_config.watcher_config,
        )
        watcher_state.update(watcher_update)

    # --- Certificate verification ---
    cert_results: list[dict[str, Any]] = []
    if exec_config.verify_certificates:
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

    return DeerFlowResult(
        output=final_text,
        messages=tuple(collected_messages),
        timing_ms=elapsed_ms,
        watcher_summary=watcher_state.get("watcher_summary", {}),
        certificates_verified=tuple(cert_results),
        metadata={
            "model_name": exec_config.model_name,
            "agent_name": kwargs["name"],
            "enable_watcher": exec_config.enable_watcher,
            "recursion_limit": config["recursion_limit"],
        },
    )
