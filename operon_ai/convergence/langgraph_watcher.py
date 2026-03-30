"""LangGraph-compatible watcher node for DeerFlow integration.

Produces a state dict suitable for use as a LangGraph node, applying
Operon's watcher convergence detection logic to accumulated stage results.
"""

from __future__ import annotations

from typing import Any


def create_watcher_config(
    max_intervention_rate: float = 0.5,
    max_retries_per_stage: int = 1,
) -> dict[str, Any]:
    """Create a watcher configuration dict for use with operon_watcher_node."""
    return {
        "max_intervention_rate": max_intervention_rate,
        "max_retries_per_stage": max_retries_per_stage,
    }


def operon_watcher_node(
    state: dict[str, Any],
    *,
    watcher_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LangGraph-compatible node that applies Operon's watcher logic.

    Reads stage results from state, applies convergence detection,
    and writes intervention recommendations back to state.

    Uses a ``_watcher_cursor`` to track processed results and avoid
    duplicate signals on re-invocation.  Action strings use lowercase
    (``retry``, ``escalate``, ``halt``) matching Operon's canonical
    ``InterventionKind`` values.

    Convergence check uses the pre-intervention count (before appending
    any new intervention from the current stage), mirroring
    ``WatcherComponent._decide_intervention()``.
    """
    cfg = watcher_config or create_watcher_config()
    max_rate = cfg.get("max_intervention_rate", 0.5)
    max_retries = cfg.get("max_retries_per_stage", 1)

    stage_results = state.get("stage_results", [])
    signals = list(state.get("watcher_signals", []))
    interventions = list(state.get("watcher_interventions", []))
    cursor = state.get("_watcher_cursor", 0)

    # Only process new stage results since last invocation.
    new_results = stage_results[cursor:]

    for result in new_results:
        stage_name = result.get("stage_name", "unknown")
        action_type = result.get("action_type", "EXECUTE")

        # Classify signal.
        signals.append({
            "stage_name": stage_name,
            "action_type": action_type,
            "category": "epistemic" if action_type == "EXECUTE" else "somatic",
        })

        # Convergence check BEFORE appending new intervention (mirrors WatcherComponent).
        total_stages = cursor + len(new_results)
        pre_intervention_count = len(interventions)
        intervention_rate = pre_intervention_count / total_stages if total_stages > 0 else 0.0

        if intervention_rate > max_rate and total_stages > 0:
            # Only add HALT if not already halted.
            if not any(i.get("action") == "halt" for i in interventions):
                interventions.append({
                    "stage_name": "__convergence__",
                    "action": "halt",
                    "reason": f"intervention rate {intervention_rate:.2f} exceeds {max_rate}",
                })
            return {
                "watcher_signals": signals,
                "watcher_interventions": interventions,
                "watcher_summary": _summary(total_stages, signals, interventions),
                "should_halt": True,
                "_watcher_cursor": len(stage_results),
            }

        # Check for failure → intervention (using lowercase action strings).
        if action_type == "FAILURE":
            stage_retries = sum(
                1 for i in interventions
                if i.get("stage_name") == stage_name and i.get("action") == "retry"
            )
            if stage_retries < max_retries:
                interventions.append({
                    "stage_name": stage_name,
                    "action": "retry",
                    "reason": f"failure on {stage_name}, retry {stage_retries + 1}/{max_retries}",
                })
            else:
                interventions.append({
                    "stage_name": stage_name,
                    "action": "escalate",
                    "reason": f"max retries ({max_retries}) exceeded for {stage_name}",
                })

    total_stages = len(stage_results)
    intervention_rate = len(interventions) / total_stages if total_stages > 0 else 0.0

    return {
        "watcher_signals": signals,
        "watcher_interventions": interventions,
        "watcher_summary": _summary(total_stages, signals, interventions),
        "should_halt": False,
        "_watcher_cursor": len(stage_results),
    }


def _summary(total_stages: int, signals: list, interventions: list) -> dict[str, Any]:
    intervention_rate = len(interventions) / total_stages if total_stages > 0 else 0.0
    return {
        "total_stages": total_stages,
        "total_signals": len(signals),
        "total_interventions": len(interventions),
        "intervention_rate": intervention_rate,
        "convergent": intervention_rate <= 0.5,
    }
