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

    Input state keys:
    - ``stage_results``: list of {stage_name, output, action_type}
    - ``watcher_signals``: list (accumulated)
    - ``watcher_interventions``: list (accumulated)

    Output state keys (merged):
    - ``watcher_signals``: updated with new signals
    - ``watcher_interventions``: updated with new intervention (if any)
    - ``watcher_summary``: current summary dict
    - ``should_halt``: bool (True if convergence threshold exceeded)
    """
    cfg = watcher_config or create_watcher_config()
    max_rate = cfg.get("max_intervention_rate", 0.5)
    max_retries = cfg.get("max_retries_per_stage", 1)

    stage_results = state.get("stage_results", [])
    signals = list(state.get("watcher_signals", []))
    interventions = list(state.get("watcher_interventions", []))

    total_stages = len(stage_results)
    total_interventions = len(interventions)

    # Process the latest stage result (if any new ones since last call).
    new_signals: list[dict[str, Any]] = []
    new_intervention: dict[str, Any] | None = None

    if stage_results:
        latest = stage_results[-1]
        stage_name = latest.get("stage_name", "unknown")
        action_type = latest.get("action_type", "EXECUTE")

        # Classify signal.
        signal = {
            "stage_name": stage_name,
            "action_type": action_type,
            "category": "epistemic" if action_type == "EXECUTE" else "somatic",
        }
        new_signals.append(signal)

        # Check for failure → intervention.
        if action_type == "FAILURE":
            # Count retries for this stage.
            stage_retries = sum(
                1 for i in interventions
                if i.get("stage_name") == stage_name and i.get("action") == "RETRY"
            )
            if stage_retries < max_retries:
                new_intervention = {
                    "stage_name": stage_name,
                    "action": "RETRY",
                    "reason": f"failure on {stage_name}, retry {stage_retries + 1}/{max_retries}",
                }
            else:
                new_intervention = {
                    "stage_name": stage_name,
                    "action": "ESCALATE",
                    "reason": f"max retries ({max_retries}) exceeded for {stage_name}",
                }

    # Update accumulated state.
    signals.extend(new_signals)
    if new_intervention is not None:
        interventions.append(new_intervention)

    # Convergence check.
    total_interventions = len(interventions)
    intervention_rate = total_interventions / total_stages if total_stages > 0 else 0.0
    should_halt = intervention_rate > max_rate and total_stages > 0

    if should_halt:
        interventions.append({
            "stage_name": "__convergence__",
            "action": "HALT",
            "reason": f"intervention rate {intervention_rate:.2f} exceeds {max_rate}",
        })

    return {
        "watcher_signals": signals,
        "watcher_interventions": interventions,
        "watcher_summary": {
            "total_stages": total_stages,
            "total_signals": len(signals),
            "total_interventions": len(interventions),
            "intervention_rate": intervention_rate,
            "convergent": not should_halt,
        },
        "should_halt": should_halt,
    }
