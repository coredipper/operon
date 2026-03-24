"""Operon Watcher Dashboard — interactive signal classification and intervention timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import gradio as gr

# ---------------------------------------------------------------------------
# Inline simulation (avoids heavy operon import for HF Space cold start)
# ---------------------------------------------------------------------------


class SignalCategory(Enum):
    EPISTEMIC = "epistemic"
    SOMATIC = "somatic"
    SPECIES_SPECIFIC = "species"


class InterventionKind(Enum):
    RETRY = "retry"
    ESCALATE = "escalate"
    HALT = "halt"


@dataclass(frozen=True)
class WatcherSignal:
    category: SignalCategory
    source: str
    stage_name: str
    value: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Intervention:
    kind: InterventionKind
    stage_name: str
    reason: str


@dataclass
class SimulatedStage:
    name: str
    role: str
    model: str  # "fast" or "deep"
    output: str
    epiplexity: float
    atp_fraction: float
    immune_threat: str  # "none", "suspicious", "confirmed", "critical"
    action_type: str = "EXECUTE"


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

def _build_normal_run() -> list[SimulatedStage]:
    return [
        SimulatedStage("intake", "Normalizer", "deterministic", "Parsed request", 0.7, 0.9, "none"),
        SimulatedStage("router", "Router", "fast", "Route: billing", 0.6, 0.85, "none"),
        SimulatedStage("analyst", "Analyst", "deep", "Risk assessment: low", 0.5, 0.75, "none"),
        SimulatedStage("reviewer", "Reviewer", "fast", "Approved", 0.55, 0.7, "none"),
    ]


def _build_stagnant_agent() -> list[SimulatedStage]:
    return [
        SimulatedStage("intake", "Normalizer", "deterministic", "Parsed request", 0.7, 0.9, "none"),
        SimulatedStage("planner", "Planner", "fast", "...", 0.12, 0.8, "none"),  # Critical epiplexity
        SimulatedStage("executor", "Executor", "deep", "Task completed (escalated)", 0.5, 0.7, "none"),
        SimulatedStage("checker", "Checker", "fast", "Verified", 0.6, 0.65, "none"),
    ]


def _build_budget_exhaustion() -> list[SimulatedStage]:
    return [
        SimulatedStage("s1", "Worker", "fast", "Step 1 done", 0.5, 0.6, "none"),
        SimulatedStage("s2", "Worker", "deep", "Step 2 done", 0.45, 0.3, "none"),
        SimulatedStage("s3", "Worker", "deep", "Step 3 started", 0.4, 0.08, "none"),  # ATP critical
        SimulatedStage("s4", "Worker", "fast", "Step 4 skipped", 0.5, 0.05, "none"),
    ]


def _build_immune_alert() -> list[SimulatedStage]:
    return [
        SimulatedStage("intake", "Parser", "deterministic", "Input parsed", 0.7, 0.9, "none"),
        SimulatedStage("agent", "Agent", "fast", "Suspicious output", 0.5, 0.85, "suspicious"),
        SimulatedStage("agent2", "Agent", "fast", "Malicious pattern", 0.45, 0.8, "critical"),
    ]


PRESETS = {
    "Normal Run": _build_normal_run,
    "Stagnant Agent (→ Escalate)": _build_stagnant_agent,
    "Budget Exhaustion (→ Low ATP)": _build_budget_exhaustion,
    "Immune Alert (→ Halt)": _build_immune_alert,
}


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _classify_signals(
    stage: SimulatedStage,
    epiplexity_thresh: float,
    atp_thresh: float,
    immune_levels: tuple[str, ...],
) -> list[WatcherSignal]:
    signals = []
    # Epistemic
    status = "healthy"
    if stage.epiplexity < 0.15:
        status = "critical"
    elif stage.epiplexity < epiplexity_thresh:
        status = "stagnant"
    signals.append(WatcherSignal(
        SignalCategory.EPISTEMIC, "epiplexity", stage.name, stage.epiplexity,
        {"status": status},
    ))
    # Somatic
    signals.append(WatcherSignal(
        SignalCategory.SOMATIC, "atp_store", stage.name, 1.0 - stage.atp_fraction,
        {"fraction": stage.atp_fraction},
    ))
    # Species
    threat_value = {"none": 0.0, "suspicious": 0.3, "confirmed": 0.7, "critical": 1.0}
    signals.append(WatcherSignal(
        SignalCategory.SPECIES_SPECIFIC, "immune", stage.name, threat_value.get(stage.immune_threat, 0.0),
        {"threat_level": stage.immune_threat},
    ))
    return signals


def _decide_intervention(
    stage: SimulatedStage,
    signals: list[WatcherSignal],
    intervention_count: int,
    total_stages: int,
    max_rate: float,
    immune_levels: tuple[str, ...],
) -> Intervention | None:
    # Convergence check
    if total_stages > 0 and intervention_count / total_stages > max_rate:
        return Intervention(InterventionKind.HALT, stage.name, "Non-convergence: intervention rate exceeded")
    # Immune
    if stage.immune_threat in immune_levels:
        return Intervention(InterventionKind.HALT, stage.name, f"Immune threat: {stage.immune_threat}")
    # Epistemic
    ep_status = None
    for s in signals:
        if s.category == SignalCategory.EPISTEMIC:
            ep_status = s.detail.get("status")
    if ep_status == "critical":
        if stage.model == "deep":
            return Intervention(InterventionKind.HALT, stage.name, "Critical epiplexity on deep model")
        return Intervention(InterventionKind.ESCALATE, stage.name, "Critical epiplexity → escalate")
    if ep_status == "stagnant" and stage.model == "fast":
        return Intervention(InterventionKind.ESCALATE, stage.name, "Stagnant on fast → escalate")
    # Failure
    if stage.action_type == "FAILURE":
        return Intervention(InterventionKind.RETRY, stage.name, "Stage failure → retry")
    return None


def _run_simulation(
    stages: list[SimulatedStage],
    epiplexity_thresh: float = 0.3,
    atp_thresh: float = 0.1,
    max_rate: float = 0.5,
    immune_levels: tuple[str, ...] = ("confirmed", "critical"),
) -> tuple[list[WatcherSignal], list[Intervention]]:
    all_signals: list[WatcherSignal] = []
    all_interventions: list[Intervention] = []
    for i, stage in enumerate(stages):
        sigs = _classify_signals(stage, epiplexity_thresh, atp_thresh, immune_levels)
        all_signals.extend(sigs)
        intv = _decide_intervention(stage, sigs, len(all_interventions), i + 1, max_rate, immune_levels)
        if intv:
            all_interventions.append(intv)
            if intv.kind == InterventionKind.HALT:
                break
    return all_signals, all_interventions


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CAT_COLORS = {
    "epistemic": "#2563eb",
    "somatic": "#16a34a",
    "species": "#dc2626",
}

_INTV_COLORS = {
    "retry": "#eab308",
    "escalate": "#f97316",
    "halt": "#ef4444",
}


def _signal_table_html(signals: list[WatcherSignal]) -> str:
    rows = ""
    for s in signals:
        color = _CAT_COLORS.get(s.category.value, "#888")
        rows += f"""<tr>
            <td><span style="color:{color};font-weight:600">{s.category.value}</span></td>
            <td>{s.source}</td>
            <td>{s.stage_name}</td>
            <td>{s.value:.2f}</td>
            <td style="font-size:12px;color:#888">{s.detail}</td>
        </tr>"""
    return f"""<table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Category</th>
            <th style="text-align:left;padding:8px">Source</th>
            <th style="text-align:left;padding:8px">Stage</th>
            <th style="text-align:left;padding:8px">Value</th>
            <th style="text-align:left;padding:8px">Detail</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _timeline_html(
    stages: list[SimulatedStage],
    signals: list[WatcherSignal],
    interventions: list[Intervention],
) -> str:
    intv_map = {i.stage_name: i for i in interventions}
    rows = ""
    for stage in stages:
        intv = intv_map.get(stage.name)
        intv_cell = ""
        if intv:
            color = _INTV_COLORS.get(intv.kind.value, "#888")
            intv_cell = f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{intv.kind.value.upper()}</span> {intv.reason}'
        stage_sigs = [s for s in signals if s.stage_name == stage.name]
        ep_val = next((s.value for s in stage_sigs if s.category == SignalCategory.EPISTEMIC), None)
        atp_val = next((s.detail.get("fraction") for s in stage_sigs if s.category == SignalCategory.SOMATIC), None)
        ep_str = f"{ep_val:.2f}" if ep_val is not None else "—"
        atp_str = f"{atp_val:.0%}" if atp_val is not None else "—"
        rows += f"""<tr style="border-bottom:1px solid #222">
            <td style="padding:10px;font-weight:600">{stage.name}</td>
            <td style="padding:10px">{stage.model}</td>
            <td style="padding:10px">{ep_str}</td>
            <td style="padding:10px">{atp_str}</td>
            <td style="padding:10px">{intv_cell or '<span style="color:#4a4">OK</span>'}</td>
        </tr>"""
    rate = f"{len(interventions)}/{len(stages)}" if stages else "0/0"
    return f"""<div style="margin-bottom:12px;font-size:13px;color:#888">
        Intervention rate: <b>{rate}</b>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Stage</th>
            <th style="text-align:left;padding:8px">Model</th>
            <th style="text-align:left;padding:8px">Epiplexity</th>
            <th style="text-align:left;padding:8px">ATP</th>
            <th style="text-align:left;padding:8px">Intervention</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def _load_preset(preset_name):
    if preset_name not in PRESETS:
        return "Select a preset.", ""
    stages = PRESETS[preset_name]()
    signals, interventions = _run_simulation(stages)
    signal_html = _signal_table_html(signals)
    timeline_html = _timeline_html(stages, signals, interventions)
    return signal_html, timeline_html


def _run_custom(preset_name, ep_thresh, atp_thresh, max_rate):
    if preset_name not in PRESETS:
        return "Select a preset first."
    stages = PRESETS[preset_name]()
    signals, interventions = _run_simulation(
        stages,
        epiplexity_thresh=ep_thresh,
        atp_thresh=atp_thresh,
        max_rate=max_rate,
    )
    return _timeline_html(stages, signals, interventions)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="Operon Watcher Dashboard") as demo:
        gr.Markdown("# Operon Watcher Dashboard\nSignal classification and intervention timeline for multi-stage workflows.")

        with gr.Tab("Signal Classification"):
            preset_dd = gr.Dropdown(choices=list(PRESETS.keys()), label="Preset Scenario", value="Normal Run")
            load_btn = gr.Button("Load & Run")
            signal_out = gr.HTML()
            timeline_out = gr.HTML()
            load_btn.click(_load_preset, inputs=[preset_dd], outputs=[signal_out, timeline_out])

        with gr.Tab("Intervention Timeline"):
            gr.Markdown("Select a preset in the first tab to see the intervention timeline above.")

        with gr.Tab("Live Configuration"):
            gr.Markdown("Adjust thresholds and re-run the selected scenario.")
            preset_dd2 = gr.Dropdown(choices=list(PRESETS.keys()), label="Preset", value="Normal Run")
            ep_slider = gr.Slider(minimum=0.05, maximum=0.8, value=0.3, step=0.05, label="Epiplexity Stagnant Threshold")
            atp_slider = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01, label="ATP Low Fraction")
            rate_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Max Intervention Rate")
            run_btn = gr.Button("Run with Custom Config")
            custom_out = gr.HTML()
            run_btn.click(_run_custom, inputs=[preset_dd2, ep_slider, atp_slider, rate_slider], outputs=[custom_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
