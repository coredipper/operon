"""Operon Developmental Staging Dashboard."""

from __future__ import annotations
from enum import Enum

import gradio as gr


class Stage(Enum):
    EMBRYONIC = "embryonic"
    JUVENILE = "juvenile"
    ADOLESCENT = "adolescent"
    MATURE = "mature"

STAGES = [Stage.EMBRYONIC, Stage.JUVENILE, Stage.ADOLESCENT, Stage.MATURE]
PLASTICITY = {Stage.EMBRYONIC: 1.0, Stage.JUVENILE: 0.75, Stage.ADOLESCENT: 0.5, Stage.MATURE: 0.25}
COLORS = {Stage.EMBRYONIC: "#4ade80", Stage.JUVENILE: "#60a5fa", Stage.ADOLESCENT: "#f59e0b", Stage.MATURE: "#a78bfa"}

CRITICAL_PERIODS = [
    ("rapid_learning", Stage.EMBRYONIC, Stage.JUVENILE, "Fast template adoption from peers"),
    ("tool_exploration", Stage.JUVENILE, Stage.ADOLESCENT, "Try new tools freely"),
    ("social_bonding", Stage.EMBRYONIC, Stage.ADOLESCENT, "Form trust with peers"),
]

TOOLS = [
    ("basic_tool", Stage.EMBRYONIC, "Simple text processing"),
    ("analyzer", Stage.JUVENILE, "Data analysis toolkit"),
    ("planner", Stage.ADOLESCENT, "Strategic planning"),
    ("orchestrator", Stage.MATURE, "Full multi-agent orchestration"),
]


def _get_stage(consumed_frac, juv_t, adol_t, mat_t):
    if consumed_frac >= mat_t / 100: return Stage.MATURE
    if consumed_frac >= adol_t / 100: return Stage.ADOLESCENT
    if consumed_frac >= juv_t / 100: return Stage.JUVENILE
    return Stage.EMBRYONIC


def _run_lifecycle(max_ops, juv_t, adol_t, mat_t):
    rows = ""
    last = None
    transitions = []
    for tick in range(1, int(max_ops) + 1):
        consumed = tick / max_ops
        stage = _get_stage(consumed, juv_t, adol_t, mat_t)
        if stage != last:
            if last is not None:
                transitions.append((tick, last, stage))
            last = stage

    for tick, old, new in transitions:
        color = COLORS[new]
        rows += f"""<tr>
            <td style="padding:8px">Tick {tick}</td>
            <td style="padding:8px;color:{COLORS[old]}">{old.value}</td>
            <td style="padding:8px">&rarr;</td>
            <td style="padding:8px;color:{color};font-weight:600">{new.value}</td>
            <td style="padding:8px">{PLASTICITY[new]:.2f}</td>
        </tr>"""

    table = f"""<table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Tick</th>
            <th style="text-align:left;padding:8px">From</th>
            <th></th>
            <th style="text-align:left;padding:8px">To</th>
            <th style="text-align:left;padding:8px">Plasticity</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""

    # Critical periods
    final_stage = _get_stage(1.0, juv_t, adol_t, mat_t)
    cp_rows = ""
    stage_order = {s: i for i, s in enumerate(STAGES)}
    for name, opens, closes, desc in CRITICAL_PERIODS:
        is_closed = stage_order[final_stage] >= stage_order[closes]
        status = "CLOSED" if is_closed else "OPEN"
        color = "#a44" if is_closed else "#4a4"
        cp_rows += f"""<tr>
            <td style="padding:8px">{name}</td>
            <td style="padding:8px">{opens.value} → {closes.value}</td>
            <td style="padding:8px;color:{color};font-weight:600">{status}</td>
            <td style="padding:8px;font-size:12px;color:#888">{desc}</td>
        </tr>"""

    cp_table = f"""<h3 style="margin-top:24px">Critical Periods at MATURE</h3>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Period</th>
            <th style="text-align:left;padding:8px">Window</th>
            <th style="text-align:left;padding:8px">Status</th>
            <th style="text-align:left;padding:8px">Description</th>
        </tr></thead>
        <tbody>{cp_rows}</tbody>
    </table>"""

    # Tools
    tool_rows = ""
    for name, min_stage, desc in TOOLS:
        can_acquire = stage_order[final_stage] >= stage_order[min_stage]
        color = "#4a4" if can_acquire else "#a44"
        tool_rows += f"""<tr>
            <td style="padding:8px">{name}</td>
            <td style="padding:8px">{min_stage.value}</td>
            <td style="padding:8px;color:{color};font-weight:600">{'AVAILABLE' if can_acquire else 'LOCKED'}</td>
            <td style="padding:8px;font-size:12px;color:#888">{desc}</td>
        </tr>"""

    tool_table = f"""<h3 style="margin-top:24px">Capability Gating at MATURE</h3>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Tool</th>
            <th style="text-align:left;padding:8px">Min Stage</th>
            <th style="text-align:left;padding:8px">Status</th>
            <th style="text-align:left;padding:8px">Description</th>
        </tr></thead>
        <tbody>{tool_rows}</tbody>
    </table>"""

    return table + cp_table + tool_table


def build_app():
    with gr.Blocks(title="Operon Developmental Staging") as demo:
        gr.Markdown("# Operon Developmental Staging\nLifecycle progression, critical periods, and capability gating.")

        max_ops = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Operations")
        juv = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Juvenile Threshold (%)")
        adol = gr.Slider(minimum=10, maximum=80, value=35, step=5, label="Adolescent Threshold (%)")
        mat = gr.Slider(minimum=30, maximum=95, value=70, step=5, label="Mature Threshold (%)")
        btn = gr.Button("Run Lifecycle")
        out = gr.HTML()
        btn.click(_run_lifecycle, inputs=[max_ops, juv, adol, mat], outputs=[out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
