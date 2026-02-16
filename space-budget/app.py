"""
Operon Budget Simulator -- Metabolic Energy Management
======================================================

Configure a multi-currency energy budget (ATP, GTP, NADH), queue tasks
with custom costs, and watch how the metabolic system manages resources:
state transitions, NADH-to-ATP conversion, conservation mode, and apoptosis.

Run locally:
    pip install gradio
    python space-budget/app.py
"""

import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import ATP_Store, EnergyType, MetabolicState


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "(custom)": {
        "atp": 100, "gtp": 0, "nadh": 0,
        "tasks": "Query LLM:10\nSummarize:10\nTranslate:10",
    },
    "Well-funded agent": {
        "atp": 100, "gtp": 20, "nadh": 30,
        "tasks": "Query LLM:10\nAnalyze sentiment:15\nGenerate summary:20\nTranslate:10\nCreate report:25",
    },
    "Constrained agent": {
        "atp": 30, "gtp": 0, "nadh": 0,
        "tasks": "Query LLM:10\nAnalyze sentiment:15\nGenerate summary:20\nTranslate:10\nCreate report:25",
    },
    "NADH reserve rescue": {
        "atp": 25, "gtp": 0, "nadh": 40,
        "tasks": "Task A:10\nTask B:10\nTask C:10\nTask D:10\nTask E:10\nTask F:10",
    },
    "Multi-currency": {
        "atp": 50, "gtp": 30, "nadh": 20,
        "tasks": "Standard query:10\nPremium tool call:15:gtp\nStandard analysis:12\nPremium synthesis:20:gtp\nFinal report:15",
    },
    "Rapid exhaustion": {
        "atp": 20, "gtp": 0, "nadh": 0,
        "tasks": "Heavy task 1:8\nHeavy task 2:8\nHeavy task 3:8\nHeavy task 4:8",
    },
    "State transition showcase": {
        "atp": 150, "gtp": 0, "nadh": 0,
        "tasks": "Task A:20\nTask B:20\nTask C:20\nTask D:20\nTask E:20\nTask F:20\nTask G:20",
    },
    "NADH-heavy rescue": {
        "atp": 10, "gtp": 0, "nadh": 80,
        "tasks": "Task 1:10\nTask 2:10\nTask 3:10\nTask 4:10\nTask 5:10\nTask 6:10",
    },
    "Critical shortage": {
        "atp": 5, "gtp": 0, "nadh": 0,
        "tasks": "Urgent task:10\nCritical task:10\nEmergency task:10",
    },
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

STATE_STYLES = {
    MetabolicState.FEASTING: ("#22c55e", "FEASTING", "Excess energy -- can do extra work"),
    MetabolicState.NORMAL: ("#3b82f6", "NORMAL", "Plenty of energy"),
    MetabolicState.CONSERVING: ("#f59e0b", "CONSERVING", "Low energy -- reducing activity"),
    MetabolicState.STARVING: ("#ef4444", "STARVING", "Critical -- survival mode only"),
    MetabolicState.DORMANT: ("#6b7280", "DORMANT", "Minimal activity"),
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _parse_tasks(tasks_text: str) -> list[tuple[str, int, EnergyType]]:
    """Parse task list from text. Format: 'Task Name:cost[:gtp|nadh]' per line."""
    tasks = []
    for line in tasks_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        name = parts[0].strip()
        cost = int(parts[1].strip()) if len(parts) > 1 else 10
        energy_str = parts[2].strip().lower() if len(parts) > 2 else "atp"
        energy = {"gtp": EnergyType.GTP, "nadh": EnergyType.NADH}.get(energy_str, EnergyType.ATP)
        tasks.append((name, cost, energy))
    return tasks


def _energy_bar(current: int, maximum: int, label: str, color: str) -> str:
    """Render an energy bar as HTML."""
    if maximum == 0:
        return ""
    pct = max(0, min(100, int(current / maximum * 100)))
    return (
        f'<div style="margin:4px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85em;">'
        f'<span>{label}</span><span>{current}/{maximum}</span></div>'
        f'<div style="background:#e5e7eb;border-radius:4px;height:16px;">'
        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;'
        f'transition:width 0.3s;"></div></div></div>'
    )


def _state_badge(state: MetabolicState) -> str:
    color, label, _ = STATE_STYLES.get(state, ("#6b7280", "UNKNOWN", ""))
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def run_budget(atp_budget, gtp_budget, nadh_reserve, tasks_text) -> tuple[str, str, str]:
    """Run the budget simulation.

    Returns (summary_html, timeline_md, report_md).
    """
    atp_budget = int(atp_budget)
    gtp_budget = int(gtp_budget)
    nadh_reserve = int(nadh_reserve)

    if not tasks_text.strip():
        return "Add at least one task.", "", ""

    tasks = _parse_tasks(tasks_text)
    if not tasks:
        return "Could not parse tasks. Use format: Task Name:cost", "", ""

    store = ATP_Store(
        budget=atp_budget,
        gtp_budget=gtp_budget,
        nadh_reserve=nadh_reserve,
        silent=True,
    )

    # --- Run tasks and record timeline ---
    timeline_rows = []
    state_transitions = []
    prev_state = store.get_state()

    timeline_rows.append(
        f"| -- | *Initial* | -- | {store.atp} | {store.gtp} | {store.nadh} "
        f"| {_state_badge(prev_state)} | -- |"
    )

    for i, (name, cost, energy_type) in enumerate(tasks, 1):
        success = store.consume(cost, name, energy_type)

        new_state = store.get_state()
        if new_state != prev_state:
            state_transitions.append((i, prev_state, new_state))
            prev_state = new_state

        status = "OK" if success else "FAILED"
        status_style = "color:#16a34a;font-weight:600;" if success else "color:#dc2626;font-weight:600;"

        timeline_rows.append(
            f"| {i} | {name} | {cost} {energy_type.value.upper()} "
            f"| {store.atp} | {store.gtp} | {store.nadh} "
            f"| {_state_badge(new_state)} "
            f'| <span style="{status_style}">{status}</span> |'
        )

    # --- Summary banner ---
    final_state = store.get_state()
    s_color, _, s_desc = STATE_STYLES.get(final_state, ("#6b7280", "UNKNOWN", ""))

    bars = ""
    bars += _energy_bar(store.atp, store.max_atp, "ATP (primary)", "#3b82f6")
    if store.max_gtp > 0:
        bars += _energy_bar(store.gtp, store.max_gtp, "GTP (premium)", "#a855f7")
    if store.max_nadh > 0:
        bars += _energy_bar(store.nadh, store.max_nadh, "NADH (reserve)", "#f59e0b")

    report = store.get_report()
    completed = sum(1 for r in timeline_rows[1:] if "OK</span>" in r)

    summary_html = (
        f'<div style="padding:16px;border-radius:8px;border:2px solid {s_color};background:#f9fafb;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        f'<span style="font-size:1.2em;font-weight:700;">Final State:</span>'
        f'{_state_badge(final_state)}'
        f'<span style="color:#6b7280;font-size:0.9em;">-- {s_desc}</span>'
        f'</div>'
        f'{bars}'
        f'<div style="margin-top:12px;display:flex;gap:20px;font-size:0.9em;">'
        f'<span>Tasks completed: <b>{completed}/{len(tasks)}</b></span>'
        f'<span>Health: <b>{report.health_score:.0%}</b></span>'
        f'<span>Utilization: <b>{report.utilization:.0%}</b></span>'
        f'</div>'
        f'</div>'
    )

    # --- Timeline table ---
    timeline_md = "| # | Task | Cost | ATP | GTP | NADH | State | Status |\n"
    timeline_md += "|---|------|------|-----|-----|------|-------|--------|\n"
    timeline_md += "\n".join(timeline_rows)

    if state_transitions:
        timeline_md += "\n\n**State transitions:**\n\n"
        for step, old, new in state_transitions:
            _, old_label, _ = STATE_STYLES.get(old, ("#6b7280", "?", ""))
            _, new_label, _ = STATE_STYLES.get(new, ("#6b7280", "?", ""))
            timeline_md += f"- Step {step}: {old_label} -> {new_label}\n"

    # --- Report ---
    stats = store.get_statistics()
    report_md = "### Metabolic Report\n\n"
    report_md += f"**Total consumed:** {stats['total_consumed']} units\n\n"
    report_md += f"**Operations:** {stats['operations_count']} "
    report_md += f"({stats['failed_operations']} failed)\n\n"
    report_md += f"**Success rate:** {stats['success_rate']:.0%}\n\n"

    report_md += "### How It Works\n\n"
    report_md += "| State | Threshold | Behavior |\n"
    report_md += "|-------|-----------|----------|\n"
    report_md += "| FEASTING | >90% capacity | Can do extra work |\n"
    report_md += "| NORMAL | 30-90% | Full operation |\n"
    report_md += "| CONSERVING | 10-30% | Reduced activity |\n"
    report_md += "| STARVING | <10% | Only critical ops |\n"
    report_md += "\nWhen ATP runs low, NADH reserve auto-converts to ATP.\n"

    return summary_html, timeline_md, report_md


def load_preset(name: str):
    preset = PRESETS.get(name)
    if not preset:
        return 100, 0, 0, ""
    return preset["atp"], preset["gtp"], preset["nadh"], preset["tasks"]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Budget Simulator") as app:
        gr.Markdown(
            "# Operon Budget Simulator\n"
            "Multi-currency metabolic energy management with **ATP** (primary), "
            "**GTP** (premium), and **NADH** (reserve). Watch state transitions, "
            "NADH-to-ATP conversion, and graceful degradation under resource pressure.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="(custom)",
                label="Load Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Simulation", variant="primary", scale=1)

        with gr.Row():
            atp_slider = gr.Slider(
                minimum=0, maximum=200, value=100, step=5,
                label="ATP Budget (primary)",
            )
            gtp_slider = gr.Slider(
                minimum=0, maximum=100, value=0, step=5,
                label="GTP Budget (premium)",
            )
            nadh_slider = gr.Slider(
                minimum=0, maximum=100, value=0, step=5,
                label="NADH Reserve (convertible)",
            )

        tasks_input = gr.Textbox(
            label="Task Queue (one per line: Task Name:cost[:gtp|nadh])",
            placeholder="Query LLM:10\nAnalyze sentiment:15\nGenerate summary:20",
            lines=6,
        )

        summary_html = gr.HTML(label="Summary")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Execution Timeline")
                timeline_md = gr.Markdown()
            with gr.Column(scale=1):
                report_md = gr.Markdown()

        # Wire events
        run_btn.click(
            fn=run_budget,
            inputs=[atp_slider, gtp_slider, nadh_slider, tasks_input],
            outputs=[summary_html, timeline_md, report_md],
        )

        preset_dropdown.change(
            fn=load_preset,
            inputs=[preset_dropdown],
            outputs=[atp_slider, gtp_slider, nadh_slider, tasks_input],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
