"""
Operon LangGraph Visualizer -- Per-Stage Graph Topology
========================================================

Build a multi-stage organism, compile it to a per-stage LangGraph,
visualize the graph topology, and run it to see which stages execute.

Run locally:  pip install gradio && python space-langgraph-visualizer/app.py
"""

import html as html_mod
import sys
from pathlib import Path

import gradio as gr

try:
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
except IndexError:
    pass  # Running on HF — operon-ai installed via requirements.txt

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.langgraph_compiler import (
    organism_to_langgraph,
    run_organism_langgraph,
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "3-stage pipeline": {
        "stages": [
            ("intake", "Normalizer", "fixed"),
            ("router", "Classifier", "fixed"),
            ("executor", "Engineer", "fuzzy"),
        ],
        "task": "Fix the login crash after session timeout",
    },
    "4-stage incident": {
        "stages": [
            ("triage", "Triager", "fixed"),
            ("classify", "Classifier", "fixed"),
            ("investigate", "Investigator", "fixed"),
            ("fix", "Engineer", "fuzzy"),
        ],
        "task": "Production auth failures after JWT migration",
    },
    "2-stage simple": {
        "stages": [
            ("analyze", "Analyst", "fixed"),
            ("respond", "Responder", "fuzzy"),
        ],
        "task": "Summarize the quarterly report",
    },
    "5-stage deep": {
        "stages": [
            ("intake", "Normalizer", "fixed"),
            ("triage", "Triager", "fixed"),
            ("plan", "Planner", "fuzzy"),
            ("execute", "Engineer", "deep"),
            ("review", "Reviewer", "fixed"),
        ],
        "task": "Refactor the authentication middleware for SOC2 compliance",
    },
}

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _node_html(name, mode, index, total, executed=False, is_start=False, is_end=False):
    """Render a single graph node."""
    if is_start:
        return (
            '<div style="display:flex;flex-direction:column;align-items:center;">'
            '<div style="width:40px;height:40px;border-radius:50%;background:#10b981;'
            'display:flex;align-items:center;justify-content:center;color:white;'
            'font-weight:700;font-size:0.8em;">START</div></div>')
    if is_end:
        return (
            '<div style="display:flex;flex-direction:column;align-items:center;">'
            '<div style="width:40px;height:40px;border-radius:50%;background:#ef4444;'
            'display:flex;align-items:center;justify-content:center;color:white;'
            'font-weight:700;font-size:0.8em;">END</div></div>')

    mode_colors = {"fixed": "#3b82f6", "fuzzy": "#f59e0b", "deep": "#8b5cf6"}
    border_color = mode_colors.get(mode, "#6b7280")
    bg = f"{border_color}15"
    check = ' <span style="color:#22c55e;">&#10003;</span>' if executed else ""

    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;">'
        f'<div style="border:3px solid {border_color};border-radius:10px;'
        f'padding:12px 20px;background:{bg};min-width:120px;text-align:center;'
        f'{"box-shadow:0 0 12px " + border_color + "40;" if executed else ""}">'
        f'<div style="font-weight:700;font-size:1.05em;">{html_mod.escape(name)}{check}</div>'
        f'<div style="font-size:0.85em;color:#6b7280;">mode: {html_mod.escape(mode)}</div>'
        f'</div></div>')


def _arrow_html(label="continue"):
    color = "#22c55e" if label == "continue" else "#ef4444"
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'padding:0 8px;">'
        f'<div style="font-size:1.5em;color:{color};">&#8594;</div>'
        f'<div style="font-size:0.7em;color:#9ca3af;">{label}</div></div>')


def _halt_arrow_html():
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'padding:0 4px;opacity:0.5;">'
        '<div style="font-size:1.2em;color:#ef4444;">&#8600;</div>'
        '<div style="font-size:0.65em;color:#ef4444;">halt</div></div>')


def build_graph_html(stages_info, executed_stages=None):
    """Build an HTML visualization of the per-stage graph."""
    executed = set(executed_stages or [])
    n = len(stages_info)

    # Main flow: START → stages → END
    nodes = []
    nodes.append(_node_html("", "", 0, n, is_start=True))
    nodes.append(_arrow_html(""))

    for i, (name, _, mode) in enumerate(stages_info):
        nodes.append(_node_html(name, mode, i, n, executed=name in executed))
        if i < n - 1:
            nodes.append(_arrow_html("continue"))
        else:
            nodes.append(_arrow_html(""))

    nodes.append(_node_html("", "", 0, n, is_end=True))

    main_flow = (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'flex-wrap:wrap;gap:4px;padding:20px 0;">'
        + "".join(nodes) + '</div>')

    # Halt edges legend
    halt_legend = (
        '<div style="text-align:center;padding:8px;color:#9ca3af;font-size:0.85em;">'
        'Each stage has a conditional edge: '
        '<span style="color:#22c55e;font-weight:600;">continue</span> &rarr; next stage, '
        '<span style="color:#ef4444;font-weight:600;">halt/blocked</span> &rarr; END'
        '</div>')

    return main_flow + halt_legend


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _parse_stages(stages_text):
    """Parse stages from text format: name, role, mode (one per line)."""
    stages = []
    for line in stages_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            stages.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 2:
            stages.append((parts[0], parts[1], "fixed"))
        elif len(parts) == 1:
            stages.append((parts[0], parts[0].title(), "fixed"))
    return stages


def visualize_and_run(stages_text, task, do_run):
    if not stages_text.strip():
        return "<p>Enter at least one stage.</p>", ""

    stages_info = _parse_stages(stages_text)
    if not stages_info:
        return "<p>Could not parse stages. Use format: name, role, mode</p>", ""

    # Build organism with deterministic handlers (avoids MockProvider
    # substring-matching collisions across stages)
    def _make_handler(stage_name, stage_role):
        def handler(task, state, outputs, stage):
            return f"[{stage_role}] Processed: task complete."
        return handler

    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))

    org = skill_organism(
        stages=[
            SkillStage(name=name, role=role,
                       handler=_make_handler(name, role),
                       mode=mode)
            for name, role, mode in stages_info
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=2000, silent=True),
    )

    # Compile to LangGraph
    graph = organism_to_langgraph(org)
    all_nodes = list(graph.nodes.keys())
    stage_nodes = [n for n in all_nodes if not n.startswith("__")]

    # Graph stats
    stats_html = (
        f'<div style="padding:12px;background:#f8fafc;border-radius:8px;'
        f'margin-bottom:12px;">'
        f'<span style="font-weight:600;">Graph Stats:</span> '
        f'{len(stage_nodes)} stage nodes, '
        f'{len(stage_nodes)} conditional edges (continue/halt), '
        f'1 START edge, 1 terminal edge'
        f'</div>')

    # Visualize
    executed_stages = []
    run_html = ""

    if do_run and task.strip():
        result = run_organism_langgraph(org, task=task.strip())
        executed_stages = result.metadata.get("stages_completed", [])

        # Build run results
        rows = ""
        for sr in result.stage_outputs.items():
            name, output = sr
            preview = str(output)[:60]
            rows += (
                f'<tr style="border-bottom:1px solid #f3f4f6;">'
                f'<td style="padding:6px 8px;font-weight:600;">{html_mod.escape(name)}</td>'
                f'<td style="padding:6px 8px;font-family:monospace;'
                f'font-size:0.9em;">{html_mod.escape(preview)}</td></tr>')

        cert_rows = ""
        for cv in result.certificates_verified:
            status = "HOLDS" if cv["holds"] else "FAILS"
            color = "#22c55e" if cv["holds"] else "#ef4444"
            cert_rows += (
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:0.85em;margin-right:6px;">'
                f'{cv["theorem"]}: {status}</span>')

        run_html = (
            f'<div style="border:2px solid #3b82f6;border-radius:8px;'
            f'margin-top:12px;overflow:hidden;">'
            f'<div style="padding:8px 14px;background:#3b82f610;'
            f'border-bottom:1px solid #3b82f6;">'
            f'<span style="font-weight:700;">Execution Results</span> '
            f'<span style="color:#6b7280;font-size:0.9em;">'
            f'({result.timing_ms:.1f} ms)</span></div>'
            f'<div style="padding:12px 14px;">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<tr style="border-bottom:2px solid #e5e7eb;color:#6b7280;">'
            f'<th style="text-align:left;padding:6px 8px;">Stage</th>'
            f'<th style="text-align:left;padding:6px 8px;">Output</th></tr>'
            f'{rows}</table>'
            f'<div style="margin-top:10px;">{cert_rows or "No certificates"}</div>'
            f'<div style="margin-top:8px;color:#6b7280;font-size:0.9em;">'
            f'Halted: {result.metadata.get("halted", False)}</div>'
            f'</div></div>')

    graph_html = stats_html + build_graph_html(stages_info, executed_stages)
    return graph_html, run_html


def load_preset(name):
    p = PRESETS.get(name)
    if not p:
        return "", ""
    lines = [f"{n}, {r}, {m}" for n, r, m in p["stages"]]
    return "\n".join(lines), p["task"]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon LangGraph Visualizer") as app:
        gr.Markdown(
            "# Operon LangGraph Visualizer\n"
            "Compile an organism to a **per-stage LangGraph** and visualize the "
            "graph topology. Each stage becomes a LangGraph node; conditional "
            "edges route based on continue/halt decisions.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)")

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="3-stage pipeline",
                label="Load Preset", scale=2)
            run_btn = gr.Button("Visualize & Run", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                stages_input = gr.Textbox(
                    value="intake, Normalizer, fixed\nrouter, Classifier, fixed\nexecutor, Engineer, fuzzy",
                    label="Stages (name, role, mode -- one per line)",
                    lines=6)
            with gr.Column(scale=1):
                task_input = gr.Textbox(
                    value="Fix the login crash after session timeout",
                    label="Task (for execution)", lines=2)
                do_run = gr.Checkbox(value=True, label="Execute after compiling")

        gr.Markdown("### Graph Topology")
        graph_output = gr.HTML()

        gr.Markdown("### Execution Results")
        run_output = gr.HTML()

        run_btn.click(
            fn=visualize_and_run,
            inputs=[stages_input, task_input, do_run],
            outputs=[graph_output, run_output])
        preset_dd.change(
            fn=load_preset,
            inputs=[preset_dd],
            outputs=[stages_input, task_input])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
