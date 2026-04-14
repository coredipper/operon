"""
Operon Harness Inspector -- Architecture Triple Explorer
=========================================================

Build a multi-stage organism and explore its categorical Architecture
triple (G, Know, Phi) from Paper 5: "Harness Engineering as Categorical
Architecture."  Map the triple to the four-pillar framework, compile to
external targets, and verify certificate preservation.

Run locally:  pip install gradio && python space-harness-inspector/app.py
"""

import html
import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.categorical import (
    extract_architecture,
    swarms_functor,
    deerflow_functor,
    ralph_functor,
    scion_functor,
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "(custom)": dict(
        s1_name="intake", s1_role="Normalizer", s1_mode="fixed",
        s2_name="router", s2_role="Classifier", s2_mode="fixed",
        s3_name="executor", s3_role="Engineer", s3_mode="fuzzy",
        budget=1000,
    ),
    "Research pipeline": dict(
        s1_name="gather", s1_role="Researcher", s1_mode="fixed",
        s2_name="synthesize", s2_role="Synthesizer", s2_mode="fuzzy",
        s3_name="report", s3_role="Writer", s3_mode="deep",
        budget=1500,
    ),
    "Code review": dict(
        s1_name="diff_parser", s1_role="Parser", s1_mode="fixed",
        s2_name="reviewer", s2_role="Reviewer", s2_mode="fuzzy",
        s3_name="commenter", s3_role="Commentator", s3_mode="fixed",
        budget=800,
    ),
    "Deep analysis": dict(
        s1_name="intake", s1_role="Triage", s1_mode="fixed",
        s2_name="analyze", s2_role="Analyst", s2_mode="deep",
        s3_name="recommend", s3_role="Advisor", s3_mode="deep",
        budget=2000,
    ),
}

MODES = ["fixed", "fuzzy", "deep"]
FUNCTORS = {
    "swarms": swarms_functor,
    "deerflow": deerflow_functor,
    "ralph": ralph_functor,
    "scion": scion_functor,
}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _build_organism(names, roles, modes, budget_val):
    fast = Nucleus(provider=MockProvider(responses={"classify": "ROUTE: a"}))
    deep = Nucleus(provider=MockProvider(responses={"a": "done"}))
    return skill_organism(
        stages=[
            SkillStage(name=names[i], role=roles[i],
                       instructions=f"{roles[i]} stage.", mode=modes[i])
            for i in range(3)
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=budget_val, silent=True),
    )


def _badge(text, color):
    return (f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;font-weight:600;">{text}</span>')


def _section(title, content, border_color="#6366f1"):
    return (
        f'<div style="border:2px solid {border_color};border-radius:8px;'
        f'margin-bottom:12px;overflow:hidden;">'
        f'<div style="padding:8px 14px;background:{border_color}15;'
        f'border-bottom:1px solid {border_color};">'
        f'<span style="font-weight:700;">{title}</span></div>'
        f'<div style="padding:12px 14px;">{content}</div></div>'
    )


def inspect_architecture(
    s1_name, s1_role, s1_mode,
    s2_name, s2_role, s2_mode,
    s3_name, s3_role, s3_mode,
    budget_val,
):
    names = [str(s).strip() for s in (s1_name, s2_name, s3_name)]
    roles = [str(s).strip() for s in (s1_role, s2_role, s3_role)]
    _valid_modes = {"fixed", "fuzzy", "deep"}
    modes = [m if m in _valid_modes else "fixed" for m in (s1_mode, s2_mode, s3_mode)]

    if not all(names) or not all(roles):
        return "<p>Please fill in all stage names and roles.</p>", "{}", ""

    budget = int(budget_val)
    org = _build_organism(names, roles, modes, budget)
    _esc = html.escape  # shorthand for escaping user input in HTML
    arch = extract_architecture(org)

    # --- Architecture Triple ---
    # G (graph)
    edges_html = " &rarr; ".join(_esc(s) for s in arch.stage_names)
    g_content = (
        f'<div style="font-family:monospace;font-size:1.1em;margin-bottom:8px;">'
        f'{edges_html}</div>'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<tr style="border-bottom:1px solid #e5e7eb;">'
        f'<td style="padding:4px 8px;color:#6b7280;">Stages</td>'
        f'<td style="padding:4px 8px;">{[_esc(s) for s in arch.stage_names]}</td></tr>'
        f'<tr style="border-bottom:1px solid #e5e7eb;">'
        f'<td style="padding:4px 8px;color:#6b7280;">Edges</td>'
        f'<td style="padding:4px 8px;">{[(_esc(a), _esc(b)) for a, b in arch.edges]}</td></tr>'
        f'<tr>'
        f'<td style="padding:4px 8px;color:#6b7280;">Sequential</td>'
        f'<td style="padding:4px 8px;">{arch.is_sequential}</td></tr>'
        f'</table>'
    )

    # Know (certificates)
    if arch.certificates:
        cert_rows = ""
        for cd in arch.certificates:
            holds = _badge("HOLDS", "#22c55e") if budget > 0 else _badge("FAILS", "#ef4444")
            cert_rows += (
                f'<div style="padding:6px 0;border-bottom:1px solid #f3f4f6;">'
                f'{holds} <b>{cd["theorem"]}</b><br>'
                f'<span style="color:#6b7280;font-size:0.9em;">{cd["conclusion"]}</span>'
                f'</div>')
        know_content = cert_rows
    else:
        know_content = '<span style="color:#9ca3af;">No certificates</span>'

    # Phi (interface)
    tier_map = {"fixed": "fast", "fuzzy": "fast&rarr;deep", "deep": "deep"}
    phi_rows = ""
    for stage_name, mode in arch.interface:
        mode_s = str(mode) if mode is not None else ""
        tier = tier_map.get(mode_s, _esc(mode_s))
        phi_rows += (
            f'<tr style="border-bottom:1px solid #f3f4f6;">'
            f'<td style="padding:4px 8px;font-family:monospace;">{_esc(str(stage_name))}</td>'
            f'<td style="padding:4px 8px;">{_esc(mode_s)}</td>'
            f'<td style="padding:4px 8px;">{tier}</td></tr>')
    phi_content = (
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<tr style="border-bottom:2px solid #e5e7eb;color:#6b7280;">'
        f'<th style="text-align:left;padding:4px 8px;">Stage</th>'
        f'<th style="text-align:left;padding:4px 8px;">Mode</th>'
        f'<th style="text-align:left;padding:4px 8px;">Tier</th></tr>'
        f'{phi_rows}</table>')

    arch_html = (
        _section("G &mdash; Graph (syntactic wiring)", g_content, "#3b82f6")
        + _section("Know &mdash; Knowledge (structural guarantees)", know_content, "#22c55e")
        + _section("&Phi; &mdash; Interface (mode &rarr; model mapping)", phi_content, "#a855f7")
    )

    # --- Four-Pillar Mapping ---
    pillar_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.95em;">'
        '<tr style="border-bottom:2px solid #e5e7eb;color:#6b7280;">'
        '<th style="text-align:left;padding:6px 8px;">Pillar</th>'
        '<th style="text-align:left;padding:6px 8px;">Arch Component</th>'
        '<th style="text-align:left;padding:6px 8px;">Operon Realization</th></tr>'
        '<tr style="border-bottom:1px solid #f3f4f6;">'
        '<td style="padding:6px 8px;font-weight:600;">Harness</td>'
        '<td style="padding:6px 8px;">G (graph)</td>'
        '<td style="padding:6px 8px;font-family:monospace;">SkillOrganism.stages + edges</td></tr>'
        '<tr style="border-bottom:1px solid #f3f4f6;">'
        '<td style="padding:6px 8px;font-weight:600;">Skills</td>'
        '<td style="padding:6px 8px;">G (graph)</td>'
        '<td style="padding:6px 8px;font-family:monospace;">SkillStage.role + handler</td></tr>'
        '<tr style="border-bottom:1px solid #f3f4f6;">'
        '<td style="padding:6px 8px;font-weight:600;">Protocols</td>'
        '<td style="padding:6px 8px;">Know (certs)</td>'
        '<td style="padding:6px 8px;font-family:monospace;">ATP_Store.certify &rarr; Certificate</td></tr>'
        '<tr>'
        '<td style="padding:6px 8px;font-weight:600;">Memory</td>'
        '<td style="padding:6px 8px;">&Phi; (interface)</td>'
        '<td style="padding:6px 8px;font-family:monospace;">mode &rarr; nucleus mapping</td></tr>'
        '</table>')

    # --- Functor Results ---
    functor_rows = ""
    for name, functor in FUNCTORS.items():
        result = functor.compile(org)
        p = result.preservation
        g_status = _badge("preserved", "#22c55e") if p.graph_preserved else _badge("enriched", "#f59e0b")
        c_status = _badge("preserved", "#22c55e") if p.certificate_preserved else _badge("LOST", "#ef4444")
        i_status = _badge("preserved", "#22c55e") if p.interface_preserved else _badge("remapped", "#f59e0b")
        functor_rows += (
            f'<tr style="border-bottom:1px solid #f3f4f6;">'
            f'<td style="padding:6px 8px;font-weight:600;">{name}</td>'
            f'<td style="padding:6px 8px;">{g_status}</td>'
            f'<td style="padding:6px 8px;">{c_status}</td>'
            f'<td style="padding:6px 8px;">{i_status}</td></tr>')

    functor_html = (
        '<table style="width:100%;border-collapse:collapse;">'
        '<tr style="border-bottom:2px solid #e5e7eb;color:#6b7280;">'
        '<th style="text-align:left;padding:6px 8px;">Functor</th>'
        '<th style="text-align:left;padding:6px 8px;">Graph</th>'
        '<th style="text-align:left;padding:6px 8px;">Certificates</th>'
        '<th style="text-align:left;padding:6px 8px;">Interface</th></tr>'
        f'{functor_rows}</table>')

    return arch_html, pillar_html, functor_html


def load_preset(name: str):
    p = PRESETS.get(name, PRESETS["(custom)"])
    return (p["s1_name"], p["s1_role"], p["s1_mode"],
            p["s2_name"], p["s2_role"], p["s2_mode"],
            p["s3_name"], p["s3_role"], p["s3_mode"],
            p.get("budget", 1000))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Harness Inspector") as app:
        gr.Markdown(
            "# Operon Harness Inspector\n"
            "Explore the **Architecture triple (G, Know, &Phi;)** from Paper 5: "
            "*Harness Engineering as Categorical Architecture*. "
            "Build an organism, extract its categorical structure, and see how "
            "compiler functors preserve properties.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article) | "
            "de los Riscos et al. [arXiv:2603.28906](https://arxiv.org/abs/2603.28906)")

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()), value="(custom)",
                label="Load Preset", scale=2)
            inspect_btn = gr.Button("Inspect", variant="primary", scale=1)

        gr.Markdown("### Pipeline Stages")
        with gr.Row():
            s1_name = gr.Textbox(value="intake", label="Stage 1 Name")
            s1_role = gr.Textbox(value="Normalizer", label="Stage 1 Role")
            s1_mode = gr.Dropdown(choices=MODES, value="fixed", label="Stage 1 Mode")
        with gr.Row():
            s2_name = gr.Textbox(value="router", label="Stage 2 Name")
            s2_role = gr.Textbox(value="Classifier", label="Stage 2 Role")
            s2_mode = gr.Dropdown(choices=MODES, value="fixed", label="Stage 2 Mode")
        with gr.Row():
            s3_name = gr.Textbox(value="executor", label="Stage 3 Name")
            s3_role = gr.Textbox(value="Engineer", label="Stage 3 Role")
            s3_mode = gr.Dropdown(choices=MODES, value="fuzzy", label="Stage 3 Mode")

        budget_slider = gr.Slider(
            minimum=0, maximum=2000, value=1000, step=50,
            label="ATP Budget (0 = failing certificate)")

        gr.Markdown("### Architecture Triple (G, Know, &Phi;)")
        arch_output = gr.HTML()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Four-Pillar Mapping")
                pillar_output = gr.HTML()
            with gr.Column():
                gr.Markdown("### Compiler Functors")
                functor_output = gr.HTML()

        all_inputs = [
            s1_name, s1_role, s1_mode,
            s2_name, s2_role, s2_mode,
            s3_name, s3_role, s3_mode,
            budget_slider,
        ]

        inspect_btn.click(
            fn=inspect_architecture, inputs=all_inputs,
            outputs=[arch_output, pillar_output, functor_output])
        preset_dd.change(
            fn=load_preset, inputs=[preset_dd],
            outputs=[s1_name, s1_role, s1_mode,
                     s2_name, s2_role, s2_mode,
                     s3_name, s3_role, s3_mode,
                     budget_slider])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
