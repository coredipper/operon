"""
Operon Convergence Compiler -- Compile & Verify
================================================

Build a multi-stage organism, compile it to an external framework
(Swarms, DeerFlow, Ralph, Scion), and verify the structural certificates
that the compiler preserves through the translation.

Run locally:  pip install gradio && python space-compiler/app.py
"""

import json
import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence import (
    organism_to_deerflow, organism_to_ralph,
    organism_to_scion, organism_to_swarms,
)
from operon_ai.core.certificate import verify_compiled

# ---------------------------------------------------------------------------
# Presets & constants
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "(custom)": dict(
        s1_name="intake", s1_role="Normalizer",
        s2_name="router", s2_role="Classifier",
        s3_name="executor", s3_role="Analyst",
        framework="Swarms", budget=1000, zero_budget=False,
    ),
    "Research pipeline": dict(
        s1_name="gather", s1_role="Researcher",
        s2_name="synthesize", s2_role="Synthesizer",
        s3_name="report", s3_role="Writer",
        framework="DeerFlow", budget=1500, zero_budget=False,
    ),
    "Code review": dict(
        s1_name="diff_parser", s1_role="Parser",
        s2_name="reviewer", s2_role="Reviewer",
        s3_name="commenter", s3_role="Commentator",
        framework="Ralph", budget=800, zero_budget=False,
    ),
    "Failing certificate": dict(
        s1_name="intake", s1_role="Normalizer",
        s2_name="router", s2_role="Classifier",
        s3_name="executor", s3_role="Analyst",
        framework="Swarms", budget=0, zero_budget=True,
    ),
}

FRAMEWORKS = ["Swarms", "DeerFlow", "Ralph", "Scion"]
_COMPILER_MAP = {
    "Swarms": organism_to_swarms, "DeerFlow": organism_to_deerflow,
    "Ralph": organism_to_ralph, "Scion": organism_to_scion,
}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _build_organism(names, roles, budget_val):
    """Build a 3-stage SkillOrganism with mock nuclei."""
    fast = Nucleus(provider=MockProvider(responses={"classify": "ROUTE: analysis"}))
    deep = Nucleus(provider=MockProvider(responses={"analysis": "Analysis complete."}))
    return skill_organism(
        stages=[
            SkillStage(name=names[0], role=roles[0],
                       handler=lambda task, state, outputs, stage: {"request": task}),
            SkillStage(name=names[1], role=roles[1],
                       instructions="Classify the incoming request.", mode="fixed"),
            SkillStage(name=names[2], role=roles[2],
                       instructions="Analyze the request in depth.", mode="fuzzy"),
        ],
        fast_nucleus=fast, deep_nucleus=deep,
        budget=ATP_Store(budget=budget_val, silent=True),
    )


def _cert_badge(holds: bool) -> str:
    color, label = ("#22c55e", "HOLDS") if holds else ("#ef4444", "FAILS")
    return (f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;font-weight:600;">{label}</span>')


def _build_cert_html(verifications):
    """Render certificate verification results as styled HTML."""
    if not verifications:
        return (
            '<div style="padding:16px;border-radius:8px;border:2px solid #f59e0b;'
            'background:#fffbeb;">'
            '<p style="margin:0;font-weight:600;color:#92400e;">'
            'No certificates found in compiled output.</p>'
            '<p style="margin:4px 0 0;color:#78716c;font-size:0.9em;">'
            'Budget may be 0 -- the organism produces no metabolic certificates '
            'when there is no energy to gate.</p></div>')

    all_hold = all(v.holds for v in verifications)
    bc = "#22c55e" if all_hold else "#ef4444"
    bg = "#f0fdf4" if all_hold else "#fef2f2"
    summary = "All certificates verified" if all_hold else "One or more certificates failed"
    n = len(verifications)

    rows = ""
    for v in verifications:
        ev = ", ".join(f"<b>{k}</b>={val}" for k, val in dict(v.evidence).items())
        rows += (
            f'<div style="padding:8px 12px;border-bottom:1px solid #e5e7eb;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'{_cert_badge(v.holds)}'
            f'<span style="font-weight:600;">{v.certificate.theorem}</span></div>'
            f'<div style="margin-top:4px;font-size:0.85em;color:#6b7280;">'
            f'{v.certificate.conclusion}</div>'
            f'<div style="margin-top:4px;font-size:0.85em;color:#374151;">'
            f'Evidence: {ev}</div></div>')

    return (
        f'<div style="padding:0;border-radius:8px;border:2px solid {bc};'
        f'background:{bg};overflow:hidden;">'
        f'<div style="padding:12px 16px;background:{bc}10;border-bottom:1px solid {bc};">'
        f'<span style="font-weight:700;font-size:1.05em;">{summary}</span>'
        f'<span style="margin-left:8px;color:#6b7280;font-size:0.9em;">'
        f'({n} certificate{"s" if n != 1 else ""})</span></div>'
        f'{rows}</div>')


def compile_and_verify(
    s1_name, s1_role, s2_name, s2_role, s3_name, s3_role,
    framework, budget_val, zero_budget,
):
    """Compile the organism and verify certificates."""
    names = [s.strip() for s in (s1_name, s2_name, s3_name)]
    roles = [s.strip() for s in (s1_role, s2_role, s3_role)]
    if not all(names) or not all(roles):
        return "{}", "<p>Please fill in all stage names and roles.</p>"

    effective_budget = 0 if zero_budget else int(budget_val)
    try:
        org = _build_organism(names, roles, effective_budget)
        compiled = _COMPILER_MAP[framework](org)
        compiled_json = json.dumps(compiled, indent=2, default=str)
        verifications = verify_compiled(compiled)
        return compiled_json, _build_cert_html(verifications)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2), (
            f'<div style="padding:16px;border-radius:8px;border:2px solid #ef4444;'
            f'background:#fef2f2;">'
            f'<p style="margin:0;font-weight:600;color:#991b1b;">Error</p>'
            f'<p style="margin:4px 0 0;color:#7f1d1d;">{e}</p></div>')


def load_preset(name: str):
    p = PRESETS.get(name)
    if not p:
        return "intake", "Normalizer", "router", "Classifier", "executor", "Analyst", "Swarms", 1000, False
    return (p["s1_name"], p["s1_role"], p["s2_name"], p["s2_role"],
            p["s3_name"], p["s3_role"], p["framework"], p["budget"], p["zero_budget"])

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Convergence Compiler") as app:
        gr.Markdown(
            "# Operon Convergence Compiler\n"
            "Build a 3-stage organism, compile it to an external agent framework, "
            "and verify that **structural certificates** (budget gating, priority "
            "guarantees) survive the translation.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)")

        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()), value="(custom)",
                label="Load Preset", scale=2)
            compile_btn = gr.Button("Compile", variant="primary", scale=1)

        gr.Markdown("### Pipeline Stages")
        with gr.Row():
            s1_name = gr.Textbox(value="intake", label="Stage 1 Name", scale=1)
            s1_role = gr.Textbox(value="Normalizer", label="Stage 1 Role", scale=1)
        with gr.Row():
            s2_name = gr.Textbox(value="router", label="Stage 2 Name", scale=1)
            s2_role = gr.Textbox(value="Classifier", label="Stage 2 Role", scale=1)
        with gr.Row():
            s3_name = gr.Textbox(value="executor", label="Stage 3 Name", scale=1)
            s3_role = gr.Textbox(value="Analyst", label="Stage 3 Role", scale=1)

        with gr.Row():
            framework_dd = gr.Dropdown(
                choices=FRAMEWORKS, value="Swarms",
                label="Target Framework", scale=1)
            budget_slider = gr.Slider(
                minimum=0, maximum=2000, value=1000, step=50,
                label="ATP Budget", scale=1)
            zero_budget_cb = gr.Checkbox(
                value=False, label="Set budget to 0 (show failing certificate)")

        gr.Markdown("### Output")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Compiled JSON")
                compiled_output = gr.Code(language="json", label="Compiled output")
            with gr.Column(scale=1):
                gr.Markdown("#### Certificate Verification")
                cert_output = gr.HTML()

        all_inputs = [s1_name, s1_role, s2_name, s2_role, s3_name, s3_role,
                      framework_dd, budget_slider, zero_budget_cb]

        compile_btn.click(fn=compile_and_verify, inputs=all_inputs,
                          outputs=[compiled_output, cert_output])
        preset_dropdown.change(fn=load_preset, inputs=[preset_dropdown],
                               outputs=all_inputs)
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
