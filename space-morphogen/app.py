"""
Operon Morphogen Gradients -- Interactive Gradio Demo
=====================================================

Two-tab demo: manually set gradient values to see strategy hints and
phenotype adaptation, or simulate multi-step orchestration and watch
gradients evolve.

Run locally:
    pip install gradio
    python space-morphogen/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import sys
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    MorphogenType,
    MorphogenGradient,
    GradientOrchestrator,
)

# ── Morphogen type ordering ───────────────────────────────────────────────

MORPHOGEN_ORDER = [
    MorphogenType.COMPLEXITY,
    MorphogenType.CONFIDENCE,
    MorphogenType.BUDGET,
    MorphogenType.ERROR_RATE,
    MorphogenType.URGENCY,
    MorphogenType.RISK,
]

MORPHOGEN_COLORS = {
    MorphogenType.COMPLEXITY: "#8b5cf6",
    MorphogenType.CONFIDENCE: "#22c55e",
    MorphogenType.BUDGET: "#3b82f6",
    MorphogenType.ERROR_RATE: "#ef4444",
    MorphogenType.URGENCY: "#f97316",
    MorphogenType.RISK: "#eab308",
}

# ── Tab 1: Manual Gradient Presets ─────────────────────────────────────────

MANUAL_PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Set your own gradient values.",
        "values": {m: 0.5 for m in MORPHOGEN_ORDER},
    },
    "Easy task, high confidence": {
        "description": "Low complexity, high budget, high confidence — smooth sailing.",
        "values": {
            MorphogenType.COMPLEXITY: 0.2,
            MorphogenType.CONFIDENCE: 0.9,
            MorphogenType.BUDGET: 0.8,
            MorphogenType.ERROR_RATE: 0.05,
            MorphogenType.URGENCY: 0.3,
            MorphogenType.RISK: 0.1,
        },
    },
    "Crisis mode": {
        "description": "Everything bad — high complexity, errors, urgency, risk, low budget/confidence.",
        "values": {
            MorphogenType.COMPLEXITY: 0.95,
            MorphogenType.CONFIDENCE: 0.1,
            MorphogenType.BUDGET: 0.05,
            MorphogenType.ERROR_RATE: 0.85,
            MorphogenType.URGENCY: 0.95,
            MorphogenType.RISK: 0.9,
        },
    },
    "Exploration phase": {
        "description": "Balanced values — moderate complexity, decent budget, exploring.",
        "values": {
            MorphogenType.COMPLEXITY: 0.5,
            MorphogenType.CONFIDENCE: 0.5,
            MorphogenType.BUDGET: 0.6,
            MorphogenType.ERROR_RATE: 0.2,
            MorphogenType.URGENCY: 0.4,
            MorphogenType.RISK: 0.3,
        },
    },
    "Budget crunch": {
        "description": "High complexity but near-zero budget — forces capability reduction.",
        "values": {
            MorphogenType.COMPLEXITY: 0.8,
            MorphogenType.CONFIDENCE: 0.4,
            MorphogenType.BUDGET: 0.05,
            MorphogenType.ERROR_RATE: 0.3,
            MorphogenType.URGENCY: 0.7,
            MorphogenType.RISK: 0.5,
        },
    },
}


def _load_manual_preset(name: str) -> tuple[float, float, float, float, float, float]:
    p = MANUAL_PRESETS.get(name, MANUAL_PRESETS["(custom)"])
    v = p["values"]
    return tuple(v[m] for m in MORPHOGEN_ORDER)


# ── Tab 2: Orchestrator Simulation Presets ─────────────────────────────────

ORCH_PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Enter steps as 'success:tokens' or 'fail:tokens' per line.",
        "steps": "",
        "budget": 2000,
    },
    "Smooth sailing": {
        "description": "8 consecutive successes — confidence rises, error rate drops.",
        "steps": "success:200\nsuccess:180\nsuccess:190\nsuccess:210\nsuccess:170\nsuccess:200\nsuccess:195\nsuccess:185",
        "budget": 2000,
    },
    "Cascading failures": {
        "description": "3 successes then 5 failures — watch confidence collapse and error rate spike.",
        "steps": "success:200\nsuccess:180\nsuccess:190\nfail:250\nfail:200\nfail:300\nfail:150\nfail:200",
        "budget": 2000,
    },
    "Recovery arc": {
        "description": "Alternating fail-success — gradients oscillate as system recovers.",
        "steps": "fail:200\nsuccess:180\nfail:250\nsuccess:150\nfail:300\nsuccess:200\nsuccess:170\nsuccess:160",
        "budget": 2000,
    },
}


def _load_orch_preset(name: str) -> tuple[str, int]:
    p = ORCH_PRESETS.get(name, ORCH_PRESETS["(custom)"])
    return p["steps"], p["budget"]


# ── Gradient visualization helper ─────────────────────────────────────────


def _render_gradient_bars(gradient: MorphogenGradient) -> str:
    """Render horizontal bars for all 6 morphogen values."""
    rows = []
    for m in MORPHOGEN_ORDER:
        val = gradient.get(m)
        color = MORPHOGEN_COLORS[m]
        pct = max(0, min(100, val * 100))
        level = gradient.get_level(m)
        rows.append(
            f'<div style="margin:4px 0">'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="width:100px;font-size:0.85em;font-weight:600">{m.value}</span>'
            f'<div style="flex:1;background:#e5e7eb;border-radius:4px;height:20px;position:relative">'
            f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;'
            f'transition:width 0.3s"></div></div>'
            f'<span style="width:60px;text-align:right;font-size:0.85em;color:#666">'
            f'{val:.2f}</span>'
            f'<span style="width:60px;font-size:0.75em;color:{color}">{level}</span>'
            f'</div></div>'
        )
    return '<div style="padding:8px">' + "".join(rows) + "</div>"


# ── Tab 1: Manual gradient ────────────────────────────────────────────────


def run_manual_gradient(
    preset_name: str,
    complexity: float,
    confidence: float,
    budget: float,
    error_rate: float,
    urgency: float,
    risk: float,
) -> tuple[str, str, str, str]:
    """Set gradient values and return analysis.

    Returns (gradient_html, hints_md, context_md, phenotype_md).
    """
    gradient = MorphogenGradient()
    values = [complexity, confidence, budget, error_rate, urgency, risk]
    for m, v in zip(MORPHOGEN_ORDER, values):
        gradient.set(m, v)

    orchestrator = GradientOrchestrator(gradient=gradient, silent=True)

    # Gradient bars
    gradient_html = _render_gradient_bars(gradient)

    # Strategy hints
    hints = gradient.get_strategy_hints()
    if hints:
        hints_md = "### Strategy Hints\n\n" + "\n".join(f"- {h}" for h in hints)
    else:
        hints_md = "### Strategy Hints\n\n*No specific hints at these levels.*"

    # Context injection
    ctx = gradient.get_context_injection()
    context_md = f"### Context Injection\n\n```\n{ctx}\n```" if ctx else "### Context Injection\n\n*Empty context.*"

    # Phenotype + coordination signals
    phenotype = orchestrator.get_phenotype_params()
    recruit = orchestrator.should_recruit_help()
    reduce = orchestrator.should_reduce_capabilities()

    pheno_lines = ["### Phenotype Parameters\n", "| Parameter | Value |", "| :--- | :--- |"]
    for k, v in phenotype.items():
        pheno_lines.append(f"| {k} | {v} |")

    pheno_lines.append("\n### Coordination Signals\n")
    pheno_lines.append(f"| Signal | Value |")
    pheno_lines.append(f"| :--- | :--- |")

    recruit_color = "#ef4444" if recruit else "#22c55e"
    recruit_label = "YES — requesting help" if recruit else "No"
    pheno_lines.append(
        f'| Should recruit help | <span style="color:{recruit_color}">{recruit_label}</span> |'
    )

    reduce_color = "#f97316" if reduce else "#22c55e"
    reduce_label = "YES — reducing capabilities" if reduce else "No"
    pheno_lines.append(
        f'| Should reduce capabilities | <span style="color:{reduce_color}">{reduce_label}</span> |'
    )

    phenotype_md = "\n".join(pheno_lines)

    return gradient_html, hints_md, context_md, phenotype_md


# ── Tab 2: Orchestrator simulation ────────────────────────────────────────


def run_orchestrator(
    preset_name: str,
    steps_text: str,
    total_budget: int,
) -> tuple[str, str, str]:
    """Run step-by-step orchestrator simulation.

    Returns (final_gradient_html, timeline_md, final_phenotype_md).
    """
    # Parse steps
    steps = []
    for line in steps_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) != 2:
            continue
        success = parts[0].strip().lower() == "success"
        try:
            tokens = int(parts[1].strip())
        except ValueError:
            tokens = 100
        steps.append((success, tokens))

    if not steps:
        return "<p>Enter steps as 'success:200' or 'fail:150', one per line.</p>", "", ""

    orchestrator = GradientOrchestrator(silent=True)
    timeline_rows = [
        "| Step | Result | Tokens | Complexity | Confidence | Budget | Error Rate | Urgency | Risk |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for i, (success, tokens) in enumerate(steps, 1):
        orchestrator.report_step_result(
            success=success,
            tokens_used=tokens,
            total_budget=int(total_budget),
        )

        g = orchestrator.gradient
        result_icon = "✓" if success else "✗"
        result_color = "#22c55e" if success else "#ef4444"

        timeline_rows.append(
            f'| {i} | <span style="color:{result_color}">{result_icon}</span> '
            f"| {tokens} "
            f"| {g.get(MorphogenType.COMPLEXITY):.2f} "
            f"| {g.get(MorphogenType.CONFIDENCE):.2f} "
            f"| {g.get(MorphogenType.BUDGET):.2f} "
            f"| {g.get(MorphogenType.ERROR_RATE):.2f} "
            f"| {g.get(MorphogenType.URGENCY):.2f} "
            f"| {g.get(MorphogenType.RISK):.2f} |"
        )

    timeline_md = "\n".join(timeline_rows)

    # Final gradient
    final_gradient_html = _render_gradient_bars(orchestrator.gradient)

    # Final phenotype
    phenotype = orchestrator.get_phenotype_params()
    recruit = orchestrator.should_recruit_help()
    reduce = orchestrator.should_reduce_capabilities()

    pheno_lines = ["### Final Phenotype\n", "| Parameter | Value |", "| :--- | :--- |"]
    for k, v in phenotype.items():
        pheno_lines.append(f"| {k} | {v} |")

    pheno_lines.append(f"\n**Recruit help**: {'YES' if recruit else 'No'} | "
                       f"**Reduce capabilities**: {'YES' if reduce else 'No'}")

    final_phenotype_md = "\n".join(pheno_lines)

    return final_gradient_html, timeline_md, final_phenotype_md


# ── Gradio UI ──────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Morphogen Gradients") as app:
        gr.Markdown(
            "# 🧪 Morphogen Gradients\n"
            "Explore **gradient-based coordination** where agents adapt "
            "behavior based on local chemical signals."
        )

        with gr.Tabs():
            # ── Tab 1: Manual Gradient ────────────────────────────────
            with gr.TabItem("Manual Gradient"):
                with gr.Row():
                    manual_preset_dd = gr.Dropdown(
                        choices=list(MANUAL_PRESETS.keys()),
                        value="Easy task, high confidence",
                        label="Preset",
                        scale=2,
                    )
                    manual_btn = gr.Button("Analyze Gradient", variant="primary", scale=1)

                with gr.Row():
                    complexity_sl = gr.Slider(0, 1, value=0.2, step=0.05, label="Complexity")
                    confidence_sl = gr.Slider(0, 1, value=0.9, step=0.05, label="Confidence")
                    budget_sl = gr.Slider(0, 1, value=0.8, step=0.05, label="Budget")

                with gr.Row():
                    error_sl = gr.Slider(0, 1, value=0.05, step=0.05, label="Error Rate")
                    urgency_sl = gr.Slider(0, 1, value=0.3, step=0.05, label="Urgency")
                    risk_sl = gr.Slider(0, 1, value=0.1, step=0.05, label="Risk")

                gradient_html = gr.HTML(label="Gradient Bars")

                with gr.Row():
                    with gr.Column():
                        hints_md = gr.Markdown(label="Strategy Hints")
                    with gr.Column():
                        context_md = gr.Markdown(label="Context Injection")

                phenotype_md = gr.Markdown(label="Phenotype & Signals")

                manual_preset_dd.change(
                    fn=_load_manual_preset,
                    inputs=[manual_preset_dd],
                    outputs=[complexity_sl, confidence_sl, budget_sl, error_sl, urgency_sl, risk_sl],
                )

                manual_btn.click(
                    fn=run_manual_gradient,
                    inputs=[manual_preset_dd, complexity_sl, confidence_sl, budget_sl, error_sl, urgency_sl, risk_sl],
                    outputs=[gradient_html, hints_md, context_md, phenotype_md],
                )

            # ── Tab 2: Orchestrator Simulation ────────────────────────
            with gr.TabItem("Orchestrator Simulation"):
                with gr.Row():
                    orch_preset_dd = gr.Dropdown(
                        choices=list(ORCH_PRESETS.keys()),
                        value="Smooth sailing",
                        label="Preset",
                        scale=2,
                    )
                    orch_btn = gr.Button("Run Simulation", variant="primary", scale=1)

                steps_tb = gr.Textbox(
                    lines=8,
                    label="Steps (one per line: 'success:tokens' or 'fail:tokens')",
                    placeholder="success:200\nfail:150\nsuccess:180\n…",
                )
                budget_orch_sl = gr.Slider(100, 5000, value=2000, step=100, label="Total budget (tokens)")

                orch_gradient_html = gr.HTML(label="Final Gradient")

                with gr.Row():
                    with gr.Column(scale=2):
                        orch_timeline_md = gr.Markdown(label="Step Timeline")
                    with gr.Column(scale=1):
                        orch_phenotype_md = gr.Markdown(label="Final Phenotype")

                orch_preset_dd.change(
                    fn=_load_orch_preset,
                    inputs=[orch_preset_dd],
                    outputs=[steps_tb, budget_orch_sl],
                )

                orch_btn.click(
                    fn=run_orchestrator,
                    inputs=[orch_preset_dd, steps_tb, budget_orch_sl],
                    outputs=[orch_gradient_html, orch_timeline_md, orch_phenotype_md],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
