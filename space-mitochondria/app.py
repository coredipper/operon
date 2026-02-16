"""
Operon Mitochondria -- Safe Calculator Demo
============================================

Interactive calculator powered by AST-based expression parsing.
No arbitrary code execution, no code injection -- expressions are parsed
into an abstract syntax tree and only whitelisted operations are allowed.

Supports four metabolic pathways:
- Glycolysis: Safe arithmetic (sqrt, sin, pi, factorial, ...)
- Krebs Cycle: Boolean/comparison logic
- Beta Oxidation: JSON and Python literal parsing
- Auto-detection: Mitochondria picks the right pathway

Run locally:
    pip install gradio
    python space-mitochondria/app.py
"""

import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import Mitochondria, MetabolicPathway


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, dict] = {
    "(custom)": {"expr": "", "pathway": "Auto-detect"},
    "Basic arithmetic": {"expr": "2 + 2 * 10", "pathway": "Auto-detect"},
    "Exponentiation": {"expr": "2 ** 10", "pathway": "Auto-detect"},
    "Math functions": {"expr": "sqrt(144) + pi", "pathway": "Glycolysis (Math)"},
    "Trigonometry": {"expr": "sin(0) + cos(0)", "pathway": "Glycolysis (Math)"},
    "Factorial": {"expr": "factorial(7)", "pathway": "Glycolysis (Math)"},
    "Natural log": {"expr": "log(e ** 3)", "pathway": "Glycolysis (Math)"},
    "Floor & ceil": {"expr": "floor(3.7) + ceil(3.2)", "pathway": "Glycolysis (Math)"},
    "Comparison": {"expr": "5 > 3 and 10 < 20", "pathway": "Krebs Cycle (Logic)"},
    "Boolean logic": {"expr": "not False or True and True", "pathway": "Krebs Cycle (Logic)"},
    "JSON parsing": {"expr": '{"name": "Alice", "age": 30}', "pathway": "Beta Oxidation (Data)"},
    "Python tuple": {"expr": "(1, 2, 3)", "pathway": "Beta Oxidation (Data)"},
    "Injection attempt": {"expr": "__import__('os').system('ls')", "pathway": "Auto-detect"},
    "Undefined variable": {"expr": "secret_password", "pathway": "Auto-detect"},
    "Division by zero": {"expr": "1 / 0", "pathway": "Auto-detect"},
}

PATHWAY_MAP = {
    "Auto-detect": None,
    "Glycolysis (Math)": MetabolicPathway.GLYCOLYSIS,
    "Krebs Cycle (Logic)": MetabolicPathway.KREBS_CYCLE,
    "Beta Oxidation (Data)": MetabolicPathway.BETA_OXIDATION,
}

PATHWAY_LABELS = {
    MetabolicPathway.GLYCOLYSIS: ("Glycolysis", "#22c55e", "Safe arithmetic via AST"),
    MetabolicPathway.KREBS_CYCLE: ("Krebs Cycle", "#3b82f6", "Boolean/comparison logic"),
    MetabolicPathway.OXIDATIVE: ("Oxidative", "#a855f7", "External tool execution"),
    MetabolicPathway.BETA_OXIDATION: ("Beta Oxidation", "#f59e0b", "Data transformation"),
}

SAFE_FUNCTIONS = sorted([
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "log", "log10", "log2", "exp", "pow",
    "ceil", "floor", "trunc", "factorial", "gcd",
    "abs", "round", "min", "max", "sum",
    "degrees", "radians",
    "pi", "e", "tau", "inf",
])


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def evaluate(expression: str, pathway_str: str) -> tuple[str, str, str]:
    """Evaluate an expression through the Mitochondria.

    Returns (result_html, details_md, functions_md).
    """
    if not expression.strip():
        return "Enter an expression to evaluate.", "", ""

    pathway = PATHWAY_MAP.get(pathway_str)
    mito = Mitochondria(silent=True)
    result = mito.metabolize(expression, pathway)

    # --- Result banner ---
    if result.success and result.atp:
        p_label, p_color, p_desc = PATHWAY_LABELS.get(
            result.atp.pathway,
            ("Unknown", "#6b7280", ""),
        )
        pathway_badge = (
            f'<span style="background:{p_color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;">{p_label}</span>'
        )

        value_str = repr(result.atp.value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."

        result_html = (
            f'<div style="padding:16px;border-radius:8px;border:2px solid #22c55e;background:#f0fdf4;">'
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
            f'<span style="font-size:1.3em;font-weight:700;color:#16a34a;">Result</span>'
            f'{pathway_badge}'
            f'</div>'
            f'<div style="font-size:1.5em;font-family:monospace;color:#15803d;margin:8px 0;">'
            f'{value_str}'
            f'</div>'
            f'<div style="font-size:0.85em;color:#6b7280;">Type: {type(result.atp.value).__name__}</div>'
            f'</div>'
        )
    else:
        result_html = (
            f'<div style="padding:16px;border-radius:8px;border:2px solid #ef4444;background:#fef2f2;">'
            f'<div style="font-size:1.2em;font-weight:700;color:#dc2626;margin-bottom:8px;">'
            f'Metabolic Failure</div>'
            f'<div style="font-family:monospace;color:#991b1b;font-size:0.95em;">'
            f'{result.error}</div>'
            f'</div>'
        )

    # --- Details ---
    details_md = ""
    if result.success and result.atp:
        details_md += f"**Pathway:** {p_desc}\n\n"
        details_md += f"**Efficiency:** {result.atp.efficiency:.2%}\n\n"
        details_md += f"**Execution Time:** {result.atp.execution_time_ms:.3f} ms\n\n"

    details_md += f"**ROS Level:** {result.ros_level:.2f}"
    if result.ros_level > 0:
        details_md += " (accumulated from errors)"
    details_md += "\n\n"

    stats = mito.get_statistics()
    details_md += f"**Health:** {stats['health']}\n"

    # --- Available functions ---
    func_md = "**Available math functions and constants:**\n\n"
    func_md += "| Function | Function | Function | Function |\n"
    func_md += "|----------|----------|----------|----------|\n"
    for i in range(0, len(SAFE_FUNCTIONS), 4):
        row = SAFE_FUNCTIONS[i:i+4]
        row += [""] * (4 - len(row))
        func_md += "| " + " | ".join(f"`{f}`" if f else "" for f in row) + " |\n"

    func_md += "\n**Operators:** `+` `-` `*` `/` `//` `%` `**`\n\n"
    func_md += "**Comparisons:** `==` `!=` `<` `>` `<=` `>=` `and` `or` `not`\n"

    return result_html, details_md, func_md


def load_example(name: str) -> tuple[str, str]:
    ex = EXAMPLES.get(name, {"expr": "", "pathway": "Auto-detect"})
    return ex["expr"], ex["pathway"]


# ---------------------------------------------------------------------------
# ROS accumulation demo
# ---------------------------------------------------------------------------

def run_ros_demo() -> str:
    """Demonstrate ROS accumulation and repair."""
    mito = Mitochondria(max_ros=0.3, silent=True)
    lines = []
    lines.append("**ROS Accumulation & Mitophagy Demo**\n")
    lines.append(f"Max ROS before dysfunction: 0.30\n")
    lines.append("")

    bad_exprs = ["invalid syntax", "undefined_var", "1 / 0"]
    for expr in bad_exprs:
        result = mito.metabolize(expr)
        lines.append(f"| `{expr}` | ROS: {result.ros_level:.2f} | {result.error[:50]} |")

    result = mito.metabolize("2 + 2")
    if not result.success:
        lines.append(f"\n`2 + 2` fails while dysfunctional: *{result.error[:60]}*\n")

    mito.repair(amount=0.5)
    result = mito.metabolize("2 + 2")
    lines.append(f"After repair (mitophagy): ROS = {mito.get_ros_level():.2f}")
    if result.success:
        lines.append(f"`2 + 2` = {result.atp.value} -- operations restored!")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Mitochondria") as app:
        gr.Markdown(
            "# Operon Mitochondria -- Safe Calculator\n"
            "AST-based expression parsing with **zero code injection risk**. "
            "Expressions are parsed into an abstract syntax tree and only whitelisted "
            "operations execute.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Tabs():
            with gr.TabItem("Calculator"):
                with gr.Row():
                    example_dropdown = gr.Dropdown(
                        choices=list(EXAMPLES.keys()),
                        value="(custom)",
                        label="Load Example",
                        scale=2,
                    )
                    pathway_dropdown = gr.Dropdown(
                        choices=list(PATHWAY_MAP.keys()),
                        value="Auto-detect",
                        label="Pathway",
                        scale=2,
                    )
                    eval_btn = gr.Button("Evaluate", variant="primary", scale=1)

                expr_input = gr.Textbox(
                    label="Expression",
                    placeholder="e.g. sqrt(144) + pi * 2",
                    lines=2,
                )

                result_html = gr.HTML(label="Result")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Details")
                        details_md = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### Available Functions")
                        functions_md = gr.Markdown()

                eval_btn.click(
                    fn=evaluate,
                    inputs=[expr_input, pathway_dropdown],
                    outputs=[result_html, details_md, functions_md],
                )
                expr_input.submit(
                    fn=evaluate,
                    inputs=[expr_input, pathway_dropdown],
                    outputs=[result_html, details_md, functions_md],
                )
                example_dropdown.change(
                    fn=load_example,
                    inputs=[example_dropdown],
                    outputs=[expr_input, pathway_dropdown],
                )

            with gr.TabItem("ROS & Self-Repair"):
                gr.Markdown(
                    "### Reactive Oxygen Species (ROS) Management\n\n"
                    "Errors accumulate ROS damage. When the threshold is exceeded, "
                    "the mitochondria becomes dysfunctional -- even valid operations fail. "
                    "**Mitophagy** (repair) restores function.\n\n"
                    "This mirrors how biological mitochondria accumulate oxidative damage "
                    "and are recycled when damaged beyond repair."
                )
                ros_btn = gr.Button("Run ROS Demo", variant="primary")
                ros_output = gr.Markdown()
                ros_btn.click(fn=run_ros_demo, outputs=[ros_output])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
