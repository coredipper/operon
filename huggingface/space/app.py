"""
Operon Chaperone — Interactive Gradio Demo
==========================================

Try the Chaperone's multi-strategy cascade for recovering structured data
from malformed LLM output. Paste broken JSON, pick a schema, and watch
the cascade (STRICT -> EXTRACTION -> LENIENT -> REPAIR) recover it.

Run locally:
    pip install gradio
    python space/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import json
import sys
from pathlib import Path

import gradio as gr
from pydantic import BaseModel

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import Chaperone, FoldingStrategy, EnhancedFoldedProtein


# ---------------------------------------------------------------------------
# Preset schemas
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    name: str
    age: int
    email: str


class TaskItem(BaseModel):
    title: str
    completed: bool
    priority: int


class APIResponse(BaseModel):
    status: str
    code: int
    data: dict


class FunctionCall(BaseModel):
    name: str
    arguments: dict


SCHEMAS: dict[str, type[BaseModel]] = {
    "UserProfile (name, age, email)": UserProfile,
    "TaskItem (title, completed, priority)": TaskItem,
    "APIResponse (status, code, data)": APIResponse,
    "FunctionCall (name, arguments)": FunctionCall,
}


# ---------------------------------------------------------------------------
# Preset examples
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, dict[str, str]] = {
    "Perfect JSON (STRICT)": {
        "input": '{"name": "Alice", "age": 30, "email": "alice@example.com"}',
        "schema": "UserProfile (name, age, email)",
    },
    "Markdown-wrapped (EXTRACTION)": {
        "input": (
            "Here is the data you requested:\n\n"
            "```json\n"
            '{"name": "Bob", "age": 25, "email": "bob@example.com"}\n'
            "```\n\n"
            "Let me know if you need anything else!"
        ),
        "schema": "UserProfile (name, age, email)",
    },
    "Wrong types (LENIENT)": {
        "input": '{"name": "Charlie", "age": "35", "email": "charlie@example.com"}',
        "schema": "UserProfile (name, age, email)",
    },
    "Single quotes + trailing comma (REPAIR)": {
        "input": "{'name': 'Diana', 'age': 28, 'email': 'diana@example.com',}",
        "schema": "UserProfile (name, age, email)",
    },
    "Python literals (REPAIR)": {
        "input": '{"title": "Buy groceries", "completed": True, "priority": 3}',
        "schema": "TaskItem (title, completed, priority)",
    },
    "XML-tagged JSON (EXTRACTION)": {
        "input": (
            "Processing complete.\n"
            '<json>{"status": "ok", "code": 200, "data": {"key": "value"}}</json>\n'
            "End of response."
        ),
        "schema": "APIResponse (status, code, data)",
    },
    "Unquoted keys (REPAIR)": {
        "input": '{name: "Eve", age: 42, email: "eve@example.com"}',
        "schema": "UserProfile (name, age, email)",
    },
    "Function call in code block (EXTRACTION)": {
        "input": (
            "I'll call the weather function for you:\n\n"
            "```\n"
            '{"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}}\n'
            "```"
        ),
        "schema": "FunctionCall (name, arguments)",
    },
    "Completely invalid (ALL FAIL)": {
        "input": "This is not JSON at all, just plain text about the weather.",
        "schema": "UserProfile (name, age, email)",
    },
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

STRATEGY_LABELS = {
    FoldingStrategy.STRICT: ("STRICT", "#22c55e"),       # green
    FoldingStrategy.EXTRACTION: ("EXTRACTION", "#3b82f6"),  # blue
    FoldingStrategy.LENIENT: ("LENIENT", "#f59e0b"),      # amber
    FoldingStrategy.REPAIR: ("REPAIR", "#ef4444"),        # red
}


def _format_confidence_bar(confidence: float) -> str:
    pct = int(confidence * 100)
    color = "#22c55e" if pct >= 90 else "#f59e0b" if pct >= 70 else "#ef4444"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="flex:1;background:#e5e7eb;border-radius:4px;height:20px;max-width:200px;">'
        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;"></div>'
        f'</div>'
        f'<span style="font-weight:600;">{pct}%</span>'
        f'</div>'
    )


def _format_strategy_badge(strategy: FoldingStrategy | None) -> str:
    if strategy is None:
        return '<span style="background:#6b7280;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">NONE</span>'
    label, color = STRATEGY_LABELS[strategy]
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">{label}</span>'


def run_chaperone(raw_input: str, schema_name: str) -> tuple[str, str, str, str]:
    """Run the Chaperone cascade and return formatted results.

    Returns (result_html, parsed_json, cascade_trace, coercions).
    """
    if not raw_input.strip():
        return "Enter some text to fold.", "", "", ""

    schema_cls = SCHEMAS.get(schema_name)
    if schema_cls is None:
        return f"Unknown schema: {schema_name}", "", "", ""

    chap = Chaperone(silent=True)
    result: EnhancedFoldedProtein = chap.fold_enhanced(raw_input, schema_cls)

    # --- Result summary ---
    if result.valid:
        badge = _format_strategy_badge(result.strategy_used)
        conf_bar = _format_confidence_bar(result.confidence)
        result_html = (
            f'<div style="padding:12px;border-radius:8px;border:1px solid #22c55e;background:#f0fdf4;">'
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
            f'<span style="font-size:1.2em;font-weight:700;color:#16a34a;">Folded Successfully</span>'
            f'{badge}'
            f'</div>'
            f'<div style="margin-top:8px;"><strong>Confidence:</strong> {conf_bar}</div>'
            f'</div>'
        )
    else:
        result_html = (
            f'<div style="padding:12px;border-radius:8px;border:1px solid #ef4444;background:#fef2f2;">'
            f'<span style="font-size:1.2em;font-weight:700;color:#dc2626;">All strategies failed</span>'
            f'<p style="margin-top:8px;color:#7f1d1d;">{result.error_trace}</p>'
            f'</div>'
        )

    # --- Parsed output ---
    if result.valid and result.structure is not None:
        parsed_json = json.dumps(result.structure.model_dump(), indent=2)
    else:
        parsed_json = ""

    # --- Cascade trace ---
    trace_rows = []
    for attempt in result.attempts:
        status = "Pass" if attempt.success else "Fail"
        status_color = "#22c55e" if attempt.success else "#ef4444"
        error_text = attempt.error[:80] if attempt.error else "-"
        trace_rows.append(
            f"| {_format_strategy_badge(attempt.strategy)} | "
            f'<span style="color:{status_color};font-weight:600;">{status}</span> | '
            f"{attempt.duration_ms:.1f} ms | "
            f"`{error_text}` |"
        )

    if trace_rows:
        cascade_trace = (
            "| Strategy | Result | Duration | Error |\n"
            "|----------|--------|----------|-------|\n"
            + "\n".join(trace_rows)
        )
    else:
        cascade_trace = "No attempts recorded (input may be empty)."

    # --- Coercions ---
    if result.coercions_applied:
        coercions = "\n".join(f"- `{c}`" for c in result.coercions_applied)
    else:
        coercions = "No coercions or repairs needed." if result.valid else "N/A (folding failed)."

    return result_html, parsed_json, cascade_trace, coercions


def load_example(example_name: str) -> tuple[str, str]:
    """Load a preset example into the input fields."""
    if example_name in EXAMPLES:
        ex = EXAMPLES[example_name]
        return ex["input"], ex["schema"]
    return "", list(SCHEMAS.keys())[0]


# ---------------------------------------------------------------------------
# BFCL benchmark results
# ---------------------------------------------------------------------------

BFCL_RESULTS_MD = """
## BFCL v4 Benchmark Results

The Chaperone cascade improves function-call accuracy by recovering valid
structured output from malformed LLM responses. These results were obtained
by wrapping base models in **prompting mode** (no native function calling)
with Operon's Chaperone decode pipeline.

### Non-Live (Synthetic Test Cases)

| Model | Overall | Simple | Multiple | Parallel | Parallel Multiple | Irrelevance |
|-------|---------|--------|----------|----------|-------------------|-------------|
| GPT-4o-mini + Chaperone | **88.73%** | 79.42% | 94.00% | 92.00% | 89.50% | 87.92% |
| Gemini-2.5-Flash + Chaperone | **88.65%** | 78.08% | 92.50% | 95.50% | 88.50% | 93.33% |

### Live (Real-World API Calls)

| Model | Overall | Simple | Multiple | Parallel | Parallel Multiple | Irrelevance | Relevance |
|-------|---------|--------|----------|----------|-------------------|-------------|-----------|
| Gemini-2.5-Flash + Chaperone | **78.31%** | 87.60% | 75.97% | 81.25% | 79.17% | 87.78% | 62.50% |
| GPT-4o-mini + Chaperone | **76.98%** | 80.23% | 76.16% | 93.75% | 66.67% | 78.85% | 93.75% |

### How It Works

The Chaperone wraps any LLM's text output in a **4-stage cascade**:

1. **STRICT** -- Direct `json.loads()` parse. No modifications. Confidence: 100%.
2. **EXTRACTION** -- Find JSON inside markdown blocks, XML tags, or bare objects. Confidence: 90%.
3. **LENIENT** -- Extract JSON and coerce types (e.g., `"42"` to `42`). Confidence: ~80%.
4. **REPAIR** -- Fix trailing commas, single quotes, Python literals (`None`/`True`/`False`), unquoted keys. Confidence: ~70%.

Each strategy is tried in order. The first one that produces valid output (passes Pydantic validation) wins.

[View on GitHub](https://github.com/coredipper/operon) | [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Chaperone Demo") as app:
        gr.Markdown(
            "# Operon Chaperone\n"
            "Recover structured data from malformed LLM output through a "
            "multi-strategy cascade: **STRICT** > **EXTRACTION** > **LENIENT** > **REPAIR**.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Tabs():
            # ---- Tab 1: Interactive Demo ----
            with gr.TabItem("Try It"):
                with gr.Row():
                    example_dropdown = gr.Dropdown(
                        choices=["(custom)"] + list(EXAMPLES.keys()),
                        value="(custom)",
                        label="Load Example",
                        scale=2,
                    )
                    schema_dropdown = gr.Dropdown(
                        choices=list(SCHEMAS.keys()),
                        value=list(SCHEMAS.keys())[0],
                        label="Target Schema",
                        scale=2,
                    )

                raw_input = gr.Textbox(
                    label="Raw LLM Output",
                    placeholder='Paste malformed JSON here, e.g. {\'name\': "Alice", \'age\': 30,}',
                    lines=6,
                )

                fold_btn = gr.Button("Fold", variant="primary", size="lg")

                result_html = gr.HTML(label="Result")

                with gr.Row():
                    parsed_output = gr.Code(
                        label="Parsed Output",
                        language="json",
                        lines=8,
                    )

                with gr.Row():
                    with gr.Column():
                        cascade_trace = gr.Markdown(label="Cascade Trace")
                    with gr.Column():
                        coercions_md = gr.Markdown(label="Repairs / Coercions Applied")

                # Wire events
                fold_btn.click(
                    fn=run_chaperone,
                    inputs=[raw_input, schema_dropdown],
                    outputs=[result_html, parsed_output, cascade_trace, coercions_md],
                )

                raw_input.submit(
                    fn=run_chaperone,
                    inputs=[raw_input, schema_dropdown],
                    outputs=[result_html, parsed_output, cascade_trace, coercions_md],
                )

                example_dropdown.change(
                    fn=load_example,
                    inputs=[example_dropdown],
                    outputs=[raw_input, schema_dropdown],
                )

            # ---- Tab 2: BFCL Results ----
            with gr.TabItem("BFCL Results"):
                gr.Markdown(BFCL_RESULTS_MD)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
