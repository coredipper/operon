"""
Operon Signal Cascade -- Multi-Stage Amplification Pipeline
===========================================================

Feed text through a multi-stage processing cascade with amplification
factors and checkpoint gates, or explore the classic MAPK three-tier
biological signaling pattern.

Run locally:
    pip install gradio
    python space-cascade/app.py
"""

import sys
import time
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    Cascade,
    CascadeStage,
    CascadeResult,
    StageResult,
    StageStatus,
    MAPKCascade,
)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "(custom)": {"text": "", "description": "Type your own input"},
    "Safe text": {
        "text": "Hello World From Operon",
        "description": "Clean text passes all stages",
    },
    "XSS attempt": {
        "text": "Hello <script>alert('xss')</script>",
        "description": "Blocked at sanitize checkpoint -- HTML tags stripped",
    },
    "Empty input": {
        "text": "",
        "description": "Blocked at validate stage -- empty/None input rejected",
    },
    "Long text": {
        "text": (
            "The Operon framework models AI agent infrastructure using biological "
            "metaphors. Each organelle serves a distinct function: the Membrane "
            "filters threats, the Mitochondria powers computation, the Ribosome "
            "synthesizes prompts, the Chaperone validates output structure, and "
            "the Lysosome recycles waste from failed operations."
        ),
        "description": "Paragraph of text -- full pipeline with high token count",
    },
    "Single word": {
        "text": "hello",
        "description": "Minimal input -- low amplification, fast pipeline",
    },
    "Unicode text": {
        "text": "Bonjour le monde! \u2603 \u2764",
        "description": "Unicode characters pass normalization cleanly",
    },
    "Special characters": {
        "text": "test@email.com & 100% success <guaranteed>",
        "description": "Mixed special chars -- sanitize strips angle brackets",
    },
    "MAPK preset": {
        "text": "signal",
        "description": "Run with the MAPK tab for biological cascade demo",
    },
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

STATUS_COLORS = {
    StageStatus.COMPLETED: ("#22c55e", "COMPLETED"),
    StageStatus.FAILED: ("#ef4444", "FAILED"),
    StageStatus.BLOCKED: ("#f59e0b", "BLOCKED"),
    StageStatus.SKIPPED: ("#6b7280", "SKIPPED"),
    StageStatus.PENDING: ("#94a3b8", "PENDING"),
    StageStatus.RUNNING: ("#3b82f6", "RUNNING"),
}


def _status_badge(status: StageStatus) -> str:
    color, label = STATUS_COLORS.get(status, ("#6b7280", "UNKNOWN"))
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Text pipeline processors
# ---------------------------------------------------------------------------

def _validate(text):
    """Validate input is non-empty."""
    if text is None or (isinstance(text, str) and not text.strip()):
        raise ValueError("Empty input")
    return text


def _normalize(text):
    """Normalize whitespace and case."""
    return " ".join(str(text).strip().split())


def _tokenize(text):
    """Split into tokens."""
    tokens = str(text).split()
    return {"text": text, "tokens": tokens, "count": len(tokens)}


def _sanitize_check(text) -> bool:
    """Checkpoint: reject input with HTML/script tags."""
    s = str(text)
    return "<script" not in s.lower() and "</script" not in s.lower()


def _filter_output(data):
    """Final filter -- clean output."""
    if isinstance(data, dict):
        text = data.get("text", "")
        # Strip any remaining HTML-like tags
        import re
        clean = re.sub(r'<[^>]+>', '', str(text))
        data["text"] = clean
        data["clean"] = True
        return data
    return {"text": str(data), "tokens": str(data).split(), "count": len(str(data).split()), "clean": True}


def _build_text_cascade() -> Cascade:
    """Build the text processing cascade."""
    cascade = Cascade(name="TextPipeline", halt_on_failure=True, silent=True)
    cascade.add_stage(CascadeStage(
        name="validate",
        processor=_validate,
        amplification=1.0,
        checkpoint=lambda x: x is not None and (not isinstance(x, str) or len(x.strip()) > 0),
    ))
    cascade.add_stage(CascadeStage(
        name="normalize",
        processor=_normalize,
        amplification=1.0,
    ))
    cascade.add_stage(CascadeStage(
        name="sanitize",
        processor=lambda x: x,  # pass-through; checkpoint does the work
        amplification=1.5,
        checkpoint=_sanitize_check,
    ))
    cascade.add_stage(CascadeStage(
        name="tokenize",
        processor=_tokenize,
        amplification=2.0,
    ))
    cascade.add_stage(CascadeStage(
        name="filter",
        processor=_filter_output,
        amplification=1.0,
    ))
    return cascade


# ---------------------------------------------------------------------------
# Core logic -- Text Pipeline
# ---------------------------------------------------------------------------

def run_text_pipeline(preset_name: str, custom_text: str) -> tuple[str, str, str]:
    """Run the text cascade pipeline.

    Returns (result_html, trace_html, summary_md).
    """
    text = custom_text.strip() if custom_text.strip() else PRESETS.get(preset_name, {}).get("text", "")

    cascade = _build_text_cascade()
    result: CascadeResult = cascade.run(text if text else None)

    # --- Result banner ---
    if result.success:
        final = result.final_output
        if isinstance(final, dict):
            display = final.get("text", str(final))
            token_count = final.get("count", "?")
        else:
            display = str(final)
            token_count = len(str(final).split())

        result_html = (
            f'<div style="padding:16px;border-radius:8px;border:2px solid #22c55e;background:#f0fdf4;">'
            f'<div style="font-size:1.3em;font-weight:700;color:#16a34a;margin-bottom:8px;">'
            f'Cascade Complete</div>'
            f'<div style="font-family:monospace;color:#15803d;font-size:1.1em;margin:8px 0;">'
            f'{display}</div>'
            f'<div style="font-size:0.85em;color:#6b7280;">'
            f'Tokens: {token_count} | '
            f'Total amplification: {result.total_amplification:.1f}x | '
            f'Time: {result.total_time_ms:.2f} ms</div>'
            f'</div>'
        )
    else:
        blocked_at = result.blocked_at or "Unknown"
        result_html = (
            f'<div style="padding:16px;border-radius:8px;border:2px solid #ef4444;background:#fef2f2;">'
            f'<div style="font-size:1.3em;font-weight:700;color:#dc2626;margin-bottom:8px;">'
            f'Cascade Blocked at: {blocked_at}</div>'
            f'<div style="font-size:0.9em;color:#991b1b;">'
            f'Completed {result.stages_completed}/{result.stages_total} stages | '
            f'Time: {result.total_time_ms:.2f} ms</div>'
            f'</div>'
        )

    # --- Stage trace ---
    trace_html = ""
    stage_colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#22c55e", "#6366f1"]
    for i, sr in enumerate(result.stage_results):
        color = stage_colors[i % len(stage_colors)]
        status_color = "#16a34a" if sr.status == StageStatus.COMPLETED else "#dc2626"

        input_preview = str(sr.input_signal)[:80] + ("..." if len(str(sr.input_signal)) > 80 else "")
        output_preview = str(sr.output_signal)[:80] + ("..." if len(str(sr.output_signal)) > 80 else "")

        trace_html += (
            f'<div style="border-left:4px solid {color};padding:8px 12px;margin:8px 0;'
            f'background:#f9fafb;border-radius:0 4px 4px 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:700;color:{color};">'
            f'Stage {i+1}: {sr.stage_name}</span>'
            f'{_status_badge(sr.status)}'
            f'</div>'
            f'<div style="font-size:0.85em;color:#374151;margin-top:4px;">'
            f'In: <code>{input_preview}</code></div>'
            f'<div style="font-size:0.85em;color:#374151;">'
            f'Out: <code>{output_preview}</code></div>'
            f'<div style="font-size:0.8em;color:#6b7280;margin-top:2px;">'
            f'Amplification: {sr.amplification_factor:.1f}x | '
            f'Time: {sr.processing_time_ms:.3f} ms'
        )
        if sr.error:
            trace_html += f' | Error: <span style="color:#dc2626;">{sr.error}</span>'
        trace_html += '</div></div>'

    # --- Summary ---
    summary_md = "### Pipeline Summary\n\n"
    summary_md += f"**Stages completed:** {result.stages_completed}/{result.stages_total}\n\n"
    summary_md += f"**Total amplification:** {result.total_amplification:.1f}x\n\n"
    summary_md += f"**Total time:** {result.total_time_ms:.2f} ms\n\n"
    if result.blocked_at:
        summary_md += f"**Blocked at:** {result.blocked_at}\n\n"

    summary_md += "### Stage Roles\n\n"
    summary_md += "| Stage | Role |\n"
    summary_md += "|-------|------|\n"
    summary_md += "| validate | Reject empty/None input |\n"
    summary_md += "| normalize | Normalize whitespace |\n"
    summary_md += "| sanitize | Checkpoint blocks script tags |\n"
    summary_md += "| tokenize | Split into tokens (2x amplification) |\n"
    summary_md += "| filter | Final cleanup pass |\n"

    return result_html, trace_html, summary_md


# ---------------------------------------------------------------------------
# Core logic -- MAPK Cascade
# ---------------------------------------------------------------------------

def run_mapk(tier1_amp: float, tier2_amp: float, tier3_amp: float) -> tuple[str, str, str]:
    """Run the MAPK cascade with configurable amplification.

    Returns (result_html, trace_html, summary_md).
    """
    mapk = MAPKCascade(
        name="MAPK-Demo",
        tier1_amplification=tier1_amp,
        tier2_amplification=tier2_amp,
        tier3_amplification=tier3_amp,
        silent=True,
    )

    result: CascadeResult = mapk.run("signal")
    total_amp = tier1_amp * tier2_amp * tier3_amp

    # --- Result banner ---
    result_html = (
        f'<div style="padding:16px;border-radius:8px;border:2px solid #8b5cf6;background:#f5f3ff;">'
        f'<div style="font-size:1.3em;font-weight:700;color:#7c3aed;margin-bottom:8px;">'
        f'MAPK Cascade Result</div>'
        f'<div style="font-family:monospace;color:#5b21b6;font-size:1.1em;margin:8px 0;">'
        f'Final output: {result.final_output}</div>'
        f'<div style="display:flex;gap:20px;font-size:0.9em;color:#6b7280;flex-wrap:wrap;">'
        f'<span>Configured amplification: <b>{total_amp:.0f}x</b></span>'
        f'<span>Actual amplification: <b>{result.total_amplification:.1f}x</b></span>'
        f'<span>Time: <b>{result.total_time_ms:.2f} ms</b></span>'
        f'</div>'
        f'</div>'
    )

    # --- Stage trace ---
    tier_names = ["MAPKKK (Tier 1)", "MAPKK (Tier 2)", "MAPK (Tier 3)"]
    tier_colors = ["#ef4444", "#f59e0b", "#22c55e"]
    trace_html = ""
    for i, sr in enumerate(result.stage_results):
        color = tier_colors[i] if i < len(tier_colors) else "#6b7280"
        label = tier_names[i] if i < len(tier_names) else sr.stage_name

        trace_html += (
            f'<div style="border-left:4px solid {color};padding:8px 12px;margin:8px 0;'
            f'background:#f9fafb;border-radius:0 4px 4px 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:700;color:{color};">{label}</span>'
            f'{_status_badge(sr.status)}'
            f'</div>'
            f'<div style="font-size:0.85em;color:#374151;margin-top:4px;">'
            f'In: <code>{sr.input_signal}</code> | Out: <code>{sr.output_signal}</code></div>'
            f'<div style="font-size:0.8em;color:#6b7280;margin-top:2px;">'
            f'Amplification: {sr.amplification_factor:.1f}x | '
            f'Time: {sr.processing_time_ms:.3f} ms</div>'
            f'</div>'
        )

    # --- Summary ---
    summary_md = "### MAPK Signaling\n\n"
    summary_md += f"**Tier 1 (MAPKKK):** {tier1_amp:.0f}x amplification\n\n"
    summary_md += f"**Tier 2 (MAPKK):** {tier2_amp:.0f}x amplification\n\n"
    summary_md += f"**Tier 3 (MAPK):** {tier3_amp:.0f}x amplification\n\n"
    summary_md += f"**Total:** {tier1_amp:.0f} x {tier2_amp:.0f} x {tier3_amp:.0f} = **{total_amp:.0f}x**\n\n"
    summary_md += "### How MAPK Works\n\n"
    summary_md += "The MAPK (Mitogen-Activated Protein Kinase) cascade is a "
    summary_md += "three-tier signaling pathway found in all eukaryotic cells. "
    summary_md += "Each tier amplifies the signal from the previous tier, "
    summary_md += "enabling a small extracellular signal to produce a large "
    summary_md += "intracellular response.\n"

    return result_html, trace_html, summary_md


def load_preset(name: str) -> str:
    return PRESETS.get(name, {}).get("text", "")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Signal Cascade") as app:
        gr.Markdown(
            "# Operon Signal Cascade\n"
            "Multi-stage signal processing with **amplification factors** and "
            "**checkpoint gates**. Each stage transforms the signal; checkpoints "
            "can block progression.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Tabs():
            with gr.TabItem("Text Pipeline"):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="(custom)",
                        label="Load Preset",
                        scale=2,
                    )
                    run_btn = gr.Button("Run Cascade", variant="primary", scale=1)

                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Type text to process through the cascade...",
                    lines=3,
                )

                result_html = gr.HTML(label="Result")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Stage Trace")
                        trace_html = gr.HTML()
                    with gr.Column(scale=1):
                        summary_md = gr.Markdown()

                run_btn.click(
                    fn=run_text_pipeline,
                    inputs=[preset_dropdown, text_input],
                    outputs=[result_html, trace_html, summary_md],
                )
                text_input.submit(
                    fn=run_text_pipeline,
                    inputs=[preset_dropdown, text_input],
                    outputs=[result_html, trace_html, summary_md],
                )
                preset_dropdown.change(
                    fn=load_preset,
                    inputs=[preset_dropdown],
                    outputs=[text_input],
                )

            with gr.TabItem("MAPK Cascade"):
                gr.Markdown(
                    "### MAPK Three-Tier Signaling\n\n"
                    "The classic MAPKKK > MAPKK > MAPK biological cascade. "
                    "Configure amplification at each tier and see how signal "
                    "strength compounds through the pathway."
                )

                with gr.Row():
                    tier1_slider = gr.Slider(
                        minimum=1, maximum=20, value=10, step=1,
                        label="Tier 1 (MAPKKK) Amplification",
                    )
                    tier2_slider = gr.Slider(
                        minimum=1, maximum=20, value=10, step=1,
                        label="Tier 2 (MAPKK) Amplification",
                    )
                    tier3_slider = gr.Slider(
                        minimum=1, maximum=20, value=10, step=1,
                        label="Tier 3 (MAPK) Amplification",
                    )

                mapk_btn = gr.Button("Run MAPK Cascade", variant="primary")

                mapk_result_html = gr.HTML(label="MAPK Result")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Tier Trace")
                        mapk_trace_html = gr.HTML()
                    with gr.Column(scale=1):
                        mapk_summary_md = gr.Markdown()

                mapk_btn.click(
                    fn=run_mapk,
                    inputs=[tier1_slider, tier2_slider, tier3_slider],
                    outputs=[mapk_result_html, mapk_trace_html, mapk_summary_md],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
