"""Operon Sleep Consolidation Dashboard — consolidation cycle and counterfactual replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr


# ---------------------------------------------------------------------------
# Inline simulation
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    template_id: str
    success: bool
    latency_ms: float
    tokens_used: int


@dataclass
class ConsolidationResult:
    templates_created: int
    memories_promoted: int
    histone_promotions: int
    counterfactuals: int
    duration_ms: float


@dataclass
class CounterfactualResult:
    corrections: int
    affected_stages: list[str]
    outcome_would_change: bool
    reasoning: str


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _fresh_session():
    return [], False

def _after_10_runs():
    records = [RunRecord("enterprise", i < 8, 1200 + i * 50, 3000 + i * 100) for i in range(10)]
    return records, False

def _with_corrections():
    records = [RunRecord("enterprise", i < 7, 1200 + i * 50, 3000 + i * 100) for i in range(10)]
    return records, True

PRESETS = {
    "Fresh Session (no history)": _fresh_session,
    "After 10 Runs (8 successful)": _after_10_runs,
    "With Bi-Temporal Corrections": _with_corrections,
}


def _simulate_consolidation(records, has_corrections):
    successful = sum(1 for r in records if r.success)
    templates_created = 1 if successful >= 3 else 0
    memories_promoted = min(successful, 5)
    histone_promotions = 1 if successful >= 5 else 0
    counterfactuals = 1 if has_corrections else 0

    cr = None
    if has_corrections:
        cr = CounterfactualResult(
            corrections=2,
            affected_stages=["researcher", "strategist"],
            outcome_would_change=True,
            reasoning="2 corrections found: methodology changed from quantitative to qualitative. Researcher and Strategist stages affected.",
        )

    return ConsolidationResult(
        templates_created=templates_created,
        memories_promoted=memories_promoted,
        histone_promotions=histone_promotions,
        counterfactuals=counterfactuals,
        duration_ms=0.5,
    ), cr


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _result_html(result, cf):
    metrics = f"""<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px">
        <div style="background:#0a1a0a;border:1px solid #1a2a1a;border-radius:6px;padding:16px;text-align:center">
            <div style="font-size:24px;font-weight:600;color:#6a6">{result.templates_created}</div>
            <div style="font-size:12px;color:#888">Templates Created</div>
        </div>
        <div style="background:#0a1a2a;border:1px solid #1a2a3a;border-radius:6px;padding:16px;text-align:center">
            <div style="font-size:24px;font-weight:600;color:#6af">{result.memories_promoted}</div>
            <div style="font-size:12px;color:#888">Memories Promoted</div>
        </div>
        <div style="background:#1a1a0a;border:1px solid #2a2a1a;border-radius:6px;padding:16px;text-align:center">
            <div style="font-size:24px;font-weight:600;color:#aa6">{result.histone_promotions}</div>
            <div style="font-size:12px;color:#888">Histone Promotions</div>
        </div>
    </div>"""

    cf_html = ""
    if cf:
        cf_html = f"""<div style="background:#1a0a0a;border:1px solid #2a1a1a;border-radius:6px;padding:16px;margin-top:16px">
            <div style="font-weight:600;color:#f66;margin-bottom:8px">Counterfactual Analysis</div>
            <p style="font-size:14px;color:#d0d0d0">{cf.reasoning}</p>
            <p style="font-size:13px;color:#888">Affected stages: {', '.join(cf.affected_stages)}</p>
            <p style="font-size:13px;color:#f66">Outcome would change: {'Yes' if cf.outcome_would_change else 'No'}</p>
        </div>"""

    return metrics + cf_html


def _tiers_html(result):
    working = max(0, 5 - result.memories_promoted)
    episodic = result.memories_promoted
    longterm = result.histone_promotions
    rows = ""
    for tier, count, color in [("WORKING", working, "#888"), ("EPISODIC", episodic, "#6af"), ("LONGTERM", longterm, "#6a6")]:
        bar_width = min(count * 20, 100)
        rows += f"""<div style="margin-bottom:8px">
            <span style="display:inline-block;width:100px;color:{color};font-weight:600;font-size:13px">{tier}</span>
            <span style="display:inline-block;width:{bar_width}%;height:16px;background:{color};border-radius:3px;opacity:0.4"></span>
            <span style="color:#888;font-size:12px;margin-left:8px">{count}</span>
        </div>"""
    return f'<div style="padding:12px">{rows}</div>'


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _run_consolidation(preset_name):
    if preset_name not in PRESETS:
        return "Select a preset.", ""
    records, has_corrections = PRESETS[preset_name]()
    result, cf = _simulate_consolidation(records, has_corrections)
    return _result_html(result, cf), _tiers_html(result)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="Operon Sleep Consolidation") as demo:
        gr.Markdown("# Operon Sleep Consolidation\nReplay, compress, counterfactual replay, and histone promotion.")

        with gr.Tab("Consolidation Cycle"):
            preset = gr.Dropdown(choices=list(PRESETS.keys()), label="Scenario", value="After 10 Runs (8 successful)")
            btn = gr.Button("Run Consolidation")
            out = gr.HTML()
            tiers = gr.HTML()
            btn.click(_run_consolidation, inputs=[preset], outputs=[out, tiers])

        with gr.Tab("Memory Tiers"):
            gr.Markdown("Memory tier distribution is shown in the Consolidation Cycle tab after running.")

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
