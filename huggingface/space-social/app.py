"""Operon Social Learning Dashboard — template sharing, trust, curiosity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr


# ---------------------------------------------------------------------------
# Inline simulation
# ---------------------------------------------------------------------------

@dataclass
class Template:
    template_id: str
    name: str
    success_rate: float

@dataclass
class TrustEntry:
    peer_id: str
    trust: float

ORGANISM_A_TEMPLATES = [
    Template("support", "Customer Support Pipeline", 0.85),
    Template("compliance", "Compliance Review", 0.92),
    Template("research", "Research Swarm", 0.60),
]

def _ema(old: float, outcome: float, alpha: float = 0.3) -> float:
    return alpha * outcome + (1.0 - alpha) * old

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _run_exchange(min_sr, default_trust):
    exported = [t for t in ORGANISM_A_TEMPLATES if t.success_rate >= min_sr / 100]
    adopted = []
    rejected = []
    for t in exported:
        effective = t.success_rate * (default_trust / 100)
        if effective >= 0.3:
            adopted.append(t)
        else:
            rejected.append(t)

    rows = ""
    for t in ORGANISM_A_TEMPLATES:
        in_export = t in exported
        in_adopted = t in adopted
        status = "ADOPTED" if in_adopted else ("REJECTED" if in_export else "NOT EXPORTED")
        color = "#4a4" if in_adopted else ("#a44" if in_export else "#888")
        rows += f"""<tr>
            <td style="padding:8px">{t.name}</td>
            <td style="padding:8px">{t.success_rate:.0%}</td>
            <td style="padding:8px;color:{color};font-weight:600">{status}</td>
        </tr>"""

    return f"""<div style="margin-bottom:12px;font-size:13px;color:#888">
        Trust: {default_trust}% | Min success rate: {min_sr}% | Adoption threshold: 30%
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Template</th>
            <th style="text-align:left;padding:8px">Success Rate</th>
            <th style="text-align:left;padding:8px">Status</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _run_trust_sim(n_success, n_failure):
    trust = 0.5
    history = [("initial", trust)]
    for i in range(int(n_success)):
        trust = _ema(trust, 1.0)
        history.append((f"success #{i+1}", trust))
    for i in range(int(n_failure)):
        trust = _ema(trust, 0.0)
        history.append((f"failure #{i+1}", trust))

    rows = ""
    for event, score in history:
        color = "#4a4" if score >= 0.5 else "#a44"
        bar = min(int(score * 100), 100)
        rows += f"""<tr>
            <td style="padding:6px">{event}</td>
            <td style="padding:6px">
                <span style="display:inline-block;width:{bar}%;height:12px;background:{color};border-radius:3px;opacity:0.5"></span>
                <span style="color:{color};margin-left:8px;font-size:12px">{score:.3f}</span>
            </td>
        </tr>"""

    return f"""<table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Event</th>
            <th style="text-align:left;padding:8px">Trust Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _run_curiosity(threshold):
    scenarios = [
        ("intake", "deterministic", 0.2, "healthy", False),
        ("classifier", "fast", 0.4, "healthy", False),
        ("novel_detector", "fast", 0.85, "exploring", True),
        ("analyst", "deep", 0.7, "exploring", False),
    ]
    rows = ""
    for name, model, novelty, status, _ in scenarios:
        would_escalate = status == "exploring" and model == "fast" and novelty > threshold / 100
        color = "#f97316" if would_escalate else "#4a4"
        action = "ESCALATE" if would_escalate else "OK"
        rows += f"""<tr>
            <td style="padding:8px">{name}</td>
            <td style="padding:8px">{model}</td>
            <td style="padding:8px">{novelty:.2f}</td>
            <td style="padding:8px">{status}</td>
            <td style="padding:8px;color:{color};font-weight:600">{action}</td>
        </tr>"""

    return f"""<div style="margin-bottom:12px;font-size:13px;color:#888">
        Curiosity threshold: {threshold}%
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Stage</th>
            <th style="text-align:left;padding:8px">Model</th>
            <th style="text-align:left;padding:8px">Novelty</th>
            <th style="text-align:left;padding:8px">Status</th>
            <th style="text-align:left;padding:8px">Action</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="Operon Social Learning", theme=gr.themes.Base()) as demo:
        gr.Markdown("# Operon Social Learning\nTemplate exchange, trust registry, and curiosity-driven exploration.")

        with gr.Tab("Template Exchange"):
            sr = gr.Slider(0, 100, value=60, step=5, label="Min Success Rate (%)")
            trust = gr.Slider(0, 100, value=50, step=5, label="Default Trust (%)")
            btn = gr.Button("Run Exchange")
            out = gr.HTML()
            btn.click(_run_exchange, inputs=[sr, trust], outputs=[out])

        with gr.Tab("Trust Registry"):
            ns = gr.Slider(0, 10, value=3, step=1, label="Successes")
            nf = gr.Slider(0, 10, value=2, step=1, label="Failures")
            btn2 = gr.Button("Simulate Trust")
            out2 = gr.HTML()
            btn2.click(_run_trust_sim, inputs=[ns, nf], outputs=[out2])

        with gr.Tab("Curiosity Signals"):
            thresh = gr.Slider(0, 100, value=60, step=5, label="Curiosity Threshold (%)")
            btn3 = gr.Button("Show Signals")
            out3 = gr.HTML()
            btn3.click(_run_curiosity, inputs=[thresh], outputs=[out3])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
