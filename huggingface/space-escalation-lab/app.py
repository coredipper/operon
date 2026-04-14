"""
Operon Escalation Lab -- Quality-Based Model Escalation
========================================================

Interactive demo of the adaptive immune layer:
VerifierComponent evaluates output quality via a rubric, and
WatcherComponent escalates from fast -> deep model when quality
falls below threshold.

Run locally:  pip install gradio && python space-escalation-lab/app.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.verifier import VerifierComponent, VerifierConfig
from operon_ai.patterns.watcher import WatcherComponent, WatcherConfig

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    task: str
    fast_response: str
    deep_response: str
    fast_quality: float   # expected quality of fast response
    description: str


SCENARIOS = {
    "Shallow bug fix": Scenario(
        name="Shallow bug fix",
        task="Fix the login crash after session timeout",
        fast_response="Add try/except around the login call.",
        deep_response=(
            "Root-cause analysis: the session token is not refreshed on 401 "
            "retry. Fix: add token refresh in the retry interceptor with "
            "exponential backoff. Added regression test for expired-token path."
        ),
        fast_quality=0.25,
        description="Fast model produces a shallow patch; deep model finds root cause.",
    ),
    "Vague summary": Scenario(
        name="Vague summary",
        task="Summarize the Q3 performance report",
        fast_response="Performance was good in Q3.",
        deep_response=(
            "Q3 highlights: revenue up 12% YoY driven by enterprise segment "
            "(+23%). Churn decreased from 4.1% to 3.2% after onboarding "
            "redesign. Two risks: APAC pipeline softening (-8%) and delayed "
            "SOC2 certification (ETA pushed to Q4)."
        ),
        fast_quality=0.15,
        description="Fast model gives a vague one-liner; deep model gives structured detail.",
    ),
    "Adequate response": Scenario(
        name="Adequate response",
        task="List the three main HTTP status code categories",
        fast_response=(
            "1xx Informational, 2xx Success, 3xx Redirection, 4xx Client Error, "
            "5xx Server Error. The three main categories are 2xx, 4xx, and 5xx."
        ),
        deep_response=(
            "The three main HTTP status code categories are 2xx (Success), "
            "4xx (Client Error), and 5xx (Server Error)."
        ),
        fast_quality=0.85,
        description="Fast model gives a good enough answer. No escalation expected.",
    ),
}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _badge(text, color):
    return (f'<span style="background:{color};color:white;padding:3px 10px;'
            f'border-radius:4px;font-size:0.85em;font-weight:600;">{text}</span>')


def _card(title, content, border_color="#e5e7eb"):
    return (
        f'<div style="border:2px solid {border_color};border-radius:8px;'
        f'margin-bottom:12px;overflow:hidden;">'
        f'<div style="padding:8px 14px;background:{border_color}15;'
        f'border-bottom:1px solid {border_color};">'
        f'<span style="font-weight:700;">{title}</span></div>'
        f'<div style="padding:12px 14px;">{content}</div></div>'
    )


def run_escalation(scenario_name, threshold):
    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        return "<p>Select a scenario.</p>"

    threshold = float(threshold)

    # Build rubric that scores based on output length + specificity
    def rubric(output: str, stage_name: str) -> float:
        if stage_name != "respond":
            return 0.8
        if output == scenario.fast_response:
            return scenario.fast_quality
        return 0.95  # deep response always scores high

    # Build organism
    fast = Nucleus(provider=MockProvider(responses={
        "respond": scenario.fast_response,
    }))
    deep = Nucleus(provider=MockProvider(responses={
        "respond": scenario.deep_response,
    }))

    watcher = WatcherComponent(config=WatcherConfig())
    verifier = VerifierComponent(
        rubric=rubric,
        config=VerifierConfig(quality_low_threshold=threshold),
    )

    org = skill_organism(
        stages=[
            SkillStage(
                name="respond",
                role="Responder",
                instructions="Respond to the task.",
                mode="fixed",
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=1000, silent=True),
        components=[watcher, verifier],
    )

    result = org.run(scenario.task)

    # Collect results
    escalated = any(
        i.kind.value == "escalate" for i in watcher.interventions
    )
    fix_scores = [(s, q) for s, q in verifier.quality_scores if s == "respond"]
    initial_quality = fix_scores[0][1] if fix_scores else 0.0

    verifier_signals = [s for s in watcher.signals if s.source == "verifier"]

    # Build HTML output
    html_parts = []

    # Scenario info
    html_parts.append(_card(
        f"Scenario: {scenario.name}",
        f'<p style="color:#6b7280;">{scenario.description}</p>'
        f'<p><b>Task:</b> {scenario.task}</p>'
        f'<p><b>Threshold:</b> {threshold:.2f}</p>',
        "#6366f1",
    ))

    # Fast model output
    fast_badge = _badge(f"quality: {initial_quality:.2f}",
                        "#ef4444" if initial_quality < threshold else "#22c55e")
    below = initial_quality < threshold
    html_parts.append(_card(
        f"Fast Model Output {fast_badge}",
        f'<p style="font-family:monospace;white-space:pre-wrap;">'
        f'{scenario.fast_response}</p>'
        f'<p style="margin-top:8px;color:#6b7280;">'
        f'{"Below threshold" if below else "Above threshold"} '
        f'({initial_quality:.2f} {"<" if below else ">="} {threshold:.2f})</p>',
        "#ef4444" if below else "#22c55e",
    ))

    # Escalation decision
    if escalated:
        intv = watcher.interventions[0]
        html_parts.append(_card(
            f"Watcher Decision: {_badge('ESCALATE', '#f59e0b')}",
            f'<p><b>Reason:</b> {intv.reason}</p>'
            f'<p style="color:#6b7280;">Fast model quality ({initial_quality:.2f}) '
            f'fell below threshold ({threshold:.2f}). '
            f'Watcher escalated to deep model.</p>',
            "#f59e0b",
        ))

        html_parts.append(_card(
            f"Deep Model Output {_badge('quality: 0.95', '#22c55e')}",
            f'<p style="font-family:monospace;white-space:pre-wrap;">'
            f'{scenario.deep_response}</p>',
            "#22c55e",
        ))
    else:
        html_parts.append(_card(
            f"Watcher Decision: {_badge('NO ESCALATION', '#22c55e')}",
            f'<p>Quality ({initial_quality:.2f}) met threshold ({threshold:.2f}). '
            f'Fast model output accepted.</p>',
            "#22c55e",
        ))

    # Final output
    final_badge = _badge("ESCALATED", "#f59e0b") if escalated else _badge("DIRECT", "#22c55e")
    html_parts.append(_card(
        f"Final Output {final_badge}",
        f'<p style="font-family:monospace;white-space:pre-wrap;font-weight:600;">'
        f'{result.final_output}</p>',
        "#3b82f6",
    ))

    # Signal trace
    sig_rows = ""
    for sig in verifier_signals:
        q = sig.detail.get("quality", 0)
        bt = sig.detail.get("below_threshold", False)
        status = _badge("BELOW", "#ef4444") if bt else _badge("OK", "#22c55e")
        sig_rows += (
            f'<tr style="border-bottom:1px solid #f3f4f6;">'
            f'<td style="padding:4px 8px;">{sig.stage_name}</td>'
            f'<td style="padding:4px 8px;">{q:.2f}</td>'
            f'<td style="padding:4px 8px;">{sig.value:.2f}</td>'
            f'<td style="padding:4px 8px;">{status}</td></tr>')

    if sig_rows:
        html_parts.append(_card(
            "Signal Trace",
            '<table style="width:100%;border-collapse:collapse;">'
            '<tr style="border-bottom:2px solid #e5e7eb;color:#6b7280;">'
            '<th style="text-align:left;padding:4px 8px;">Stage</th>'
            '<th style="text-align:left;padding:4px 8px;">Quality</th>'
            '<th style="text-align:left;padding:4px 8px;">Severity</th>'
            '<th style="text-align:left;padding:4px 8px;">Status</th></tr>'
            f'{sig_rows}</table>',
            "#8b5cf6",
        ))

    return "\n".join(html_parts)


def load_scenario(name):
    s = SCENARIOS.get(name)
    if s:
        return s.description
    return ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Escalation Lab") as app:
        gr.Markdown(
            "# Operon Escalation Lab\n"
            "Explore **quality-based escalation**: the VerifierComponent scores "
            "each stage's output, and the WatcherComponent escalates from the "
            "fast model to the deep model when quality drops below threshold.\n\n"
            "**Biological analogy:** Innate immunity (Watcher) detects generic anomalies. "
            "Adaptive immunity (Verifier) evaluates against a specific rubric.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)")

        with gr.Row():
            scenario_dd = gr.Dropdown(
                choices=list(SCENARIOS.keys()),
                value="Shallow bug fix",
                label="Scenario", scale=2)
            run_btn = gr.Button("Run", variant="primary", scale=1)

        scenario_desc = gr.Markdown("Fast model produces a shallow patch; deep model finds root cause.")

        threshold_slider = gr.Slider(
            minimum=0.1, maximum=0.95, value=0.5, step=0.05,
            label="Quality Threshold (below this = escalate)")

        gr.Markdown("### Results")
        results_output = gr.HTML()

        run_btn.click(
            fn=run_escalation,
            inputs=[scenario_dd, threshold_slider],
            outputs=[results_output])
        scenario_dd.change(
            fn=load_scenario,
            inputs=[scenario_dd],
            outputs=[scenario_desc])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
