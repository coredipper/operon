"""
Operon Feedback Loop Homeostasis -- Interactive Gradio Demo
===========================================================

Simulate a NegativeFeedbackLoop controlling a value toward a setpoint.
Configure gain, damping, and disturbances to watch convergence, oscillation,
or overdamping in real time.

Run locally:
    pip install gradio
    python space-feedback/app.py

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

from operon_ai import NegativeFeedbackLoop

# ── Presets ────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own feedback loop parameters.",
        "setpoint": 0.0,
        "initial": 10.0,
        "gain": 0.5,
        "damping": 0.1,
        "iterations": 30,
        "disturbance_step": 0,
        "disturbance_magnitude": 0.0,
    },
    "Temperature control": {
        "description": "Smooth cooling from 85°F toward 72°F setpoint — classic thermostat behavior.",
        "setpoint": 72.0,
        "initial": 85.0,
        "gain": 0.3,
        "damping": 0.05,
        "iterations": 30,
        "disturbance_step": 0,
        "disturbance_magnitude": 0.0,
    },
    "Oscillating convergence": {
        "description": "High gain causes overshooting around the setpoint before settling.",
        "setpoint": 0.0,
        "initial": 10.0,
        "gain": 0.8,
        "damping": 0.0,
        "iterations": 40,
        "disturbance_step": 0,
        "disturbance_magnitude": 0.0,
    },
    "Overdamped": {
        "description": "Heavy damping — slow, stable approach with no overshoot.",
        "setpoint": 50.0,
        "initial": 0.0,
        "gain": 0.2,
        "damping": 0.3,
        "iterations": 50,
        "disturbance_step": 0,
        "disturbance_magnitude": 0.0,
    },
    "Underdamped": {
        "description": "Light damping + high gain — fast oscillations that ring before settling.",
        "setpoint": 50.0,
        "initial": 0.0,
        "gain": 0.9,
        "damping": 0.02,
        "iterations": 40,
        "disturbance_step": 0,
        "disturbance_magnitude": 0.0,
    },
    "Disturbance rejection": {
        "description": "System at setpoint gets a -30 disturbance at step 10 — watch recovery.",
        "setpoint": 100.0,
        "initial": 100.0,
        "gain": 0.3,
        "damping": 0.05,
        "iterations": 40,
        "disturbance_step": 10,
        "disturbance_magnitude": -30.0,
    },
}


def _load_preset(
    name: str,
) -> tuple[float, float, float, float, int, int, float]:
    """Return slider values for a preset."""
    p = PRESETS.get(name, PRESETS["(custom)"])
    return (
        p["setpoint"],
        p["initial"],
        p["gain"],
        p["damping"],
        p["iterations"],
        p["disturbance_step"],
        p["disturbance_magnitude"],
    )


# ── Core simulation ───────────────────────────────────────────────────────


def run_feedback(
    preset_name: str,
    setpoint: float,
    initial: float,
    gain: float,
    damping: float,
    iterations: int,
    disturbance_step: int,
    disturbance_magnitude: float,
) -> tuple[str, str, str]:
    """Run the feedback loop simulation.

    Returns (convergence_banner_html, timeline_md, analysis_md).
    """
    loop = NegativeFeedbackLoop(
        setpoint=setpoint,
        gain=gain,
        damping=damping,
        silent=True,
    )

    current = initial
    rows: list[dict] = []
    initial_error = abs(initial - setpoint)

    for step in range(1, int(iterations) + 1):
        note = ""

        # Apply disturbance
        if disturbance_step > 0 and step == int(disturbance_step):
            current += disturbance_magnitude
            note = f"Disturbance: {disturbance_magnitude:+.1f}"

        error_before = current - setpoint
        correction = loop.apply(current)
        current = correction  # apply() returns the corrected value

        error_after = current - setpoint

        rows.append({
            "step": step,
            "value": current,
            "error": error_after,
            "correction": current - (error_before + setpoint),
            "note": note,
        })

    # ── Convergence analysis ──────────────────────────────────────────
    final_error = abs(rows[-1]["error"]) if rows else initial_error
    threshold = max(initial_error * 0.01, 0.01)  # 1% of initial distance
    converged = final_error <= threshold

    converge_step = None
    for r in rows:
        if abs(r["error"]) <= threshold:
            converge_step = r["step"]
            break

    if converged:
        color, label = "#22c55e", "CONVERGED"
        detail = f"Final error: {final_error:.4f} (within 1% of initial distance {initial_error:.2f})"
        if converge_step:
            detail += f" — converged at step {converge_step}"
    else:
        color, label = "#ef4444", "NOT CONVERGED"
        detail = f"Final error: {final_error:.4f} (threshold: {threshold:.4f})"

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{color}20;border:2px solid {color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f"{label}</span><br>"
        f'<span style="color:#888;font-size:0.9em">{detail}</span></div>'
    )

    # ── Timeline table ────────────────────────────────────────────────
    lines = ["| Step | Value | Error | Correction | Notes |", "| ---: | ---: | ---: | ---: | :--- |"]
    for r in rows:
        lines.append(
            f"| {r['step']} | {r['value']:.4f} | {r['error']:.4f} "
            f"| {r['correction']:.4f} | {r['note']} |"
        )
    timeline_md = "\n".join(lines)

    # ── Analysis ──────────────────────────────────────────────────────
    errors = [abs(r["error"]) for r in rows]
    max_error = max(errors) if errors else 0
    min_error = min(errors) if errors else 0
    overshoots = sum(
        1
        for i in range(1, len(rows))
        if (rows[i]["error"] > 0) != (rows[i - 1]["error"] > 0)
    )

    analysis = f"""### Loop Analysis

| Metric | Value |
| :--- | :--- |
| Setpoint | {setpoint:.2f} |
| Initial value | {initial:.2f} |
| Initial distance | {initial_error:.2f} |
| Final value | {rows[-1]['value']:.4f} if rows else 'N/A' |
| Final error | {final_error:.4f} |
| Max |error| | {max_error:.4f} |
| Min |error| | {min_error:.4f} |
| Zero-crossings | {overshoots} |
| Converge step | {converge_step or 'N/A'} |

### Parameter Guide

- **Gain** ({gain}): Higher gain → faster correction but more oscillation
- **Damping** ({damping}): Higher damping → smoother approach but slower convergence
- **Gain > 0.5 with damping ≈ 0**: Expect oscillation (underdamped)
- **Gain < 0.3 with damping > 0.2**: Expect slow, monotonic approach (overdamped)
"""

    return banner, timeline_md, analysis


# ── Gradio UI ──────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Feedback Loop Homeostasis") as app:
        gr.Markdown(
            "# ⚖️ Feedback Loop Homeostasis\n"
            "Simulate a **NegativeFeedbackLoop** controlling a value toward a "
            "setpoint. Adjust gain, damping, and disturbance to explore "
            "convergence dynamics."
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Temperature control",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Loop", variant="primary", scale=1)

        with gr.Row():
            setpoint_sl = gr.Slider(-100, 200, value=72.0, step=0.5, label="Setpoint")
            initial_sl = gr.Slider(-100, 200, value=85.0, step=0.5, label="Initial value")

        with gr.Row():
            gain_sl = gr.Slider(0.01, 1.0, value=0.3, step=0.01, label="Gain")
            damping_sl = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Damping")
            iter_sl = gr.Slider(10, 50, value=30, step=1, label="Iterations")

        with gr.Row():
            dist_step = gr.Number(value=0, label="Disturbance step (0=none)", precision=0)
            dist_mag = gr.Number(value=0.0, label="Disturbance magnitude")

        banner_html = gr.HTML(label="Convergence")
        with gr.Row():
            with gr.Column(scale=2):
                timeline_md = gr.Markdown(label="Timeline")
            with gr.Column(scale=1):
                analysis_md = gr.Markdown(label="Analysis")

        # ── Event wiring ──────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[setpoint_sl, initial_sl, gain_sl, damping_sl, iter_sl, dist_step, dist_mag],
        )

        run_btn.click(
            fn=run_feedback,
            inputs=[preset_dd, setpoint_sl, initial_sl, gain_sl, damping_sl, iter_sl, dist_step, dist_mag],
            outputs=[banner_html, timeline_md, analysis_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
