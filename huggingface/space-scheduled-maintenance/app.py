"""
Operon Oscillator-Scheduled Maintenance -- Interactive Gradio Demo
===================================================================

Simulate oscillator-driven maintenance cycles that prune noisy context
via AutophagyDaemon. A NegativeFeedbackLoop adjusts the toxicity
threshold to maintain noise ratio at a configurable setpoint.

Architecture:
  [Oscillator (sin-based)]  -->  phase drives maintenance window
  [AutophagyDaemon]         -->  prunes stale/noisy context
  [NegativeFeedbackLoop]    -->  adjusts toxicity threshold toward setpoint
  [Lysosome]                -->  disposes waste from pruning

CRITICAL: Oscillator phases are computed mathematically (sin-based).
We do NOT call start()/stop() -- no threads are spawned.

Run locally:
    pip install gradio
    python space-scheduled-maintenance/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import math
import sys
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import HistoneStore, Lysosome, Waste, WasteType, NegativeFeedbackLoop
from operon_ai.healing import AutophagyDaemon, create_simple_summarizer

# -- Presets ----------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own maintenance parameters.",
        "frequency": 0.2,
        "noise_rate": 0.4,
        "setpoint": 0.15,
        "feedback_gain": 0.5,
        "num_cycles": 15,
    },
    "Gradual pollution": {
        "description": (
            "Low noise rate accumulates slowly. Maintenance keeps up easily, "
            "feedback loop barely needs to adjust."
        ),
        "frequency": 0.2,
        "noise_rate": 0.3,
        "setpoint": 0.15,
        "feedback_gain": 0.3,
        "num_cycles": 20,
    },
    "Sudden noise spike": {
        "description": (
            "High noise rate overwhelms initial threshold. Watch the feedback "
            "loop tighten the threshold aggressively to compensate."
        ),
        "frequency": 0.25,
        "noise_rate": 0.8,
        "setpoint": 0.10,
        "feedback_gain": 0.7,
        "num_cycles": 20,
    },
    "Feedback correction": {
        "description": (
            "Starts with a very loose threshold (high setpoint). Aggressive gain "
            "ratchets it down as noise accumulates -- demonstrates convergence."
        ),
        "frequency": 0.3,
        "noise_rate": 0.5,
        "setpoint": 0.20,
        "feedback_gain": 0.8,
        "num_cycles": 25,
    },
    "Fast oscillator": {
        "description": (
            "High frequency means more maintenance windows per simulation. "
            "Short cycles keep context clean with minimal feedback adjustment."
        ),
        "frequency": 0.8,
        "noise_rate": 0.5,
        "setpoint": 0.15,
        "feedback_gain": 0.4,
        "num_cycles": 15,
    },
}


def _load_preset(name: str) -> tuple[float, float, float, float, int]:
    """Return slider values for the selected preset."""
    p = PRESETS.get(name, PRESETS["(custom)"])
    return p["frequency"], p["noise_rate"], p["setpoint"], p["feedback_gain"], p["num_cycles"]


# -- Noise helpers ----------------------------------------------------------

USEFUL_LINES = [
    "User asked about Python best practices.",
    "Agent explained PEP 8 style guidelines.",
    "User asked about type hints in Python 3.12.",
    "Agent provided examples of TypeVar and ParamSpec.",
    "User asked about async/await patterns.",
    "Agent explained asyncio event loop.",
]

NOISE_TEMPLATES = [
    "thinking... " * 5,
    "=== " * 10,
    "processing processing processing processing",
    "... ... ... ... ...",
    "analyzing analyzing analyzing analyzing",
    "*** *** *** ***",
]


def _is_noise_line(line: str) -> bool:
    """Heuristic noise detector matching the example logic."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(set(stripped)) <= 2 and len(stripped) > 5:
        return True
    if stripped.lower() in ("...", "---", "***", "===", "thinking..."):
        return True
    words = stripped.split()
    if len(words) > 3 and len(set(words)) == 1:
        return True
    return False


def _noise_ratio(context: str) -> float:
    """Compute fraction of lines that are noise."""
    lines = context.split("\n")
    if not lines:
        return 0.0
    noise_count = sum(1 for l in lines if _is_noise_line(l))
    return noise_count / len(lines)


# -- Oscillator phase (math only, no threads) -------------------------------

def _compute_phase(cycle: int, frequency: float) -> tuple[str, float]:
    """
    Compute oscillator phase from a sine wave.

    Returns (phase_name, amplitude) where amplitude is |sin(2*pi*f*cycle)|.
    """
    t = frequency * cycle
    amplitude = abs(math.sin(2 * math.pi * t))
    frac = (t % 1.0)
    if frac < 0.25:
        phase = "rising"
    elif frac < 0.5:
        phase = "peak"
    elif frac < 0.75:
        phase = "falling"
    else:
        phase = "trough"
    return phase, amplitude


# -- Core simulation -------------------------------------------------------

def run_simulation(
    preset_name: str,
    frequency: float,
    noise_rate: float,
    setpoint: float,
    feedback_gain: float,
    num_cycles: int,
) -> tuple[str, str, str]:
    """Run oscillator-scheduled maintenance simulation.

    Returns (status_banner_html, cycle_timeline_md, analysis_md).
    """
    num_cycles = int(num_cycles)

    # Build components
    histone_store = HistoneStore(silent=True)
    lysosome = Lysosome(silent=True)
    autophagy = AutophagyDaemon(
        histone_store=histone_store,
        lysosome=lysosome,
        summarizer=create_simple_summarizer(),
        toxicity_threshold=0.8,
        silent=True,
    )

    feedback = NegativeFeedbackLoop(
        setpoint=setpoint,
        gain=feedback_gain,
        damping=0.0,
        silent=True,
    )

    # Build initial context
    context_lines = list(USEFUL_LINES)
    context = "\n".join(context_lines)

    toxicity_threshold = 0.5  # Starting threshold

    # Tracking
    timeline_rows: list[dict] = []
    total_tokens_freed = 0
    maintenance_count = 0
    initial_noise = _noise_ratio(context)

    for cycle in range(num_cycles):
        # Add noise based on noise_rate probability
        if (cycle + 1) % max(1, int(1.0 / max(noise_rate, 0.01))) == 0:
            noise_idx = cycle % len(NOISE_TEMPLATES)
            context = context + "\n" + NOISE_TEMPLATES[noise_idx]

        # Compute oscillator phase
        phase, amplitude = _compute_phase(cycle, frequency)

        # Current noise ratio
        nr = _noise_ratio(context)
        context_tokens = len(context) // 4  # rough token estimate

        # Maintenance window: run when amplitude > 0.7 (peak region)
        ran_maintenance = False
        tokens_freed = 0

        if amplitude > 0.7:
            # Run autophagy
            pruned_context, prune_result = autophagy.check_and_prune(
                context, max_tokens=8000,
            )

            if prune_result and prune_result.tokens_freed > 0:
                tokens_freed = prune_result.tokens_freed
                total_tokens_freed += tokens_freed
                context = pruned_context
                ran_maintenance = True
                maintenance_count += 1

            # Dispose waste if noise exceeds threshold
            if nr > toxicity_threshold:
                lysosome.ingest(Waste(
                    waste_type=WasteType.FAILED_OPERATION,
                    content=f"Cycle {cycle}: noise {nr:.0%} > threshold {toxicity_threshold:.0%}",
                    source="maintenance_scheduler",
                ))
                lysosome.digest()
                ran_maintenance = True
                if maintenance_count == 0 or not prune_result:
                    maintenance_count += 1

            # Feedback: adjust toxicity threshold
            post_nr = _noise_ratio(context)
            error = post_nr - setpoint
            adjustment = -error * feedback_gain
            toxicity_threshold = max(0.05, min(0.8, toxicity_threshold + adjustment))

        post_nr = _noise_ratio(context)

        timeline_rows.append({
            "cycle": cycle,
            "phase": phase,
            "amplitude": amplitude,
            "noise_ratio": nr,
            "post_noise": post_nr,
            "context_tokens": context_tokens,
            "maintenance": ran_maintenance,
            "tokens_freed": tokens_freed,
            "threshold": toxicity_threshold,
        })

    # -- Status banner ------------------------------------------------------
    final_noise = _noise_ratio(context)
    final_tokens = len(context) // 4

    if final_noise <= setpoint * 1.5:
        color, label = "#22c55e", "HOMEOSTASIS ACHIEVED"
        detail = (
            f"Final noise ratio {final_noise:.0%} is within target "
            f"(setpoint {setpoint:.0%})"
        )
    elif final_noise <= setpoint * 3.0:
        color, label = "#eab308", "PARTIALLY CONTROLLED"
        detail = (
            f"Final noise ratio {final_noise:.0%} above setpoint "
            f"{setpoint:.0%} but improving"
        )
    else:
        color, label = "#ef4444", "NOISE EXCEEDS TARGET"
        detail = (
            f"Final noise ratio {final_noise:.0%} significantly above "
            f"setpoint {setpoint:.0%}"
        )

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f'background:{color}20;border:2px solid {color};margin-bottom:8px">'
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f'{label}</span><br>'
        f'<span style="color:#888;font-size:0.9em">{detail}</span></div>'
    )

    # -- Cycle timeline table -----------------------------------------------
    lines = [
        "| Cycle | Phase | Amplitude | Noise % | Maintenance | Tokens Freed | Threshold |",
        "| ---: | :--- | ---: | ---: | :--- | ---: | ---: |",
    ]
    for r in timeline_rows:
        maint_icon = "Yes" if r["maintenance"] else "-"
        maint_color = "#22c55e" if r["maintenance"] else "#888"
        lines.append(
            f'| {r["cycle"]} | {r["phase"]} | {r["amplitude"]:.2f} '
            f'| {r["noise_ratio"]:.0%} '
            f'| <span style="color:{maint_color}">{maint_icon}</span> '
            f'| {r["tokens_freed"]} | {r["threshold"]:.3f} |'
        )
    timeline_md = "\n".join(lines)

    # -- Analysis -----------------------------------------------------------
    noise_values = [r["noise_ratio"] for r in timeline_rows]
    threshold_values = [r["threshold"] for r in timeline_rows]
    max_noise = max(noise_values) if noise_values else 0
    min_noise = min(noise_values) if noise_values else 0
    avg_noise = sum(noise_values) / len(noise_values) if noise_values else 0

    analysis = f"""### Simulation Analysis

| Metric | Value |
| :--- | :--- |
| Total cycles | {num_cycles} |
| Oscillator frequency | {frequency} |
| Maintenance runs | {maintenance_count} |
| Total tokens freed | {total_tokens_freed} |
| Initial noise ratio | {initial_noise:.0%} |
| Final noise ratio | {final_noise:.0%} |
| Peak noise ratio | {max_noise:.0%} |
| Min noise ratio | {min_noise:.0%} |
| Avg noise ratio | {avg_noise:.0%} |
| Noise setpoint | {setpoint:.0%} |
| Final toxicity threshold | {toxicity_threshold:.3f} |
| Final context tokens | {final_tokens} |

### How It Works

1. **Oscillator** computes a sine-based phase each cycle (frequency={frequency}).
   Maintenance runs when amplitude > 0.7 (peak region).

2. **Noise injection** adds filler lines at a rate proportional to the noise rate
   slider ({noise_rate}). This simulates context pollution over time.

3. **AutophagyDaemon** prunes noisy context during maintenance windows,
   freeing tokens and reducing noise ratio.

4. **NegativeFeedbackLoop** adjusts the toxicity threshold toward the
   setpoint ({setpoint:.0%}) with gain={feedback_gain}:
   - If noise > setpoint: threshold tightens (prune more aggressively)
   - If noise < setpoint: threshold loosens (prune less)

5. **Lysosome** disposes waste generated when noise exceeds the threshold.

### Parameter Guide

- **Frequency**: Higher = more maintenance windows per simulation
- **Noise rate**: Higher = faster context pollution
- **Setpoint**: Target noise ratio the feedback loop tries to maintain
- **Feedback gain**: How aggressively the threshold adjusts (higher = faster correction)
- **Num cycles**: Total simulation steps
"""

    return banner, timeline_md, analysis


# -- Gradio UI -------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Oscillator-Scheduled Maintenance") as app:
        gr.Markdown(
            "# Oscillator-Scheduled Maintenance\n"
            "Simulate oscillator-driven maintenance cycles that prune noisy "
            "context via **AutophagyDaemon**. A **NegativeFeedbackLoop** adjusts "
            "the toxicity threshold to maintain noise ratio at a configurable setpoint."
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Gradual pollution",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Simulation", variant="primary", scale=1)

        with gr.Row():
            freq_sl = gr.Slider(
                0.05, 1.0, value=0.2, step=0.05,
                label="Oscillator frequency",
            )
            noise_sl = gr.Slider(
                0.0, 1.0, value=0.3, step=0.05,
                label="Noise rate",
            )

        with gr.Row():
            setpoint_sl = gr.Slider(
                0.0, 1.0, value=0.15, step=0.05,
                label="Setpoint (target noise ratio)",
            )
            gain_sl = gr.Slider(
                0.01, 1.0, value=0.3, step=0.01,
                label="Feedback gain",
            )
            cycles_sl = gr.Slider(
                5, 30, value=20, step=1,
                label="Number of cycles",
            )

        banner_html = gr.HTML(label="Status")
        with gr.Row():
            with gr.Column(scale=2):
                timeline_md = gr.Markdown(label="Cycle Timeline")
            with gr.Column(scale=1):
                analysis_md = gr.Markdown(label="Analysis")

        # -- Event wiring ---------------------------------------------------
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[freq_sl, noise_sl, setpoint_sl, gain_sl, cycles_sl],
        )

        run_btn.click(
            fn=run_simulation,
            inputs=[preset_dd, freq_sl, noise_sl, setpoint_sl, gain_sl, cycles_sl],
            outputs=[banner_html, timeline_md, analysis_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
