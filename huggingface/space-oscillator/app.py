"""
Operon Biological Oscillator Patterns -- Interactive Gradio Demo
================================================================

Two-tab demo: compute and display waveform shapes (sine, square, sawtooth,
triangle, pulse) with damping, plus show biological oscillator phase
structures (Circadian, Heartbeat, CellCycle).

CRITICAL: Oscillators use background threads. We do NOT call start()/stop().
Waveform values are computed mathematically via _compute_waveform_value().

Run locally:
    pip install gradio
    python space-oscillator/app.py

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

from operon_ai import WaveformType

# ── Waveform computation (matches oscillator.py lines 413-433) ────────────


def _compute_waveform_value(waveform: WaveformType, phase: float) -> float:
    """Compute waveform output for a given phase in [0, 1]."""
    if waveform == WaveformType.SINE:
        return math.sin(2 * math.pi * phase)
    elif waveform == WaveformType.SQUARE:
        return 1.0 if phase < 0.5 else -1.0
    elif waveform == WaveformType.SAWTOOTH:
        return 2.0 * phase - 1.0
    elif waveform == WaveformType.TRIANGLE:
        return 4.0 * phase - 1.0 if phase < 0.5 else 3.0 - 4.0 * phase
    elif waveform == WaveformType.PULSE:
        return 1.0 if phase < 0.1 else 0.0
    return 0.0


# ── Tab 1: Waveform Presets ───────────────────────────────────────────────

WAVEFORM_PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own waveform parameters.",
        "waveform": "sine",
        "frequency": 1.0,
        "amplitude": 1.0,
        "damping": 0.0,
        "cycles": 2,
    },
    "Sine wave": {
        "description": "Classic sinusoidal oscillation — the fundamental waveform.",
        "waveform": "sine",
        "frequency": 1.0,
        "amplitude": 1.0,
        "damping": 0.0,
        "cycles": 2,
    },
    "Square pulse": {
        "description": "Binary on/off switching at 2 Hz — digital clock behavior.",
        "waveform": "square",
        "frequency": 2.0,
        "amplitude": 1.0,
        "damping": 0.0,
        "cycles": 3,
    },
    "Damped sine": {
        "description": "Sine wave with exponential decay — models energy dissipation.",
        "waveform": "sine",
        "frequency": 1.0,
        "amplitude": 1.0,
        "damping": 0.1,
        "cycles": 5,
    },
    "Fast triangle": {
        "description": "High-frequency triangle wave — linear ramps up and down.",
        "waveform": "triangle",
        "frequency": 5.0,
        "amplitude": 1.0,
        "damping": 0.0,
        "cycles": 3,
    },
    "Sawtooth": {
        "description": "Linear ramp from -1 to +1 — used in synthesis and scanning.",
        "waveform": "sawtooth",
        "frequency": 1.0,
        "amplitude": 1.0,
        "damping": 0.0,
        "cycles": 3,
    },
}


def _load_waveform_preset(name: str) -> tuple[str, float, float, float, int]:
    p = WAVEFORM_PRESETS.get(name, WAVEFORM_PRESETS["(custom)"])
    return p["waveform"], p["frequency"], p["amplitude"], p["damping"], p["cycles"]


# ── Tab 2: Biological Oscillator Presets ──────────────────────────────────

BIO_PRESETS: dict[str, dict] = {
    "Standard circadian": {
        "description": "16-hour day / 8-hour night cycle — governs sleep-wake patterns.",
        "type": "circadian",
        "day_hours": 16.0,
        "night_hours": 8.0,
    },
    "Fast heartbeat": {
        "description": "120 BPM heartbeat — elevated heart rate during exercise.",
        "type": "heartbeat",
        "bpm": 120.0,
    },
    "Cell division": {
        "description": "24-hour cell cycle: G1 (40%), S (30%), G2 (20%), M (10%).",
        "type": "cell_cycle",
        "cycle_hours": 24.0,
        "phases": {"G1": 0.4, "S": 0.3, "G2": 0.2, "M": 0.1},
    },
}


# ── Tab 1: Waveform computation ──────────────────────────────────────────


def run_waveform(
    preset_name: str,
    waveform_name: str,
    frequency: float,
    amplitude: float,
    damping: float,
    cycles: int,
) -> tuple[str, str, str]:
    """Compute waveform samples.

    Returns (config_banner_html, sample_table_md, stats_md).
    """
    waveform = WaveformType(waveform_name)
    cycles = int(cycles)

    # Config banner
    banner = (
        f'<div style="padding:10px 14px;border-radius:8px;background:#f0f9ff;'
        f'border:1px solid #bae6fd;margin-bottom:8px">'
        f'<span style="font-weight:700;font-size:1.1em">{waveform.value.upper()}</span> '
        f'<span style="color:#666"> | freq={frequency} Hz | amp={amplitude} | '
        f'damping={damping} | cycles={cycles}</span></div>'
    )

    # Compute samples (50 samples per cycle)
    samples_per_cycle = 50
    total_samples = samples_per_cycle * cycles
    period = 1.0 / frequency if frequency > 0 else 1.0

    rows = [
        "| # | Time (s) | Phase | Cycle | Amplitude | Value |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    values = []
    for i in range(total_samples + 1):
        t = i * (cycles * period) / total_samples
        # Phase within current cycle
        cycle_time = t / period
        current_cycle = int(cycle_time)
        phase = cycle_time - current_cycle
        if phase < 0:
            phase = 0.0
        if current_cycle >= cycles:
            current_cycle = cycles - 1
            phase = 1.0

        # Apply damping: amplitude decays exponentially
        damped_amp = amplitude * math.exp(-damping * t) if damping > 0 else amplitude
        raw = _compute_waveform_value(waveform, phase)
        value = raw * damped_amp
        values.append(value)

        # Show every 5th sample to keep table manageable
        if i % 5 == 0 or i == total_samples:
            rows.append(
                f"| {i} | {t:.4f} | {phase:.3f} | {current_cycle + 1} "
                f"| {damped_amp:.4f} | {value:.4f} |"
            )

    table_md = "\n".join(rows)

    # Stats
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    zero_crossings = sum(
        1 for i in range(1, len(values))
        if (values[i] >= 0) != (values[i - 1] >= 0)
    )

    stats = f"""### Waveform Statistics

| Metric | Value |
| :--- | :--- |
| Samples computed | {len(values)} |
| Min value | {min_val:.4f} |
| Max value | {max_val:.4f} |
| Mean value | {mean_val:.4f} |
| Zero crossings | {zero_crossings} |
| Period | {period:.4f} s |
| Total duration | {cycles * period:.4f} s |

### Waveform Guide

- **SINE**: Smooth continuous oscillation. Most natural biological rhythm.
- **SQUARE**: Binary switching. Models on/off states (gene expression toggle).
- **SAWTOOTH**: Linear ramp with sharp reset. Models gradual buildup + sudden release.
- **TRIANGLE**: Symmetric linear ramp. Models gradual charge/discharge cycles.
- **PULSE**: Brief spike. Models action potentials and trigger signals.
"""

    return banner, table_md, stats


# ── Tab 2: Biological oscillator display ─────────────────────────────────

PHASE_COLORS = {
    "Day": "#fbbf24",
    "Night": "#1e40af",
    "Systole": "#ef4444",
    "Diastole": "#3b82f6",
    "G1": "#22c55e",
    "S": "#3b82f6",
    "G2": "#a855f7",
    "M": "#ef4444",
}


def run_bio_oscillator(preset_name: str) -> tuple[str, str, str]:
    """Display biological oscillator phase structure.

    Returns (type_banner_html, phase_table_md, explanation_md).
    """
    p = BIO_PRESETS.get(preset_name)
    if not p:
        return "<p>Select a biological oscillator preset.</p>", "", ""

    osc_type = p["type"]

    if osc_type == "circadian":
        day_h = p["day_hours"]
        night_h = p["night_hours"]
        total = day_h + night_h
        banner = (
            f'<div style="padding:10px 14px;border-radius:8px;background:#fffbeb;'
            f'border:1px solid #fde68a">'
            f'<span style="font-weight:700;font-size:1.1em">CIRCADIAN OSCILLATOR</span> '
            f'<span style="color:#666"> | {total}h cycle = {day_h}h day + {night_h}h night</span></div>'
        )

        phases = [
            ("Day", day_h, day_h / total),
            ("Night", night_h, night_h / total),
        ]

        explanation = f"""### Circadian Rhythm

The circadian oscillator governs the sleep-wake cycle across a {total}-hour period.

- **Day phase** ({day_h}h, {day_h / total * 100:.0f}%): Active metabolism, high gene expression for repair enzymes, elevated cortisol
- **Night phase** ({night_h}h, {night_h / total * 100:.0f}%): Reduced metabolism, memory consolidation, growth hormone release

In agent systems, circadian oscillators can gate when expensive operations are allowed (day = active processing, night = background maintenance).
"""

    elif osc_type == "heartbeat":
        bpm = p["bpm"]
        period_ms = 60000 / bpm
        systole_ms = period_ms * 0.35
        diastole_ms = period_ms * 0.65
        banner = (
            f'<div style="padding:10px 14px;border-radius:8px;background:#fef2f2;'
            f'border:1px solid #fecaca">'
            f'<span style="font-weight:700;font-size:1.1em">HEARTBEAT OSCILLATOR</span> '
            f'<span style="color:#666"> | {bpm} BPM | period={period_ms:.0f}ms</span></div>'
        )

        phases = [
            ("Systole", systole_ms / 1000, 0.35),
            ("Diastole", diastole_ms / 1000, 0.65),
        ]

        explanation = f"""### Heartbeat Rhythm

The heartbeat oscillator at {bpm} BPM has a period of {period_ms:.0f}ms.

- **Systole** ({systole_ms:.0f}ms, 35%): Contraction phase — pumping output. In agents: burst processing, sending responses.
- **Diastole** ({diastole_ms:.0f}ms, 65%): Relaxation phase — refilling. In agents: gathering input, buffering requests.

At {bpm} BPM, the system processes {bpm} pulse cycles per minute, each with a work burst followed by a collection period.
"""

    else:  # cell_cycle
        cycle_h = p["cycle_hours"]
        phase_pcts = p["phases"]
        banner = (
            f'<div style="padding:10px 14px;border-radius:8px;background:#f0fdf4;'
            f'border:1px solid #bbf7d0">'
            f'<span style="font-weight:700;font-size:1.1em">CELL CYCLE OSCILLATOR</span> '
            f'<span style="color:#666"> | {cycle_h}h total cycle</span></div>'
        )

        phases = [
            (name, cycle_h * pct, pct)
            for name, pct in phase_pcts.items()
        ]

        explanation = f"""### Cell Division Cycle

The cell cycle oscillator spans {cycle_h} hours with four distinct phases:

- **G1** ({cycle_h * 0.4:.1f}h, 40%): Gap 1 — cell growth, organelle duplication. In agents: planning and resource allocation.
- **S** ({cycle_h * 0.3:.1f}h, 30%): Synthesis — DNA replication. In agents: core work/computation phase.
- **G2** ({cycle_h * 0.2:.1f}h, 20%): Gap 2 — error checking, preparation for division. In agents: validation and testing.
- **M** ({cycle_h * 0.1:.1f}h, 10%): Mitosis — actual division. In agents: spawning sub-agents or forking work.

Checkpoints between phases ensure quality: G1/S checkpoint (commit to replication?), G2/M checkpoint (ready to divide?).
"""

    # Phase breakdown table
    table_lines = [
        "| Phase | Duration | Fraction | Visual |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for name, duration, fraction in phases:
        color = PHASE_COLORS.get(name, "#888")
        bar_width = max(5, int(fraction * 300))
        duration_str = f"{duration:.2f}h" if duration >= 0.01 else f"{duration * 3600:.0f}s"
        table_lines.append(
            f"| **{name}** | {duration_str} | {fraction * 100:.0f}% | "
            f'<span style="display:inline-block;width:{bar_width}px;height:16px;'
            f'background:{color};border-radius:3px"></span> |'
        )
    table_md = "\n".join(table_lines)

    return banner, table_md, explanation


# ── Gradio UI ──────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Biological Oscillators") as app:
        gr.Markdown(
            "# 🔃 Biological Oscillator Patterns\n"
            "Explore waveform shapes and biological oscillator phase "
            "structures — computed mathematically, no threads needed."
        )

        with gr.Tabs():
            # ── Tab 1: Waveform Explorer ──────────────────────────────
            with gr.TabItem("Waveform Explorer"):
                with gr.Row():
                    wave_preset_dd = gr.Dropdown(
                        choices=list(WAVEFORM_PRESETS.keys()),
                        value="Sine wave",
                        label="Preset",
                        scale=2,
                    )
                    wave_btn = gr.Button("Compute Waveform", variant="primary", scale=1)

                with gr.Row():
                    waveform_dd = gr.Dropdown(
                        choices=[w.value for w in WaveformType],
                        value="sine",
                        label="Waveform",
                    )
                    freq_sl = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Frequency (Hz)")
                    amp_sl = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Amplitude")

                with gr.Row():
                    damp_sl = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="Damping factor")
                    cycles_sl = gr.Slider(1, 10, value=2, step=1, label="Cycles")

                wave_banner = gr.HTML(label="Configuration")
                with gr.Row():
                    with gr.Column(scale=2):
                        wave_table = gr.Markdown(label="Samples")
                    with gr.Column(scale=1):
                        wave_stats = gr.Markdown(label="Statistics")

                wave_preset_dd.change(
                    fn=_load_waveform_preset,
                    inputs=[wave_preset_dd],
                    outputs=[waveform_dd, freq_sl, amp_sl, damp_sl, cycles_sl],
                )

                wave_btn.click(
                    fn=run_waveform,
                    inputs=[wave_preset_dd, waveform_dd, freq_sl, amp_sl, damp_sl, cycles_sl],
                    outputs=[wave_banner, wave_table, wave_stats],
                )

            # ── Tab 2: Biological Oscillators ─────────────────────────
            with gr.TabItem("Biological Oscillators"):
                with gr.Row():
                    bio_preset_dd = gr.Dropdown(
                        choices=list(BIO_PRESETS.keys()),
                        value="Standard circadian",
                        label="Biological Oscillator",
                        scale=2,
                    )
                    bio_btn = gr.Button("Show Phases", variant="primary", scale=1)

                bio_banner = gr.HTML(label="Oscillator Type")
                bio_table = gr.Markdown(label="Phase Breakdown")
                bio_explanation = gr.Markdown(label="Explanation")

                bio_btn.click(
                    fn=run_bio_oscillator,
                    inputs=[bio_preset_dd],
                    outputs=[bio_banner, bio_table, bio_explanation],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
