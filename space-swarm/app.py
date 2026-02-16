"""
Operon Regenerative Swarm -- Interactive Gradio Demo
====================================================

Simulate a swarm of workers that detect entropy collapse, self-terminate
(apoptosis), and respawn with summarized learnings from predecessors.

Run locally:
    pip install gradio
    python space-swarm/app.py

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
    RegenerativeSwarm,
    SwarmResult,
    SimpleWorker,
    WorkerMemory,
)
from operon_ai.healing import create_default_summarizer

# ── Trace instrumentation ─────────────────────────────────────────────────

_trace_log: list[dict] = []


def _make_worker_factory(behavior: str):
    """Create a worker factory with the specified behavior pattern.

    Each factory returns a SimpleWorker whose work_function follows
    a scripted behavior for demonstration purposes.
    """
    worker_counter = [0]

    def factory(task: str, hints: list[str]) -> SimpleWorker:
        worker_counter[0] += 1
        worker_id = f"worker-{worker_counter[0]}"
        step_counter = [0]

        def work(t: str, memory: WorkerMemory) -> str:
            step_counter[0] += 1
            step = step_counter[0]

            if behavior == "stuck":
                # Worker repeats same output → entropy collapse
                # If hints were injected (successor), solve it
                if hints:
                    output = f"[{worker_id}] SOLVED using hint: {hints[0]}"
                    _trace_log.append({
                        "worker": worker_id, "step": step,
                        "output": output, "event": "solved_with_hint",
                    })
                    return output
                output = f"[{worker_id}] Trying approach A..."
                _trace_log.append({
                    "worker": worker_id, "step": step,
                    "output": output, "event": "repeated",
                })
                return output

            elif behavior == "quick":
                output = f"[{worker_id}] SOLVED: Solution found immediately!"
                _trace_log.append({
                    "worker": worker_id, "step": step,
                    "output": output, "event": "solved",
                })
                return output

            elif behavior == "gradual":
                if step < 4:
                    output = f"[{worker_id}] Exploring angle {step}..."
                    _trace_log.append({
                        "worker": worker_id, "step": step,
                        "output": output, "event": "exploring",
                    })
                    return output
                output = f"[{worker_id}] SOLVED: Converged on solution at step {step}!"
                _trace_log.append({
                    "worker": worker_id, "step": step,
                    "output": output, "event": "solved",
                })
                return output

            elif behavior == "error_prone":
                if hints:
                    output = f"[{worker_id}] SOLVED: Fixed with predecessor knowledge"
                    _trace_log.append({
                        "worker": worker_id, "step": step,
                        "output": output, "event": "solved_with_hint",
                    })
                    return output
                if step % 2 == 0:
                    # Return error string (don't raise — let entropy collapse handle it)
                    output = f"[{worker_id}] ERROR at step {step}: computation failed"
                    _trace_log.append({
                        "worker": worker_id, "step": step,
                        "output": output, "event": "error",
                    })
                    return output
                output = f"[{worker_id}] Partial progress step {step}"
                _trace_log.append({
                    "worker": worker_id, "step": step,
                    "output": output, "event": "partial",
                })
                return output

            else:  # always_stuck
                output = f"[{worker_id}] Still stuck..."
                _trace_log.append({
                    "worker": worker_id, "step": step,
                    "output": output, "event": "repeated",
                })
                return output

        return SimpleWorker(id=worker_id, work_function=work)

    return factory


# ── Presets ────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own swarm parameters.",
        "behavior": "stuck",
        "entropy_threshold": 0.9,
        "max_steps": 10,
        "max_regenerations": 3,
    },
    "Stuck worker (entropy collapse)": {
        "description": "Worker repeats same output → entropy collapse → apoptosis → successor with hints solves it.",
        "behavior": "stuck",
        "entropy_threshold": 0.9,
        "max_steps": 10,
        "max_regenerations": 3,
    },
    "Quick solver": {
        "description": "Worker finds the solution immediately on the first step.",
        "behavior": "quick",
        "entropy_threshold": 0.9,
        "max_steps": 10,
        "max_regenerations": 3,
    },
    "Gradual convergence": {
        "description": "Worker explores multiple angles before converging after 4-5 steps.",
        "behavior": "gradual",
        "entropy_threshold": 0.9,
        "max_steps": 10,
        "max_regenerations": 3,
    },
    "Error-prone recovery": {
        "description": "Errors accumulate → worker killed → next worker succeeds with predecessor knowledge.",
        "behavior": "error_prone",
        "entropy_threshold": 0.9,
        "max_steps": 10,
        "max_regenerations": 3,
    },
    "Max regenerations exhausted": {
        "description": "All workers get stuck — regeneration limit reached, swarm fails.",
        "behavior": "always_stuck",
        "entropy_threshold": 0.9,
        "max_steps": 5,
        "max_regenerations": 2,
    },
}


def _load_preset(name: str) -> tuple[float, int, int]:
    p = PRESETS.get(name, PRESETS["(custom)"])
    return p["entropy_threshold"], p["max_steps"], p["max_regenerations"]


# ── Core simulation ───────────────────────────────────────────────────────

# Apoptosis reason styling
_REASON_COLORS: dict[str, str] = {
    "entropy_collapse": "#f97316",
    "error_accumulation": "#ef4444",
    "timeout": "#eab308",
    "resource_exhaustion": "#a855f7",
    "manual": "#6b7280",
}


def run_swarm(
    preset_name: str,
    entropy_threshold: float,
    max_steps: int,
    max_regenerations: int,
) -> tuple[str, str, str, str]:
    """Run the regenerative swarm simulation.

    Returns (result_banner, worker_timeline_html, apoptosis_md, regeneration_md).
    """
    global _trace_log
    _trace_log = []

    p = PRESETS.get(preset_name, PRESETS["(custom)"])
    behavior = p["behavior"]

    factory = _make_worker_factory(behavior)
    swarm = RegenerativeSwarm(
        worker_factory=factory,
        summarizer=create_default_summarizer(),
        entropy_threshold=entropy_threshold,
        max_steps_per_worker=int(max_steps),
        max_regenerations=int(max_regenerations),
        silent=True,
    )

    result: SwarmResult = swarm.supervise("Solve the task")

    # ── Result banner ─────────────────────────────────────────────────
    if result.success:
        color, label = "#22c55e", "SUCCESS"
        detail = f"Output: {result.output}"
    else:
        color, label = "#ef4444", "FAILURE"
        detail = f"Swarm exhausted {result.total_workers_spawned} workers without solving the task."

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{color}20;border:2px solid {color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f"{label}</span>"
        f'<span style="color:#888;margin-left:12px">'
        f"Workers spawned: {result.total_workers_spawned}</span><br>"
        f'<span style="font-size:0.9em">{detail}</span></div>'
    )

    # ── Worker timeline (from trace log) ──────────────────────────────
    event_colors = {
        "solved": "#22c55e",
        "solved_with_hint": "#10b981",
        "exploring": "#3b82f6",
        "partial": "#eab308",
        "repeated": "#f97316",
        "error": "#ef4444",
    }

    timeline_rows = []
    for entry in _trace_log:
        color = event_colors.get(entry["event"], "#888")
        timeline_rows.append(
            f'<tr>'
            f'<td style="padding:4px 8px;font-family:monospace">{entry["worker"]}</td>'
            f'<td style="padding:4px 8px;text-align:center">{entry["step"]}</td>'
            f'<td style="padding:4px 8px">{entry["output"]}</td>'
            f'<td style="padding:4px 8px">'
            f'<span style="background:{color}20;color:{color};padding:1px 6px;'
            f'border-radius:3px;font-size:0.85em">{entry["event"]}</span></td>'
            f'</tr>'
        )

    timeline_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
        '<tr style="background:#f0f0f0">'
        '<th style="padding:6px 8px;text-align:left">Worker</th>'
        '<th style="padding:6px 8px;text-align:center">Step</th>'
        '<th style="padding:6px 8px;text-align:left">Output</th>'
        '<th style="padding:6px 8px;text-align:left">Event</th></tr>'
        + "".join(timeline_rows)
        + "</table>"
    )

    if not timeline_rows:
        timeline_html = '<p style="color:#888">No trace data captured.</p>'

    # ── Apoptosis events ──────────────────────────────────────────────
    if result.apoptosis_events:
        apo_lines = ["### Apoptosis Events\n"]
        for evt in result.apoptosis_events:
            reason_color = _REASON_COLORS.get(evt.reason.value, "#888")
            apo_lines.append(
                f"**Worker `{evt.worker_id}`** — "
                f'<span style="color:{reason_color}">{evt.reason.value}</span>\n'
            )
            if evt.memory_summary:
                apo_lines.append("Memory summary passed to successor:")
                for hint in evt.memory_summary:
                    apo_lines.append(f"- {hint}")
            if evt.details:
                apo_lines.append(f"\n> {evt.details}")
            apo_lines.append("")
        apoptosis_md = "\n".join(apo_lines)
    else:
        apoptosis_md = "*No apoptosis events — worker solved it without getting stuck.*"

    # ── Regeneration events ───────────────────────────────────────────
    if result.regeneration_events:
        regen_lines = ["### Regeneration Events\n"]
        for evt in result.regeneration_events:
            regen_lines.append(
                f"**`{evt.old_worker_id}`** → **`{evt.new_worker_id}`**\n"
            )
            if evt.injected_summary:
                regen_lines.append("Injected hints:")
                for hint in evt.injected_summary:
                    regen_lines.append(f"- _{hint}_")
            regen_lines.append("")
        regeneration_md = "\n".join(regen_lines)
    else:
        regeneration_md = "*No regeneration events — first worker succeeded.*"

    return banner, timeline_html, apoptosis_md, regeneration_md


# ── Gradio UI ──────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Regenerative Swarm") as app:
        gr.Markdown(
            "# 🧬 Regenerative Swarm\n"
            "Simulate workers that detect entropy collapse, self-terminate "
            "(apoptosis), and respawn with summarized learnings."
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Stuck worker (entropy collapse)",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Swarm", variant="primary", scale=1)

        with gr.Row():
            entropy_sl = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Entropy threshold")
            steps_sl = gr.Slider(1, 20, value=10, step=1, label="Max steps per worker")
            regens_sl = gr.Slider(1, 5, value=3, step=1, label="Max regenerations")

        banner_html = gr.HTML(label="Result")
        gr.Markdown("### Worker Timeline")
        timeline_html = gr.HTML(label="Timeline")

        with gr.Row():
            with gr.Column():
                apoptosis_md = gr.Markdown(label="Apoptosis Events")
            with gr.Column():
                regeneration_md = gr.Markdown(label="Regeneration Events")

        # ── Event wiring ──────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[entropy_sl, steps_sl, regens_sl],
        )

        run_btn.click(
            fn=run_swarm,
            inputs=[preset_dd, entropy_sl, steps_sl, regens_sl],
            outputs=[banner_html, timeline_html, apoptosis_md, regeneration_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
