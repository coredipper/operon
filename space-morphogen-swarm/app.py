"""
Operon Morphogen-Guided Swarm -- Interactive Gradio Demo
========================================================

Simulate a swarm of workers that adapt strategy based on morphogen
gradient signals. Failed workers update gradients, and successors
read them to avoid repeating mistakes.

Run locally:
    pip install gradio
    python space-morphogen-swarm/app.py

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
    MorphogenType,
    MorphogenGradient,
    GradientOrchestrator,
    RegenerativeSwarm,
    SimpleWorker,
    WorkerMemory,
    SwarmResult,
)
from operon_ai.healing import create_default_summarizer

# ── Trace instrumentation ────────────────────────────────────────────────

_trace_log: list[dict] = []
_gradient_snapshots: list[dict] = []


# ── Presets ──────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own morphogen-guided swarm parameters.",
        "initial_complexity": 0.5,
        "initial_confidence": 0.5,
        "budget_tokens": 1000,
        "max_regenerations": 3,
    },
    "Normal problem solving": {
        "description": "Moderate complexity, decent confidence. Worker solves after brief exploration.",
        "initial_complexity": 0.3,
        "initial_confidence": 0.7,
        "budget_tokens": 2000,
        "max_regenerations": 3,
    },
    "Gradient adaptation": {
        "description": "High complexity causes first worker to fail. Successor reads elevated error_rate and adapts strategy.",
        "initial_complexity": 0.7,
        "initial_confidence": 0.5,
        "budget_tokens": 1500,
        "max_regenerations": 3,
    },
    "Budget exhaustion": {
        "description": "Tight budget forces workers to become concise. Budget morphogen drops rapidly.",
        "initial_complexity": 0.5,
        "initial_confidence": 0.5,
        "budget_tokens": 500,
        "max_regenerations": 2,
    },
    "Complex high-risk": {
        "description": "Very high complexity with low confidence. Multiple regenerations needed, gradient evolves significantly.",
        "initial_complexity": 0.9,
        "initial_confidence": 0.2,
        "budget_tokens": 3000,
        "max_regenerations": 4,
    },
}


def _load_preset(name: str) -> tuple[float, float, int, int]:
    p = PRESETS.get(name, PRESETS["(custom)"])
    return (
        p["initial_complexity"],
        p["initial_confidence"],
        p["budget_tokens"],
        p["max_regenerations"],
    )


# ── Worker factory ───────────────────────────────────────────────────────


def _make_gradient_worker_factory(
    orchestrator: GradientOrchestrator,
    total_budget: int,
):
    """Create a gradient-aware worker factory that logs trace events."""
    tokens_used = [0]
    worker_counter = [0]

    def factory(name: str, memory_hints: list[str]) -> SimpleWorker:
        worker_counter[0] += 1
        generation = worker_counter[0]

        # Read current gradient state
        gradient = orchestrator.gradient
        error_rate = gradient.get(MorphogenType.ERROR_RATE)
        confidence = gradient.get(MorphogenType.CONFIDENCE)
        complexity = gradient.get(MorphogenType.COMPLEXITY)
        budget_ratio = gradient.get(MorphogenType.BUDGET)
        hints = gradient.get_strategy_hints()

        # Record gradient snapshot
        _gradient_snapshots.append({
            "step": generation,
            "complexity": complexity,
            "confidence": confidence,
            "budget": budget_ratio,
            "error_rate": error_rate,
            "hints": list(hints),
        })

        # Determine strategy based on gradient signals
        use_alternative = error_rate >= 0.3 or any(
            "different" in hint.lower() or "error" in hint.lower()
            for hint in memory_hints
        )
        be_concise = budget_ratio < 0.3

        _trace_log.append({
            "worker": name,
            "step": 0,
            "output": (
                f"[{name}] Created: error_rate={error_rate:.2f} "
                f"confidence={confidence:.2f} budget={budget_ratio:.2f} "
                f"strategy={'alternative' if use_alternative else 'default'}"
            ),
            "event": "created",
        })

        factory_ref_tokens = tokens_used

        def work(task: str, memory: WorkerMemory) -> str:
            step = len(memory.output_history)
            tokens_per_step = 50 if be_concise else 100
            factory_ref_tokens[0] += tokens_per_step

            # Update budget morphogen
            remaining = max(0, total_budget - factory_ref_tokens[0])
            orchestrator.gradient.set(
                MorphogenType.BUDGET,
                remaining / total_budget,
            )

            if use_alternative and generation >= 3:
                # Worker with accumulated gradient knowledge
                if step == 0:
                    output = f"[{name}] Reading gradient signals, trying alternative approach"
                    event = "exploring"
                elif step == 1:
                    output = f"[{name}] Alternative approach showing results"
                    event = "exploring"
                else:
                    output = f"[{name}] DONE: Problem solved using gradient-informed strategy!"
                    event = "solved"
            elif use_alternative:
                # First alternative worker, still learning
                if step < 2:
                    output = f"[{name}] Trying modified approach (step {step})"
                    event = "exploring"
                elif step == 2:
                    output = f"[{name}] Modified approach working..."
                    event = "exploring"
                else:
                    output = f"[{name}] FINISHED: Solved with modified approach!"
                    event = "solved"
            else:
                # Default strategy - will get stuck (repeating identical output)
                output = f"[{name}] Analyzing problem..."
                event = "repeated"

            _trace_log.append({
                "worker": name,
                "step": step + 1,
                "output": output,
                "event": event,
            })
            return output

        return SimpleWorker(id=name, work_function=work)

    def report_outcome(success: bool):
        """Report worker outcome and update gradient."""
        tokens_used[0] += 100
        orchestrator.report_step_result(
            success=success,
            tokens_used=tokens_used[0],
            total_budget=total_budget,
        )

    return factory, report_outcome


# ── Core simulation ─────────────────────────────────────────────────────


def run_morphogen_swarm(
    preset_name: str,
    initial_complexity: float,
    initial_confidence: float,
    budget_tokens: int,
    max_regenerations: int,
) -> tuple[str, str, str]:
    """Run the morphogen-guided swarm simulation.

    Returns (result_banner_html, worker_timeline_html, gradient_evolution_md).
    """
    global _trace_log, _gradient_snapshots
    _trace_log = []
    _gradient_snapshots = []

    budget_tokens = int(budget_tokens)
    max_regenerations = int(max_regenerations)

    # Initialize gradient orchestrator
    orchestrator = GradientOrchestrator(silent=True)
    orchestrator.gradient.set(
        MorphogenType.COMPLEXITY, initial_complexity,
        description="Task complexity level",
    )
    orchestrator.gradient.set(
        MorphogenType.CONFIDENCE, initial_confidence,
        description="Solution confidence",
    )
    orchestrator.gradient.set(
        MorphogenType.BUDGET, 1.0,
        description="Budget remaining ratio",
    )

    # Create gradient-aware factory
    factory, report_outcome = _make_gradient_worker_factory(
        orchestrator, budget_tokens,
    )

    # Create swarm
    swarm = RegenerativeSwarm(
        worker_factory=factory,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=max_regenerations,
        silent=True,
    )

    result: SwarmResult = swarm.supervise("Solve the complex optimization problem")

    # Report final outcome
    report_outcome(result.success)

    # Take final gradient snapshot
    gradient = orchestrator.gradient
    _gradient_snapshots.append({
        "step": len(_gradient_snapshots) + 1,
        "complexity": gradient.get(MorphogenType.COMPLEXITY),
        "confidence": gradient.get(MorphogenType.CONFIDENCE),
        "budget": gradient.get(MorphogenType.BUDGET),
        "error_rate": gradient.get(MorphogenType.ERROR_RATE),
        "hints": list(gradient.get_strategy_hints()),
    })

    # ── Result banner ────────────────────────────────────────────────
    if result.success:
        color, label = "#22c55e", "SUCCESS"
        detail = f"Output: {result.output}"
    else:
        color, label = "#ef4444", "FAILURE"
        detail = (
            f"Swarm exhausted {result.total_workers_spawned} workers "
            f"without solving the task."
        )

    final_budget = gradient.get(MorphogenType.BUDGET)
    final_error = gradient.get(MorphogenType.ERROR_RATE)
    final_conf = gradient.get(MorphogenType.CONFIDENCE)

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{color}20;border:2px solid {color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f"{label}</span>"
        f'<span style="color:#888;margin-left:12px">'
        f"Workers: {result.total_workers_spawned} | "
        f"Apoptosis: {len(result.apoptosis_events)} | "
        f"Regenerations: {len(result.regeneration_events)}"
        f"</span><br>"
        f'<span style="font-size:0.9em">{detail}</span><br>'
        f'<span style="font-size:0.85em;color:#666">'
        f"Final gradient: confidence={final_conf:.2f} "
        f"error_rate={final_error:.2f} budget={final_budget:.0%}"
        f"</span></div>"
    )

    # ── Worker timeline ──────────────────────────────────────────────
    event_colors = {
        "created": "#a855f7",
        "solved": "#22c55e",
        "exploring": "#3b82f6",
        "repeated": "#f97316",
    }

    timeline_rows = []
    for entry in _trace_log:
        color = event_colors.get(entry["event"], "#888")
        timeline_rows.append(
            f"<tr>"
            f'<td style="padding:4px 8px;font-family:monospace">{entry["worker"]}</td>'
            f'<td style="padding:4px 8px;text-align:center">{entry["step"]}</td>'
            f'<td style="padding:4px 8px">{entry["output"]}</td>'
            f"<td style=\"padding:4px 8px\">"
            f'<span style="background:{color}20;color:{color};padding:1px 6px;'
            f'border-radius:3px;font-size:0.85em">{entry["event"]}</span></td>'
            f"</tr>"
        )

    if timeline_rows:
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
    else:
        timeline_html = '<p style="color:#888">No trace data captured.</p>'

    # ── Gradient evolution ───────────────────────────────────────────
    if _gradient_snapshots:
        lines = [
            "### Gradient Evolution\n",
            "| Step | Complexity | Confidence | Budget | Error Rate | Hints |",
            "| ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
        for snap in _gradient_snapshots:
            hint_summary = (
                "; ".join(h[:35] for h in snap["hints"][:2])
                if snap["hints"]
                else "none"
            )
            hint_summary = hint_summary.replace("|", "\\|")
            lines.append(
                f'| {snap["step"]} '
                f'| {snap["complexity"]:.2f} '
                f'| {snap["confidence"]:.2f} '
                f'| {snap["budget"]:.2f} '
                f'| {snap["error_rate"]:.2f} '
                f"| {hint_summary} |"
            )

        # Final strategy hints
        final_hints = gradient.get_strategy_hints()
        if final_hints:
            lines.append("\n### Final Strategy Hints\n")
            for hint in final_hints:
                lines.append(f"- {hint}")

        # Summary statistics
        lines.append("\n### Summary\n")
        lines.append(f"- **Workers spawned**: {result.total_workers_spawned}")
        lines.append(f"- **Apoptosis events**: {len(result.apoptosis_events)}")
        lines.append(f"- **Regeneration events**: {len(result.regeneration_events)}")
        lines.append(f"- **Gradient snapshots**: {len(_gradient_snapshots)}")
        lines.append(f"- **Final budget**: {final_budget:.0%}")

        gradient_md = "\n".join(lines)
    else:
        gradient_md = "*No gradient data captured.*"

    return banner, timeline_html, gradient_md


# ── Gradio UI ────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Morphogen-Guided Swarm") as app:
        gr.Markdown(
            "# Morphogen-Guided Swarm\n"
            "Workers adapt strategy based on **morphogen gradient signals**. "
            "Failed workers update gradients, successors read them to avoid "
            "repeating mistakes."
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Gradient adaptation",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Swarm", variant="primary", scale=1)

        with gr.Row():
            complexity_sl = gr.Slider(
                0, 1, value=0.7, step=0.05, label="Initial complexity",
            )
            confidence_sl = gr.Slider(
                0, 1, value=0.5, step=0.05, label="Initial confidence",
            )
            budget_sl = gr.Slider(
                500, 5000, value=1500, step=100, label="Budget (tokens)",
            )
            regens_sl = gr.Slider(
                1, 5, value=3, step=1, label="Max regenerations",
            )

        banner_html = gr.HTML(label="Result")
        gr.Markdown("### Worker Timeline")
        timeline_html = gr.HTML(label="Timeline")
        gradient_md = gr.Markdown(label="Gradient Evolution")

        # ── Event wiring ─────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[complexity_sl, confidence_sl, budget_sl, regens_sl],
        )

        run_btn.click(
            fn=run_morphogen_swarm,
            inputs=[preset_dd, complexity_sl, confidence_sl, budget_sl, regens_sl],
            outputs=[banner_html, timeline_html, gradient_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
