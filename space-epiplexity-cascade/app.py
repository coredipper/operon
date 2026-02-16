"""
Operon Epiplexity Healing Cascade -- Interactive Gradio Demo
============================================================

Simulate an agent that detects epistemic stagnation via the
EpiplexityMonitor and escalates through increasingly aggressive
healing interventions: autophagy, regeneration, and abort.

Run locally:
    pip install gradio
    python space-epiplexity-cascade/app.py

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

from operon_ai import HistoneStore, Lysosome
from operon_ai.health import EpiplexityMonitor, MockEmbeddingProvider, HealthStatus
from operon_ai.healing import (
    AutophagyDaemon,
    RegenerativeSwarm,
    SimpleWorker,
    WorkerMemory,
    create_default_summarizer,
    create_simple_summarizer,
)

# ── Status styling ───────────────────────────────────────────────────────

STATUS_STYLES: dict[HealthStatus, tuple[str, str]] = {
    HealthStatus.HEALTHY: ("#22c55e", "HEALTHY"),
    HealthStatus.EXPLORING: ("#3b82f6", "EXPLORING"),
    HealthStatus.CONVERGING: ("#a855f7", "CONVERGING"),
    HealthStatus.STAGNANT: ("#f97316", "STAGNANT"),
    HealthStatus.CRITICAL: ("#ef4444", "CRITICAL"),
}


def _status_badge(status: HealthStatus) -> str:
    color, label = STATUS_STYLES.get(status, ("#888", str(status)))
    return (
        f'<span style="background:{color}20;color:{color};padding:2px 8px;'
        f'border-radius:4px;font-weight:600;border:1px solid {color}">'
        f"{label}</span>"
    )


# ── Presets ──────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Enter your own messages (one per line).",
        "messages": [],
        "window_size": 5,
        "threshold": 0.2,
        "max_messages": 15,
    },
    "Healthy diverse agent": {
        "description": "Diverse analytical messages stay HEALTHY throughout. No interventions triggered.",
        "messages": [
            "First, let me analyze the requirements.",
            "The key constraint is memory efficiency.",
            "I'll use a hash map for O(1) lookups.",
            "Testing edge cases: empty input, large input.",
            "Implementation complete. Here are the results.",
            "Performance benchmarks show 2x improvement.",
        ],
        "window_size": 5,
        "threshold": 0.2,
        "max_messages": 15,
    },
    "Stagnant repetitive agent": {
        "description": "Messages become repetitive, triggering Stage 1: autophagy context pruning.",
        "messages": [
            "Let me think about this problem.",
            "I need to consider the constraints.",
            "Hmm, let me think about this problem.",
            "I need to consider the constraints.",
            "Hmm, let me think about this problem.",
            "I need to consider the constraints.",
            "Let me try a completely different approach.",
            "Using dynamic programming instead.",
        ],
        "window_size": 5,
        "threshold": 0.2,
        "max_messages": 15,
    },
    "Critical deeply stuck agent": {
        "description": "Identical repeated output triggers all three stages: autophagy, regeneration, and abort.",
        "messages": [
            "Processing request.",
            "Processing request.",
            "Processing request.",
            "Processing request.",
            "Processing request.",
            "Processing request.",
            "Processing request.",
            "Processing request.",
            "Processing request.",
            "Processing request.",
        ],
        "window_size": 5,
        "threshold": 0.2,
        "max_messages": 15,
    },
    "Recovery after stagnation": {
        "description": "Agent gets stuck, then breaks out with a fresh approach. Shows stagnation detection followed by recovery.",
        "messages": [
            "Let me try to optimize the database query.",
            "The query is slow because of the join.",
            "The join is slow.",
            "Still looking at the slow join.",
            "The join is the bottleneck.",
            "Wait, let me try a completely different approach!",
            "Instead of optimizing the join, I'll denormalize the data.",
            "Created a materialized view for the dashboard metrics.",
            "The materialized view refreshes every 5 minutes.",
            "Query time dropped from 3s to 50ms with the new approach.",
        ],
        "window_size": 5,
        "threshold": 0.2,
        "max_messages": 15,
    },
}


def _load_preset(name: str) -> tuple[str, int, float, int]:
    p = PRESETS.get(name, PRESETS["(custom)"])
    messages_text = "\n".join(p["messages"]) if p["messages"] else ""
    return messages_text, p["window_size"], p["threshold"], p["max_messages"]


# ── Intervention logic ───────────────────────────────────────────────────

INTERVENTION_STYLES = {
    "none": ("#22c55e", "NONE"),
    "autophagy": ("#f97316", "AUTOPHAGY"),
    "regeneration": ("#a855f7", "REGENERATION"),
    "abort": ("#ef4444", "ABORT"),
}


def _run_regeneration(failed_messages: list[str], silent: bool = True) -> str | None:
    """Attempt recovery via a RegenerativeSwarm."""
    summary = "; ".join(failed_messages[-3:])[:200]

    def create_recovery_worker(name: str, hints: list[str]) -> SimpleWorker:
        has_context = bool(hints)

        def work(task: str, memory: WorkerMemory) -> str:
            step = len(memory.output_history)
            if has_context and step >= 1:
                return "DONE: Recovered from stagnation with fresh approach!"
            return f"RECOVERY: Analyzing previous failures (step {step})"

        return SimpleWorker(id=name, work_function=work)

    swarm = RegenerativeSwarm(
        worker_factory=create_recovery_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=0.9,
        max_steps_per_worker=5,
        max_regenerations=1,
        silent=silent,
    )

    result = swarm.supervise(f"Recover from stagnation. Context: {summary}")
    if result.success:
        return result.output
    return None


# ── Core simulation ─────────────────────────────────────────────────────


def run_epiplexity_cascade(
    preset_name: str,
    custom_messages: str,
    window_size: int,
    threshold: float,
    max_messages: int,
) -> tuple[str, str, str, str]:
    """Run the epiplexity healing cascade simulation.

    Returns (status_banner_html, intervention_timeline_md,
             epiplexity_history_md, diagnostic_report_md).
    """
    # Parse messages
    messages = [
        line.strip()
        for line in custom_messages.strip().split("\n")
        if line.strip()
    ]

    if not messages:
        empty = "Enter messages (one per line) to analyze."
        return empty, "", "", ""

    window_size = int(window_size)
    max_messages = int(max_messages)
    messages = messages[:max_messages]

    # Set up monitoring
    monitor = EpiplexityMonitor(
        embedding_provider=MockEmbeddingProvider(dim=64),
        alpha=0.5,
        window_size=window_size,
        threshold=threshold,
        critical_duration=3,
    )

    # Set up autophagy (Stage 1)
    histone_store = HistoneStore()
    lysosome = Lysosome(silent=True)
    autophagy = AutophagyDaemon(
        histone_store=histone_store,
        lysosome=lysosome,
        summarizer=create_simple_summarizer(),
        toxicity_threshold=0.5,
        silent=True,
    )

    # Tracking
    measurements: list[dict] = []
    interventions: list[dict] = []
    current_intervention = "none"
    stagnant_count = 0
    critical_count = 0
    context_pruned = False
    worker_regenerated = False
    output_lines: list[str] = []
    final_success = True

    for i, message in enumerate(messages):
        result = monitor.measure(message)
        status = result.status

        measurements.append({
            "index": i + 1,
            "message": message,
            "epiplexity": result.epiplexity,
            "novelty": result.embedding_novelty,
            "perplexity": result.normalized_perplexity,
            "integral": result.epiplexic_integral,
            "status": status,
        })

        if status in (HealthStatus.HEALTHY, HealthStatus.EXPLORING, HealthStatus.CONVERGING):
            output_lines.append(message)
            stagnant_count = 0
            continue

        if status == HealthStatus.STAGNANT:
            stagnant_count += 1

            if stagnant_count == 1 and current_intervention == "none":
                # Stage 1: Autophagy
                context = "\n".join(messages[:i]) or message
                pruned_context, prune_result = autophagy.check_and_prune(
                    context, max_tokens=4000,
                )
                context_pruned = prune_result is not None
                current_intervention = "autophagy"

                detail = (
                    f"Pruned {prune_result.tokens_freed} tokens"
                    if prune_result
                    else "Context assessed, no pruning needed"
                )
                interventions.append({
                    "stage": 1,
                    "name": "Autophagy",
                    "message_index": i + 1,
                    "status": status.value,
                    "detail": detail,
                })

                if context_pruned and prune_result:
                    output_lines.append(
                        f"[Context pruned: {prune_result.tokens_freed} tokens freed]"
                    )
                continue

        if status == HealthStatus.CRITICAL:
            critical_count += 1

            if current_intervention in ("none", "autophagy"):
                # Stage 2: Regeneration
                regen_output = _run_regeneration(messages[:i])
                worker_regenerated = regen_output is not None
                current_intervention = "regeneration"

                interventions.append({
                    "stage": 2,
                    "name": "Regeneration",
                    "message_index": i + 1,
                    "status": status.value,
                    "detail": (
                        f"Recovery output: {regen_output[:80]}"
                        if regen_output
                        else "Regeneration attempted"
                    ),
                })

                if regen_output:
                    output_lines.append(f"[Regenerated: {regen_output}]")
                    # Regeneration succeeded; remaining messages are post-recovery
                    continue

            elif current_intervention == "regeneration":
                # Stage 3: Abort
                current_intervention = "abort"
                final_success = False

                interventions.append({
                    "stage": 3,
                    "name": "Abort",
                    "message_index": i + 1,
                    "status": status.value,
                    "detail": "All interventions exhausted. Aborting.",
                })
                break

    # ── Final status banner ──────────────────────────────────────────
    final_status = measurements[-1]["status"] if measurements else HealthStatus.HEALTHY
    inv_color, inv_label = INTERVENTION_STYLES.get(
        current_intervention, ("#888", "UNKNOWN")
    )

    if final_success:
        banner_color, banner_label = "#22c55e", "COMPLETED"
    else:
        banner_color, banner_label = "#ef4444", "ABORTED"

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{banner_color}20;border:2px solid {banner_color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{banner_color}">'
        f"{banner_label}</span>"
        f'<span style="color:#888;margin-left:12px">'
        f"Messages: {len(measurements)} | "
        f"Final status: {final_status.value} | "
        f'Highest intervention: <span style="color:{inv_color};font-weight:600">'
        f"{inv_label}</span></span><br>"
        f'<span style="font-size:0.85em;color:#666">'
        f"Stagnant messages: {stagnant_count} | "
        f"Critical messages: {critical_count} | "
        f"Context pruned: {'yes' if context_pruned else 'no'} | "
        f"Worker regenerated: {'yes' if worker_regenerated else 'no'}"
        f"</span></div>"
    )

    # ── Intervention timeline ────────────────────────────────────────
    if interventions:
        stage_colors = {1: "#f97316", 2: "#a855f7", 3: "#ef4444"}
        lines = [
            "### Intervention Timeline\n",
            "| Stage | Intervention | At Message | Status | Detail |",
            "| :---: | :--- | :---: | :--- | :--- |",
        ]
        for inv in interventions:
            s_color = stage_colors.get(inv["stage"], "#888")
            detail = inv["detail"].replace("|", "\\|")
            lines.append(
                f'| <span style="color:{s_color};font-weight:700">'
                f'{inv["stage"]}</span> '
                f'| {inv["name"]} '
                f'| {inv["message_index"]} '
                f"| {inv['status']} "
                f"| {detail} |"
            )
        intervention_md = "\n".join(lines)
    else:
        intervention_md = (
            "### Intervention Timeline\n\n"
            "*No interventions triggered -- agent stayed healthy throughout.*"
        )

    # ── Epiplexity history ───────────────────────────────────────────
    if measurements:
        lines = [
            "### Epiplexity History\n",
            "| # | Message | Novelty | Perplexity | Epiplexity | Integral | Status |",
            "| ---: | :--- | ---: | ---: | ---: | ---: | :--- |",
        ]
        for m in measurements:
            preview = m["message"][:40] + "..." if len(m["message"]) > 40 else m["message"]
            preview = preview.replace("|", "\\|")
            lines.append(
                f'| {m["index"]} '
                f"| {preview} "
                f'| {m["novelty"]:.3f} '
                f'| {m["perplexity"]:.3f} '
                f'| {m["epiplexity"]:.3f} '
                f'| {m["integral"]:.3f} '
                f"| {_status_badge(m['status'])} |"
            )
        epiplexity_md = "\n".join(lines)
    else:
        epiplexity_md = "*No measurements recorded.*"

    # ── Diagnostic report ────────────────────────────────────────────
    epiplexities = [m["epiplexity"] for m in measurements]
    novelties = [m["novelty"] for m in measurements]

    status_counts: dict[str, int] = {}
    for m in measurements:
        s = m["status"].value
        status_counts[s] = status_counts.get(s, 0) + 1

    status_breakdown = " | ".join(
        f"**{k}**: {v}" for k, v in status_counts.items()
    )

    transitions = sum(
        1
        for j in range(1, len(measurements))
        if measurements[j]["status"] != measurements[j - 1]["status"]
    )

    diag_lines = ["### Diagnostic Report\n"]
    diag_lines.append("| Metric | Value |")
    diag_lines.append("| :--- | :--- |")
    diag_lines.append(f"| Total messages | {len(measurements)} |")
    diag_lines.append(f"| Stagnant messages | {stagnant_count} |")
    diag_lines.append(f"| Critical messages | {critical_count} |")
    diag_lines.append(f"| Interventions applied | {len(interventions)} |")
    diag_lines.append(f"| Context pruned | {'Yes' if context_pruned else 'No'} |")
    diag_lines.append(f"| Worker regenerated | {'Yes' if worker_regenerated else 'No'} |")
    diag_lines.append(f"| Status transitions | {transitions} |")

    if epiplexities:
        diag_lines.append(f"| Mean epiplexity | {sum(epiplexities) / len(epiplexities):.4f} |")
        diag_lines.append(f"| Min epiplexity | {min(epiplexities):.4f} |")
        diag_lines.append(f"| Max epiplexity | {max(epiplexities):.4f} |")
        diag_lines.append(f"| Mean novelty | {sum(novelties) / len(novelties):.4f} |")
        diag_lines.append(f"| Final integral | {measurements[-1]['integral']:.4f} |")

    diag_lines.append(f"\n**Status distribution**: {status_breakdown}")

    diag_lines.append("\n### Cascade Stages Explained\n")
    diag_lines.append("| Stage | Intervention | Trigger | Action |")
    diag_lines.append("| :---: | :--- | :--- | :--- |")
    diag_lines.append(
        "| 1 | Autophagy | STAGNANT detected | "
        "Prune stale context to break the loop |"
    )
    diag_lines.append(
        "| 2 | Regeneration | CRITICAL detected | "
        "Kill stuck worker, spawn fresh one with summary |"
    )
    diag_lines.append(
        "| 3 | Abort | Still CRITICAL after regeneration | "
        "Give up with diagnostic report |"
    )

    diagnostic_md = "\n".join(diag_lines)

    return banner, intervention_md, epiplexity_md, diagnostic_md


# ── Gradio UI ────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Epiplexity Healing Cascade") as app:
        gr.Markdown(
            "# Epiplexity Healing Cascade\n"
            "Detect epistemic stagnation via the **EpiplexityMonitor** and "
            "watch escalating interventions: autophagy, regeneration, abort."
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Stagnant repetitive agent",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Cascade", variant="primary", scale=1)

        messages_tb = gr.Textbox(
            lines=8,
            label="Messages (one per line)",
            placeholder="Enter agent messages here, one per line...",
        )

        with gr.Row():
            window_sl = gr.Slider(
                3, 10, value=5, step=1, label="Window size",
            )
            thresh_sl = gr.Slider(
                0.05, 0.5, value=0.2, step=0.01, label="Stagnation threshold",
            )
            max_msg_sl = gr.Slider(
                5, 20, value=15, step=1, label="Max messages",
            )

        banner_html = gr.HTML(label="Status")
        intervention_md = gr.Markdown(label="Intervention Timeline")
        epiplexity_md = gr.Markdown(label="Epiplexity History")
        diagnostic_md = gr.Markdown(label="Diagnostic Report")

        # ── Event wiring ─────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[messages_tb, window_sl, thresh_sl, max_msg_sl],
        )

        run_btn.click(
            fn=run_epiplexity_cascade,
            inputs=[preset_dd, messages_tb, window_sl, thresh_sl, max_msg_sl],
            outputs=[banner_html, intervention_md, epiplexity_md, diagnostic_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
