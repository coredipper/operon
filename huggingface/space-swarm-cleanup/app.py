"""
Operon LLM Swarm with Graceful Cleanup -- Interactive Gradio Demo
=================================================================

Simulate an LLM-powered swarm where dying workers clean up their context
via autophagy before passing state to successors.  Successors inherit a
clean summary instead of raw noise.

Run locally:
    pip install gradio
    python space-swarm-cleanup/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import HistoneStore, Lysosome, Waste, WasteType, MarkerType
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider, ProviderConfig
from operon_ai.coordination.morphogen import MorphogenGradient, MorphogenType
from operon_ai.healing import (
    RegenerativeSwarm,
    SimpleWorker,
    WorkerMemory,
    AutophagyDaemon,
    create_default_summarizer,
    create_simple_summarizer,
)


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class CleanupRecord:
    """Record of a worker's cleanup before death."""
    worker_id: str
    context_before: int   # chars
    context_after: int    # chars
    tokens_freed: int
    summary_stored: str
    noise_disposed: int


# ── LLM Swarm Worker Factory ────────────────────────────────────────────


class LLMSwarmWorkerFactory:
    """
    Factory that creates LLM-powered workers with graceful cleanup.

    Each worker:
    1. Uses Nucleus + MockProvider for "LLM" responses
    2. Accumulates context from responses
    3. Before dying, runs autophagy to clean context
    4. Stores clean summary in HistoneStore for successors
    """

    def __init__(
        self,
        responses: dict[str, str],
        gradient: MorphogenGradient,
        toxicity_threshold: float = 0.6,
    ):
        self.gradient = gradient

        # Shared state across workers
        self.histone_store = HistoneStore()
        self.lysosome = Lysosome(silent=True)
        self.autophagy = AutophagyDaemon(
            histone_store=self.histone_store,
            lysosome=self.lysosome,
            summarizer=create_simple_summarizer(),
            toxicity_threshold=toxicity_threshold,
            silent=True,
        )

        # Nucleus for LLM calls
        self.nucleus = Nucleus(provider=MockProvider(responses=responses))

        # Tracking
        self._cleanup_records: list[CleanupRecord] = []
        self._worker_count = 0
        self._worker_timeline: list[dict] = []

    def create_worker(self, name: str, memory_hints: list[str]) -> SimpleWorker:
        """Create a cleanup-aware worker."""
        self._worker_count += 1
        generation = self._worker_count

        # Check if we have hints from predecessor (via summarizer or histone)
        inherited_context = ""
        if memory_hints:
            retrieval = self.histone_store.retrieve_context(
                " ".join(memory_hints[:3]),
                limit=3,
            )
            if retrieval.formatted_context:
                inherited_context = retrieval.formatted_context
            else:
                inherited_context = "; ".join(memory_hints)

        has_ctx = bool(inherited_context)
        self._worker_timeline.append({
            "worker": name,
            "generation": generation,
            "event": "spawned",
            "detail": "with inherited context" if has_ctx else "fresh start",
        })

        # Build worker context
        accumulated_context: list[str] = []
        if inherited_context:
            accumulated_context.append(
                f"[Inherited summary]: {inherited_context[:200]}"
            )

        factory_ref = self

        def work(task: str, memory: WorkerMemory) -> str:
            step = len(memory.output_history)

            # Simulate LLM response
            prompt_key = f"step_{step}"
            try:
                response = factory_ref.nucleus.transcribe(
                    prompt_key,
                    config=ProviderConfig(temperature=0.0, max_tokens=256),
                )
                output = response.content
            except Exception:
                output = f"Processing step {step}..."

            # Accumulate context
            accumulated_context.append(output)

            # Update gradient
            factory_ref.gradient.set(
                MorphogenType.CONFIDENCE,
                max(0.1, 1.0 - step * 0.15),
            )

            # Workers with inherited context solve faster
            if inherited_context and generation >= 2:
                if step == 0:
                    factory_ref._worker_timeline.append({
                        "worker": name,
                        "generation": generation,
                        "event": "strategy",
                        "detail": f"Starting from inherited summary (gen {generation})",
                    })
                    return f"STRATEGY: Starting from inherited summary (gen {generation})"
                elif step == 1:
                    factory_ref._worker_timeline.append({
                        "worker": name,
                        "generation": generation,
                        "event": "progress",
                        "detail": "Building on predecessor's work",
                    })
                    return "PROGRESS: Building on predecessor's work"
                elif step >= 2:
                    # Run cleanup before returning success
                    factory_ref._cleanup_worker(
                        name, "\n".join(accumulated_context),
                    )
                    factory_ref._worker_timeline.append({
                        "worker": name,
                        "generation": generation,
                        "event": "solved",
                        "detail": "DONE with clean state inheritance",
                    })
                    return "DONE: Completed with clean state inheritance!"

            # Default: accumulate noise, get stuck (identical output)
            factory_ref._worker_timeline.append({
                "worker": name,
                "generation": generation,
                "event": "stuck",
                "detail": "Still processing (identical output)",
            })
            return "THINKING: Still processing..."

        return SimpleWorker(id=name, work_function=work)

    def _cleanup_worker(self, worker_id: str, context: str) -> CleanupRecord:
        """Run graceful cleanup before worker death."""
        context_before = len(context)

        # Run autophagy
        cleaned_context, prune_result = self.autophagy.check_and_prune(
            context, max_tokens=2000,
        )

        tokens_freed = prune_result.tokens_freed if prune_result else 0
        summary = cleaned_context[:300] if cleaned_context else context[:100]

        # Store clean summary in HistoneStore
        self.histone_store.add_marker(
            content=f"Worker {worker_id} summary: {summary}",
            marker_type=MarkerType.ACETYLATION,
            tags=["worker_summary", worker_id],
            context=f"Cleanup from {worker_id} before apoptosis",
        )

        # Dispose noise via Lysosome
        noise_count = 0
        if prune_result and prune_result.tokens_freed > 0:
            self.lysosome.ingest(Waste(
                waste_type=WasteType.EXPIRED_CACHE,
                content=f"Noise from {worker_id}: {tokens_freed} tokens",
                source=worker_id,
            ))
            digest = self.lysosome.digest()
            noise_count = digest.disposed

        record = CleanupRecord(
            worker_id=worker_id,
            context_before=context_before,
            context_after=len(cleaned_context),
            tokens_freed=tokens_freed,
            summary_stored=summary[:100],
            noise_disposed=noise_count,
        )
        self._cleanup_records.append(record)
        return record

    def get_cleanup_records(self) -> list[CleanupRecord]:
        return list(self._cleanup_records)

    def get_worker_timeline(self) -> list[dict]:
        return list(self._worker_timeline)


# ── Presets ──────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own swarm parameters.",
        "entropy_threshold": 0.9,
        "max_steps": 5,
        "max_regenerations": 3,
        "responses": {
            "step_0": "Initial research findings on the topic.",
            "step_1": "Deeper analysis reveals three key factors.",
            "step_2": "Cross-referencing sources confirms hypothesis.",
            "step_3": "Still processing...",
            "step_4": "Still processing...",
        },
    },
    "Research with cleanup": {
        "description": "Worker accumulates noisy context, gets stuck, cleans up, and dies. Successor inherits clean summary and completes the task.",
        "entropy_threshold": 0.9,
        "max_steps": 5,
        "max_regenerations": 3,
        "responses": {
            "step_0": "Initial research findings on the topic.",
            "step_1": "Deeper analysis reveals three key factors.",
            "step_2": "Cross-referencing sources confirms hypothesis.",
            "step_3": "Still processing...",
            "step_4": "Still processing...",
        },
    },
    "Context pollution comparison": {
        "description": "Compare how context cleanup prevents noise from degrading successor performance across generations.",
        "entropy_threshold": 0.9,
        "max_steps": 5,
        "max_regenerations": 3,
        "responses": {
            "step_0": "Finding relevant data...",
            "step_1": "Analyzing patterns in data...",
            "step_2": "Drawing conclusions...",
            "step_3": "Still processing...",
        },
    },
    "Fast cleanup": {
        "description": "Low entropy threshold triggers faster worker turnover. Cleanup keeps context lean across rapid regenerations.",
        "entropy_threshold": 0.6,
        "max_steps": 3,
        "max_regenerations": 3,
        "responses": {
            "step_0": "Quick scan of available data.",
            "step_1": "Preliminary results ready.",
            "step_2": "Done.",
        },
    },
    "Multi-generation": {
        "description": "High regeneration limit allows many worker generations. Each cleans up before dying, building a rich HistoneStore.",
        "entropy_threshold": 0.9,
        "max_steps": 4,
        "max_regenerations": 5,
        "responses": {
            "step_0": "Generation checkpoint: scanning knowledge base.",
            "step_1": "Aggregating findings from prior workers.",
            "step_2": "Synthesizing cross-generation insights.",
            "step_3": "Still processing...",
        },
    },
}


def _load_preset(name: str) -> tuple[float, int, int]:
    p = PRESETS.get(name, PRESETS["(custom)"])
    return p["entropy_threshold"], p["max_steps"], p["max_regenerations"]


# ── Core simulation ─────────────────────────────────────────────────────

_EVENT_COLORS: dict[str, str] = {
    "spawned": "#3b82f6",
    "strategy": "#8b5cf6",
    "progress": "#eab308",
    "solved": "#22c55e",
    "stuck": "#f97316",
}


def run_swarm(
    preset_name: str,
    entropy_threshold: float,
    max_steps: int,
    max_regenerations: int,
) -> tuple[str, str, str, str]:
    """Run the LLM swarm with graceful cleanup simulation.

    Returns (result_banner, worker_timeline_html, cleanup_records_md, gradient_md).
    """
    p = PRESETS.get(preset_name, PRESETS["(custom)"])
    responses = p["responses"]

    gradient = MorphogenGradient()

    factory = LLMSwarmWorkerFactory(
        responses=responses,
        gradient=gradient,
    )

    swarm = RegenerativeSwarm(
        worker_factory=factory.create_worker,
        summarizer=create_default_summarizer(),
        entropy_threshold=entropy_threshold,
        max_steps_per_worker=int(max_steps),
        max_regenerations=int(max_regenerations),
        silent=True,
    )

    result = swarm.supervise("Research the impact of morphogen gradients")

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

    cleanups = factory.get_cleanup_records()
    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{color}20;border:2px solid {color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f"{label}</span>"
        f'<span style="color:#888;margin-left:12px">'
        f"Workers spawned: {result.total_workers_spawned} | "
        f"Cleanups performed: {len(cleanups)}</span><br>"
        f'<span style="font-size:0.9em">{detail}</span></div>'
    )

    # ── Worker timeline HTML table ───────────────────────────────────
    timeline = factory.get_worker_timeline()
    timeline_rows = []
    for entry in timeline:
        ec = _EVENT_COLORS.get(entry["event"], "#888")
        timeline_rows.append(
            f'<tr>'
            f'<td style="padding:4px 8px;font-family:monospace">{entry["worker"]}</td>'
            f'<td style="padding:4px 8px;text-align:center">{entry["generation"]}</td>'
            f'<td style="padding:4px 8px">'
            f'<span style="background:{ec}20;color:{ec};padding:1px 6px;'
            f'border-radius:3px;font-size:0.85em">{entry["event"]}</span></td>'
            f'<td style="padding:4px 8px">{entry["detail"]}</td>'
            f'</tr>'
        )

    if timeline_rows:
        timeline_html = (
            '<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
            '<tr style="background:#f0f0f0">'
            '<th style="padding:6px 8px;text-align:left">Worker</th>'
            '<th style="padding:6px 8px;text-align:center">Gen</th>'
            '<th style="padding:6px 8px;text-align:left">Event</th>'
            '<th style="padding:6px 8px;text-align:left">Detail</th></tr>'
            + "".join(timeline_rows)
            + "</table>"
        )
    else:
        timeline_html = '<p style="color:#888">No timeline data captured.</p>'

    # ── Cleanup records markdown ─────────────────────────────────────
    if cleanups:
        cleanup_lines = ["### Cleanup Records\n"]
        cleanup_lines.append(
            "| Worker | Context Before | Context After | Tokens Freed | Summary Stored |"
        )
        cleanup_lines.append(
            "| :--- | ---: | ---: | ---: | :--- |"
        )
        for rec in cleanups:
            summary_preview = rec.summary_stored[:60]
            summary_preview = summary_preview.replace("|", "\\|")
            cleanup_lines.append(
                f"| `{rec.worker_id}` | {rec.context_before} chars "
                f"| {rec.context_after} chars | {rec.tokens_freed} "
                f"| {summary_preview} |"
            )
        cleanup_lines.append("")
        cleanup_lines.append("### How Cleanup Works\n")
        cleanup_lines.append("1. **AutophagyDaemon** prunes stale/noisy context")
        cleanup_lines.append("2. **Lysosome** disposes of extracted waste")
        cleanup_lines.append("3. **HistoneStore** saves the clean summary for successors")
        cleanup_lines.append(
            "4. Successor workers inherit summaries, not raw noise"
        )
        cleanup_md = "\n".join(cleanup_lines)
    else:
        cleanup_md = (
            "*No cleanup records -- first worker solved the task "
            "without needing regeneration.*"
        )

    # ── Gradient evolution markdown ──────────────────────────────────
    gradient_lines = ["### Morphogen Gradient (Final State)\n"]
    gradient_lines.append("| Signal | Value | Level |")
    gradient_lines.append("| :--- | ---: | :--- |")

    for mtype in [
        MorphogenType.CONFIDENCE,
        MorphogenType.ERROR_RATE,
        MorphogenType.COMPLEXITY,
        MorphogenType.URGENCY,
    ]:
        val = gradient.get(mtype)
        level = gradient.get_level(mtype)

        if mtype == MorphogenType.CONFIDENCE:
            color = "#22c55e" if val > 0.5 else "#ef4444"
        elif mtype == MorphogenType.ERROR_RATE:
            color = "#ef4444" if val > 0.3 else "#22c55e"
        else:
            color = "#888"

        gradient_lines.append(
            f'| {mtype.value} '
            f'| <span style="color:{color}">{val:.3f}</span> '
            f"| {level} |"
        )

    gradient_lines.append("\n### Swarm Statistics\n")
    gradient_lines.append(f"- **Total workers spawned**: {result.total_workers_spawned}")
    gradient_lines.append(f"- **Apoptosis events**: {len(result.apoptosis_events)}")
    gradient_lines.append(f"- **Regeneration events**: {len(result.regeneration_events)}")
    gradient_lines.append(f"- **HistoneStore markers**: stored {len(cleanups)} summaries")

    if result.apoptosis_events:
        gradient_lines.append("\n### Apoptosis Events\n")
        for evt in result.apoptosis_events:
            gradient_lines.append(
                f"- **`{evt.worker_id}`**: {evt.reason.value}"
            )
            if evt.memory_summary:
                for hint in evt.memory_summary:
                    gradient_lines.append(f"  - _{hint}_")

    gradient_md = "\n".join(gradient_lines)

    return banner, timeline_html, cleanup_md, gradient_md


# ── Gradio UI ────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LLM Swarm with Graceful Cleanup") as app:
        gr.Markdown(
            "# 🧹 LLM Swarm with Graceful Cleanup\n"
            "Simulate an LLM-powered swarm where dying workers clean up "
            "context via **autophagy** before passing state to successors.  "
            "Successors inherit a **clean summary** instead of raw noise."
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Research with cleanup",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Swarm", variant="primary", scale=1)

        with gr.Row():
            entropy_sl = gr.Slider(
                0.5, 1.0, value=0.9, step=0.05, label="Entropy threshold"
            )
            steps_sl = gr.Slider(
                3, 10, value=5, step=1, label="Max steps per worker"
            )
            regens_sl = gr.Slider(
                1, 5, value=3, step=1, label="Max regenerations"
            )

        banner_html = gr.HTML(label="Result")
        gr.Markdown("### Worker Timeline")
        timeline_html = gr.HTML(label="Timeline")

        with gr.Row():
            with gr.Column():
                cleanup_md = gr.Markdown(label="Cleanup Records")
            with gr.Column():
                gradient_md = gr.Markdown(label="Gradient Evolution")

        # ── Event wiring ─────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[entropy_sl, steps_sl, regens_sl],
        )

        run_btn.click(
            fn=run_swarm,
            inputs=[preset_dd, entropy_sl, steps_sl, regens_sl],
            outputs=[banner_html, timeline_html, cleanup_md, gradient_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
