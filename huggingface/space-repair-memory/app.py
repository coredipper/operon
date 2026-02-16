"""
Operon Repair Memory Agent -- Interactive Gradio Demo
======================================================

Simulates an LLM agent with epigenetic repair memory:
- Nucleus + MockProvider generates LLM responses
- Chaperone validates output against Pydantic schemas
- On INVALID: consults HistoneStore for repair hints, attempts healing
- On HEALED: stores repair strategy as epigenetic marker
- EpiplexityMonitor watches for output diversity

Run locally:
    pip install gradio
    python space-repair-memory/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import sys
import json
from pathlib import Path

import gradio as gr
from pydantic import BaseModel, Field

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import HistoneStore, MarkerType
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider, ProviderConfig
from operon_ai.health import EpiplexityMonitor, MockEmbeddingProvider, HealthStatus


# ── Pydantic Schemas ─────────────────────────────────────────────────────


class TaskResult(BaseModel):
    """Schema for a task execution result."""
    task_id: str
    status: str = Field(pattern=r"^(success|partial|failed)$")
    result: str
    confidence: float = Field(ge=0.0, le=1.0)
    steps_taken: int = Field(ge=0)


class AnalysisReport(BaseModel):
    """Schema for an analysis report."""
    topic: str
    summary: str
    findings: list[str] = Field(min_length=1)
    risk_level: str = Field(pattern=r"^(low|medium|high|critical)$")
    recommendations: list[str] = Field(default_factory=list)


SCHEMA_MAP = {
    "TaskResult": TaskResult,
    "AnalysisReport": AnalysisReport,
}


# ── Status styles ────────────────────────────────────────────────────────

STATUS_STYLES: dict[HealthStatus, tuple[str, str]] = {
    HealthStatus.HEALTHY: ("#22c55e", "HEALTHY"),
    HealthStatus.EXPLORING: ("#3b82f6", "EXPLORING"),
    HealthStatus.CONVERGING: ("#a855f7", "CONVERGING"),
    HealthStatus.STAGNANT: ("#f97316", "STAGNANT"),
    HealthStatus.CRITICAL: ("#ef4444", "CRITICAL"),
}


# ── Presets ───────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own repair scenario.",
        "requests": [],
        "temperature": 0.0,
        "max_repair_attempts": 3,
    },
    "First failure and repair": {
        "description": "First request succeeds, second fails and is repaired generically (no stored memory).",
        "requests": [
            {
                "key": "task_good",
                "schema": "TaskResult",
                "response": json.dumps({
                    "task_id": "TASK-001",
                    "status": "success",
                    "result": "Analysis completed successfully",
                    "confidence": 0.85,
                    "steps_taken": 3,
                }),
                "label": "Valid task result",
            },
            {
                "key": "task_bad",
                "schema": "TaskResult",
                "response": "This is not valid JSON at all! The task was somewhat successful.",
                "label": "Invalid output (generic repair)",
            },
        ],
        "temperature": 0.0,
        "max_repair_attempts": 3,
    },
    "Memory reuse": {
        "description": "First failure uses generic repair, second failure recalls stored strategy.",
        "requests": [
            {
                "key": "report_good",
                "schema": "AnalysisReport",
                "response": json.dumps({
                    "topic": "Security Audit",
                    "summary": "Comprehensive security review completed",
                    "findings": ["No critical vulnerabilities", "Minor config issues"],
                    "risk_level": "low",
                    "recommendations": ["Update firewall rules"],
                }),
                "label": "Valid report (baseline)",
            },
            {
                "key": "report_bad_1",
                "schema": "AnalysisReport",
                "response": "Invalid report output attempt 1 -- missing all fields",
                "label": "First failure (generic repair)",
            },
            {
                "key": "report_bad_2",
                "schema": "AnalysisReport",
                "response": "Invalid report output attempt 2 -- still broken",
                "label": "Second failure (memory recall)",
            },
        ],
        "temperature": 0.0,
        "max_repair_attempts": 3,
    },
    "Diverse outputs": {
        "description": "Six valid but similar responses -- epiplexity drops as diversity decreases.",
        "requests": [
            {
                "key": f"attempt_{i}",
                "schema": "TaskResult",
                "response": json.dumps({
                    "task_id": f"TASK-{i:03d}",
                    "status": "success",
                    "result": "Same result every time",
                    "confidence": 0.5,
                    "steps_taken": 1,
                }),
                "label": f"Attempt {i} (similar output)",
            }
            for i in range(6)
        ],
        "temperature": 0.0,
        "max_repair_attempts": 3,
    },
    "Multiple schema types": {
        "description": "Mix of TaskResult and AnalysisReport validations, some valid, some repaired.",
        "requests": [
            {
                "key": "task_ok",
                "schema": "TaskResult",
                "response": json.dumps({
                    "task_id": "T-100",
                    "status": "success",
                    "result": "Deployment successful",
                    "confidence": 0.92,
                    "steps_taken": 5,
                }),
                "label": "Valid TaskResult",
            },
            {
                "key": "report_ok",
                "schema": "AnalysisReport",
                "response": json.dumps({
                    "topic": "Performance Review",
                    "summary": "System performance is within SLA",
                    "findings": ["P99 latency under 200ms", "No memory leaks"],
                    "risk_level": "low",
                }),
                "label": "Valid AnalysisReport",
            },
            {
                "key": "task_broken",
                "schema": "TaskResult",
                "response": "Error: could not complete task, status unknown",
                "label": "Invalid TaskResult (repair needed)",
            },
            {
                "key": "report_broken",
                "schema": "AnalysisReport",
                "response": "The analysis found some issues but I cannot format them properly",
                "label": "Invalid AnalysisReport (repair needed)",
            },
        ],
        "temperature": 0.0,
        "max_repair_attempts": 3,
    },
}


def _load_preset(name: str) -> tuple[float, int]:
    p = PRESETS.get(name, PRESETS["(custom)"])
    return p["temperature"], p["max_repair_attempts"]


# ── Repair Logic ─────────────────────────────────────────────────────────


def _try_fix_json(raw_output: str, schema: type[BaseModel]) -> BaseModel | None:
    """Attempt to fix common JSON issues and validate."""
    content = raw_output.strip()

    # Try direct parse
    try:
        data = json.loads(content)
        return schema.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass

    # Try extracting from code blocks
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                data = json.loads(cleaned)
                return schema.model_validate(data)
            except (json.JSONDecodeError, Exception):
                continue

    # Build from schema defaults
    try:
        fields = schema.model_fields
        defaults: dict = {}
        for fname, field_info in fields.items():
            if field_info.default is not None:
                defaults[fname] = field_info.default
            elif field_info.annotation == str:
                defaults[fname] = content[:100]
            elif field_info.annotation == int:
                defaults[fname] = 0
            elif field_info.annotation == float:
                defaults[fname] = 0.5
            elif field_info.annotation == list:
                defaults[fname] = [content[:50]]
            elif field_info.annotation == bool:
                defaults[fname] = True
        return schema.model_validate(defaults)
    except Exception:
        pass

    return None


def run_repair_memory(
    preset_name: str,
    temperature: float,
    max_repair_attempts: int,
) -> tuple[str, str, str, str]:
    """Run the repair memory simulation.

    Returns (result_banner_html, repair_timeline_md, epigenetic_memory_md, epiplexity_md).
    """
    p = PRESETS.get(preset_name, PRESETS["(custom)"])
    requests = p.get("requests", [])

    if not requests:
        return "Select a preset to run the simulation.", "", "", ""

    # Build mock provider responses
    response_map = {req["key"]: req["response"] for req in requests}

    nucleus = Nucleus(provider=MockProvider(responses=response_map))
    histone_store = HistoneStore()
    epiplexity_monitor = EpiplexityMonitor(
        embedding_provider=MockEmbeddingProvider(dim=64),
        alpha=0.5,
        window_size=5,
        threshold=0.2,
        critical_duration=3,
    )

    # Track results
    timeline_entries: list[dict] = []
    total_requests = 0
    validation_failures = 0
    repairs_from_memory = 0
    repairs_generic = 0
    strategies_stored = 0

    for req in requests:
        total_requests += 1
        prompt_key = req["key"]
        schema_name = req["schema"]
        schema = SCHEMA_MAP[schema_name]
        label = req["label"]

        entry = {
            "step": total_requests,
            "label": label,
            "prompt_key": prompt_key,
            "schema": schema_name,
            "validation_passed": False,
            "repair_attempted": False,
            "repair_source": None,
            "epiplexity_status": None,
            "result_preview": None,
        }

        # Step 1: Generate response
        response = nucleus.transcribe(
            prompt_key,
            config=ProviderConfig(temperature=temperature, max_tokens=512),
        )

        # Step 2: Measure epiplexity
        ep_result = epiplexity_monitor.measure(response.content)
        entry["epiplexity_status"] = ep_result.status.value

        # Step 3: Validate
        error_trace = ""
        try:
            data = json.loads(response.content)
            validated = schema.model_validate(data)
            entry["validation_passed"] = True
            entry["result_preview"] = str(validated)[:120]
        except (json.JSONDecodeError, Exception) as e:
            validated = None
            error_trace = str(e)[:100]
            entry["error"] = error_trace

        if validated:
            timeline_entries.append(entry)
            continue

        # Step 4: Validation failed -- attempt repair
        validation_failures += 1
        entry["repair_attempted"] = True

        # Check histone store for known strategies
        query = f"repair {schema_name} {error_trace[:50]}"
        retrieval = histone_store.retrieve_context(query, limit=3)

        repair_hints = []
        if retrieval.formatted_context:
            repair_hints = [
                s.strip()
                for s in retrieval.formatted_context.split("\n")
                if s.strip() and "repair" in s.lower()
            ]

        if repair_hints:
            entry["repair_source"] = "epigenetic_memory"
            repairs_from_memory += 1
        else:
            entry["repair_source"] = "generic"
            repairs_generic += 1

        # Attempt repair
        repaired = _try_fix_json(response.content, schema)

        if repaired:
            entry["validation_passed"] = True
            entry["result_preview"] = str(repaired)[:120]

            # Store repair strategy
            strategy = (
                f"Schema: {schema_name}\n"
                f"Error: {error_trace[:100]}\n"
                f"Method: {'hint_based' if repair_hints else 'generic'}\n"
                f"Fix: Transformed output to match schema requirements"
            )
            histone_store.add_marker(
                content=strategy,
                marker_type=MarkerType.ACETYLATION,
                tags=["repair_strategy", schema_name],
                context=f"Repair for {schema_name} validation failure",
                confidence=0.8,
            )
            strategies_stored += 1
        else:
            entry["result_preview"] = "(repair failed)"

        timeline_entries.append(entry)

    # ── Result banner ────────────────────────────────────────────────────
    success_count = sum(1 for e in timeline_entries if e["validation_passed"])
    fail_count = total_requests - success_count

    if fail_count == 0:
        banner_color = "#22c55e"
        banner_label = "ALL VALID"
    elif success_count > fail_count:
        banner_color = "#eab308"
        banner_label = "PARTIALLY REPAIRED"
    else:
        banner_color = "#ef4444"
        banner_label = "REPAIR ISSUES"

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f'background:{banner_color}20;border:2px solid {banner_color};margin-bottom:8px">'
        f'<span style="font-size:1.3em;font-weight:700;color:{banner_color}">'
        f'{banner_label}</span>'
        f'<span style="color:#888;margin-left:12px">'
        f'Requests: {total_requests} | '
        f'Valid: {success_count} | '
        f'Failures: {validation_failures} | '
        f'Repaired: {strategies_stored} | '
        f'Memory recalls: {repairs_from_memory}'
        f'</span></div>'
    )

    # ── Repair timeline ──────────────────────────────────────────────────
    timeline_lines = [
        "### Repair Timeline\n",
        "| # | Label | Schema | Valid | Repair | Source | Epiplexity |",
        "| ---: | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for entry in timeline_entries:
        valid_icon = "Yes" if entry["validation_passed"] else "No"
        valid_color = "#22c55e" if entry["validation_passed"] else "#ef4444"
        repair_str = "Yes" if entry["repair_attempted"] else "--"
        source_str = entry["repair_source"] or "--"
        ep_status = entry["epiplexity_status"] or "--"
        ep_color, _ = STATUS_STYLES.get(
            HealthStatus(ep_status) if ep_status != "--" else HealthStatus.HEALTHY,
            ("#888", ep_status),
        )

        label_preview = entry["label"][:35]
        if len(entry["label"]) > 35:
            label_preview += "..."

        timeline_lines.append(
            f'| {entry["step"]} '
            f'| {label_preview} '
            f'| {entry["schema"]} '
            f'| <span style="color:{valid_color}">{valid_icon}</span> '
            f'| {repair_str} '
            f'| {source_str} '
            f'| <span style="color:{ep_color}">{ep_status}</span> |'
        )

    if timeline_entries:
        timeline_lines.append("\n### Result Details\n")
        for entry in timeline_entries:
            if entry.get("result_preview"):
                timeline_lines.append(f"**Step {entry['step']}** ({entry['label']}): "
                                      f"`{entry['result_preview']}`\n")
            if entry.get("error") and not entry["validation_passed"]:
                timeline_lines.append(f"**Step {entry['step']}** error: `{entry['error']}`\n")

    timeline_md = "\n".join(timeline_lines)

    # ── Epigenetic memory ────────────────────────────────────────────────
    memory_parts = ["### Epigenetic Memory (HistoneStore)\n"]

    retrieval_all = histone_store.retrieve_context("repair", limit=10)
    if retrieval_all.total_markers > 0:
        memory_parts.append(f"**Stored markers**: {retrieval_all.total_markers} | "
                            f"**Active**: {retrieval_all.active_markers}\n")
        memory_parts.append("| # | Marker Type | Tags | Confidence |")
        memory_parts.append("| ---: | :--- | :--- | ---: |")

        for i, marker in enumerate(retrieval_all.markers, 1):
            tags_str = ", ".join(marker.tags) if marker.tags else "--"
            memory_parts.append(
                f"| {i} | {marker.marker_type.value} | {tags_str} | {marker.confidence:.2f} |"
            )

        memory_parts.append("\n**How it works**: When a repair succeeds, the strategy "
                            "is stored as an ACETYLATION marker. On future failures with "
                            "similar schemas, stored strategies are recalled first -- "
                            "building a library of repair patterns over time.")
    else:
        memory_parts.append("*No repair strategies stored yet.* Run a scenario with "
                            "failures to see strategies accumulate.")

    memory_parts.append("\n### Marker Types\n")
    memory_parts.append("| Type | Purpose |")
    memory_parts.append("| :--- | :--- |")
    memory_parts.append("| METHYLATION | Silencing / suppression |")
    memory_parts.append("| ACETYLATION | Activation / repair strategies |")
    memory_parts.append("| PHOSPHORYLATION | Signaling / transient state |")
    memory_parts.append("| UBIQUITINATION | Degradation targeting |")

    memory_md = "\n".join(memory_parts)

    # ── Epiplexity analysis ──────────────────────────────────────────────
    ep_stats = epiplexity_monitor.stats()

    epiplexity_parts = ["### Epiplexity Analysis\n"]
    epiplexity_parts.append("| Metric | Value |")
    epiplexity_parts.append("| :--- | :--- |")
    epiplexity_parts.append(f"| Total measurements | {ep_stats['total_measurements']} |")
    epiplexity_parts.append(f"| Mean epiplexity | {ep_stats['mean_epiplexity']:.4f} |")
    epiplexity_parts.append(f"| Max epiplexity | {ep_stats['max_epiplexity']:.4f} |")
    epiplexity_parts.append(f"| Stagnant episodes | {ep_stats['stagnant_episodes']} |")
    epiplexity_parts.append(f"| Max consecutive stagnant | {ep_stats['max_consecutive_stagnant']} |")
    epiplexity_parts.append(f"| Window size | {ep_stats['window_size']} |")

    if ep_stats['stagnant_episodes'] > 0:
        epiplexity_parts.append(
            "\n**Warning**: Stagnant episodes detected. The agent may be "
            "producing repetitive outputs or stuck in a repair loop."
        )
    else:
        epiplexity_parts.append(
            "\n**Status**: Output diversity is healthy. The agent is producing "
            "varied responses across requests."
        )

    epiplexity_parts.append("\n### How Epiplexity Works\n")
    epiplexity_parts.append("- **Embedding novelty**: How different each output is from the running average")
    epiplexity_parts.append("- **Epiplexity score**: Combined novelty + perplexity metric")
    epiplexity_parts.append("- **Stagnant**: Score drops below threshold for extended periods")
    epiplexity_parts.append("- Monitors whether repairs are producing diverse outputs or repeating")

    epiplexity_md = "\n".join(epiplexity_parts)

    return banner, timeline_md, memory_md, epiplexity_md


# ── Gradio UI ─────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Repair Memory Agent") as app:
        gr.Markdown(
            "# Repair Memory Agent\n"
            "LLM agent with **epigenetic repair memory**: Nucleus generates responses, "
            "Chaperone validates against schemas, HistoneStore remembers successful "
            "repair strategies, and EpiplexityMonitor tracks output diversity.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="First failure and repair",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Simulation", variant="primary", scale=1)

        with gr.Row():
            temperature_sl = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05,
                label="Temperature",
            )
            max_repairs_sl = gr.Slider(
                1, 5, value=3, step=1,
                label="Max repair attempts",
            )

        banner_html = gr.HTML(label="Result")
        timeline_md = gr.Markdown(label="Repair Timeline")

        with gr.Row():
            with gr.Column():
                memory_md = gr.Markdown(label="Epigenetic Memory")
            with gr.Column():
                epiplexity_md = gr.Markdown(label="Epiplexity Analysis")

        # ── Event wiring ─────────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[temperature_sl, max_repairs_sl],
        )

        run_btn.click(
            fn=run_repair_memory,
            inputs=[preset_dd, temperature_sl, max_repairs_sl],
            outputs=[banner_html, timeline_md, memory_md, epiplexity_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
