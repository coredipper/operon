"""
Operon Chaperone Healing Loop & Autophagy -- Interactive Gradio Demo
====================================================================

Two-tab demo:
  Tab 1: Mock LLM generates invalid output, Chaperone validates,
         error feedback triggers refolding.
  Tab 2: Autophagy daemon monitors context window fullness and
         prunes when critical.

Run locally:
    pip install gradio
    python space-healing/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import sys
from pathlib import Path

import gradio as gr
from pydantic import BaseModel

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import Chaperone, HistoneStore, Lysosome
from operon_ai.healing import (
    ChaperoneLoop,
    HealingResult,
    HealingOutcome,
    AutophagyDaemon,
    ContextHealthStatus,
    create_simple_summarizer,
    create_mock_healing_generator,
)


# ── Schema for healing demo ───────────────────────────────────────────────


class PriceQuote(BaseModel):
    product: str
    price: float
    currency: str


# ── Tab 1: Healing Loop Presets ───────────────────────────────────────────

OUTCOME_STYLES: dict[HealingOutcome, tuple[str, str]] = {
    HealingOutcome.VALID_FIRST_TRY: ("#22c55e", "VALID FIRST TRY"),
    HealingOutcome.HEALED: ("#eab308", "HEALED"),
    HealingOutcome.DEGRADED: ("#ef4444", "DEGRADED"),
}

HEALING_PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own healing scenario.",
        "initial_output": '{"product": "Widget", "price": 9.99, "currency": "USD"}',
        "healed_output": '{"product": "Widget", "price": 9.99, "currency": "USD"}',
        "heal_on_error_containing": "never_match",
        "max_retries": 3,
    },
    "Valid first try": {
        "description": "Generator produces correct JSON immediately — no refolding needed.",
        "initial_output": '{"product": "Laptop", "price": 999.99, "currency": "USD"}',
        "healed_output": '{"product": "Laptop", "price": 999.99, "currency": "USD"}',
        "heal_on_error_containing": "never_match",
        "max_retries": 3,
    },
    "Healed after retry": {
        "description": "Initial output has wrong type → error feedback triggers correct output.",
        "initial_output": '{"product": "Phone", "price": "expensive", "currency": "USD"}',
        "healed_output": '{"product": "Phone", "price": 699.99, "currency": "USD"}',
        "heal_on_error_containing": "strategies failed",
        "max_retries": 3,
    },
    "Degraded (all retries fail)": {
        "description": "Trigger never matches error → all retries fail → ubiquitin tagged.",
        "initial_output": '{"product": "Gadget", "price": "free", "currency": 123}',
        "healed_output": '{"product": "Gadget", "price": 49.99, "currency": "USD"}',
        "heal_on_error_containing": "xyz_never_matches",
        "max_retries": 3,
    },
    "Complex schema healing": {
        "description": "Multiple fields wrong → error feedback triggers refolding → all fixed.",
        "initial_output": '{"product": 42, "price": "none", "currency": true}',
        "healed_output": '{"product": "Sensor", "price": 29.99, "currency": "EUR"}',
        "heal_on_error_containing": "strategies failed",
        "max_retries": 5,
    },
}


def _load_healing_preset(name: str) -> int:
    p = HEALING_PRESETS.get(name, HEALING_PRESETS["(custom)"])
    return p["max_retries"]


# ── Tab 2: Autophagy Presets ──────────────────────────────────────────────

HEALTH_STYLES: dict[ContextHealthStatus, tuple[str, str]] = {
    ContextHealthStatus.HEALTHY: ("#22c55e", "HEALTHY"),
    ContextHealthStatus.ACCUMULATING: ("#eab308", "ACCUMULATING"),
    ContextHealthStatus.CRITICAL: ("#ef4444", "CRITICAL"),
    ContextHealthStatus.PRUNING: ("#a855f7", "PRUNING"),
}

AUTOPHAGY_PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Enter your own context text.",
        "context": "",
        "max_tokens": 500,
        "force": False,
    },
    "Healthy context": {
        "description": "Short text, well under limit — health check returns HEALTHY.",
        "context": (
            "The user asked about Python list comprehensions. "
            "I explained the syntax and gave two examples. "
            "They understood and moved on to dictionary comprehensions."
        ),
        "max_tokens": 500,
        "force": False,
    },
    "Critical context": {
        "description": "Long noisy text at 90%+ fill → auto prune triggers.",
        "context": (
            "Step 1: The user asked about deployment. "
            "Step 2: I explained Docker basics. "
            "Step 3: We discussed Kubernetes pods. "
            "Step 4: Reviewed the Dockerfile line by line. "
            "Step 5: Debugged a port mapping issue. "
            "Step 6: Fixed the nginx configuration. "
            "Step 7: Set up health checks. "
            "Step 8: Configured environment variables. "
            "Step 9: Added volume mounts for persistence. "
            "Step 10: Tested the deployment locally. "
            "Step 11: Pushed to the container registry. "
            "Step 12: Created the Kubernetes deployment manifest. "
            "Step 13: Applied the manifest to the cluster. "
            "Step 14: Verified the pods are running. "
            "Step 15: Set up the ingress controller. "
            "Step 16: Configured TLS certificates. "
            "Step 17: Tested external access. "
            "Step 18: Added monitoring with Prometheus. "
            "Step 19: Created Grafana dashboards. "
            "Step 20: Set up alerting rules. "
            "Step 21: Documented the entire process. "
            "Step 22: Reviewed security best practices. "
            "Step 23: Hardened the container image. "
            "Step 24: Scanned for vulnerabilities. "
            "Step 25: Final deployment to production."
        ),
        "max_tokens": 200,
        "force": False,
    },
    "Force prune": {
        "description": "Moderate text, forced pruning regardless of health status.",
        "context": (
            "We discussed authentication options: JWT tokens vs session cookies. "
            "Decided on JWT with refresh tokens. "
            "Implemented the token generation and validation middleware. "
            "Added rate limiting to prevent brute force attacks. "
            "Set up CORS policies for the frontend domain."
        ),
        "max_tokens": 500,
        "force": True,
    },
}


def _load_autophagy_preset(name: str) -> tuple[str, int, bool]:
    p = AUTOPHAGY_PRESETS.get(name, AUTOPHAGY_PRESETS["(custom)"])
    return p["context"], p["max_tokens"], p["force"]


# ── Tab 1: Healing Loop logic ─────────────────────────────────────────────


def run_healing(
    preset_name: str,
    max_retries: int,
) -> tuple[str, str, str]:
    """Run the ChaperoneLoop healing simulation.

    Returns (outcome_banner_html, attempts_md, summary_md).
    """
    p = HEALING_PRESETS.get(preset_name, HEALING_PRESETS["(custom)"])

    generator = create_mock_healing_generator(
        initial_output=p["initial_output"],
        healed_output=p["healed_output"],
        heal_on_error_containing=p["heal_on_error_containing"],
    )

    loop = ChaperoneLoop(
        generator=generator,
        chaperone=Chaperone(silent=True),
        schema=PriceQuote,
        max_retries=int(max_retries),
        confidence_decay=0.1,
        silent=True,
    )

    result: HealingResult = loop.heal("Generate a price quote")

    # ── Outcome banner ────────────────────────────────────────────────
    color, label = OUTCOME_STYLES.get(result.outcome, ("#888", str(result.outcome)))
    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{color}20;border:2px solid {color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f"{label}</span>"
        f'<span style="color:#888;margin-left:12px">'
        f"Confidence: {result.final_confidence:.2f} | "
        f'Ubiquitin tagged: {"yes" if result.ubiquitin_tagged else "no"}</span></div>'
    )

    # ── Attempts timeline ─────────────────────────────────────────────
    lines = [
        "| # | Raw Output | Error | Success | Confidence |",
        "| ---: | :--- | :--- | :--- | ---: |",
    ]
    for attempt in result.attempts:
        raw_preview = attempt.raw_output[:60] + "…" if len(attempt.raw_output) > 60 else attempt.raw_output
        raw_preview = raw_preview.replace("|", "\\|")
        error_str = attempt.error_trace[:50] + "…" if attempt.error_trace and len(attempt.error_trace) > 50 else (attempt.error_trace or "—")
        error_str = error_str.replace("|", "\\|")
        success_icon = "✓" if attempt.success else "✗"
        success_color = "#22c55e" if attempt.success else "#ef4444"
        lines.append(
            f'| {attempt.attempt_number} | `{raw_preview}` '
            f'| {error_str} | <span style="color:{success_color}">{success_icon}</span> '
            f"| {attempt.confidence:.2f} |"
        )
    attempts_md = "\n".join(lines)

    # ── Summary ───────────────────────────────────────────────────────
    summary_parts = [f"### Healing Summary\n"]
    summary_parts.append(f"- **Outcome**: {result.outcome.value}")
    summary_parts.append(f"- **Total attempts**: {len(result.attempts)}")
    summary_parts.append(f"- **Final confidence**: {result.final_confidence:.2f}")
    summary_parts.append(f"- **Ubiquitin tagged**: {'Yes — marked for disposal' if result.ubiquitin_tagged else 'No'}")

    if result.folded:
        summary_parts.append(f"\n**Folded output**: `{result.folded}`")

    summary_parts.append("\n### How It Works\n")
    summary_parts.append("1. Generator produces raw output (simulated LLM response)")
    summary_parts.append("2. Chaperone validates against PriceQuote schema")
    summary_parts.append("3. On failure: error message is fed back to generator for refolding")
    summary_parts.append("4. Confidence decays with each retry (−0.1 per attempt)")
    summary_parts.append("5. If all retries fail → output is **ubiquitin tagged** for degradation")

    summary_md = "\n".join(summary_parts)

    return banner, attempts_md, summary_md


# ── Tab 2: Autophagy logic ───────────────────────────────────────────────


def run_autophagy(
    preset_name: str,
    context: str,
    max_tokens: int,
    force: bool,
) -> tuple[str, str, str]:
    """Run the AutophagyDaemon health check and optional pruning.

    Returns (health_badge_html, context_display_md, prune_stats_md).
    """
    if not context.strip():
        return "Enter context text to analyze.", "", ""

    daemon = AutophagyDaemon(
        histone_store=HistoneStore(silent=True),
        lysosome=Lysosome(silent=True),
        summarizer=create_simple_summarizer(),
        toxicity_threshold=0.8,
        silent=True,
    )

    # Health assessment
    metrics = daemon.assess_health(context, int(max_tokens))

    color, label = HEALTH_STYLES.get(metrics.status, ("#888", str(metrics.status)))
    fill_pct = metrics.fill_percentage * 100

    badge = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{color}20;border:2px solid {color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f"{label}</span>"
        f'<span style="color:#888;margin-left:12px">'
        f"Fill: {fill_pct:.0f}% | Est. tokens: {metrics.estimated_tokens} / {metrics.max_tokens}"
        f"</span><br>"
        f'<div style="margin-top:8px;background:#e5e7eb;border-radius:4px;height:20px">'
        f'<div style="width:{min(fill_pct, 100):.0f}%;background:{color};height:100%;'
        f'border-radius:4px;transition:width 0.3s"></div></div></div>'
    )

    # Prune if needed
    pruned_context, prune_result = daemon.check_and_prune(context, int(max_tokens), force=force)

    # Context display
    context_parts = ["### Context\n"]
    if prune_result and prune_result.pruned:
        context_parts.append("**Before pruning:**\n")
        context_parts.append(f"```\n{context[:500]}{'…' if len(context) > 500 else ''}\n```\n")
        context_parts.append("**After pruning:**\n")
        context_parts.append(f"```\n{pruned_context[:500]}{'…' if len(pruned_context) > 500 else ''}\n```")
    else:
        context_parts.append(f"```\n{context[:500]}{'…' if len(context) > 500 else ''}\n```")
        context_parts.append("\n*No pruning performed.*")
    context_md = "\n".join(context_parts)

    # Prune stats
    stats_parts = ["### Health Metrics\n"]
    stats_parts.append(f"| Metric | Value |")
    stats_parts.append(f"| :--- | :--- |")
    stats_parts.append(f"| Status | {metrics.status.value} |")
    stats_parts.append(f"| Estimated tokens | {metrics.estimated_tokens} |")
    stats_parts.append(f"| Max tokens | {metrics.max_tokens} |")
    stats_parts.append(f"| Fill percentage | {fill_pct:.1f}% |")
    stats_parts.append(f"| Useful content ratio | {metrics.useful_content_ratio:.2f} |")

    if prune_result and prune_result.pruned:
        stats_parts.append(f"\n### Prune Results\n")
        stats_parts.append(f"| Metric | Value |")
        stats_parts.append(f"| :--- | :--- |")
        stats_parts.append(f"| Tokens before | {prune_result.tokens_before} |")
        stats_parts.append(f"| Tokens after | {prune_result.tokens_after} |")
        stats_parts.append(f"| Tokens freed | {prune_result.tokens_freed} |")
        stats_parts.append(f"| Waste items flushed | {prune_result.waste_items_flushed} |")
        stats_parts.append(f"| Duration | {prune_result.duration_ms:.1f} ms |")
        if prune_result.summary_stored:
            stats_parts.append(f"\n**Summary stored**: _{prune_result.summary_stored}_")
    elif force:
        stats_parts.append(f"\n*Force prune requested but no pruning was needed/performed.*")

    prune_stats_md = "\n".join(stats_parts)

    return badge, context_md, prune_stats_md


# ── Gradio UI ──────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Chaperone Healing & Autophagy") as app:
        gr.Markdown(
            "# 🩹 Chaperone Healing Loop & Autophagy\n"
            "**Tab 1**: Structural healing of invalid LLM output via the "
            "Chaperone Loop. **Tab 2**: Context pruning via the Autophagy Daemon."
        )

        with gr.Tabs():
            # ── Tab 1: Healing Loop ───────────────────────────────────
            with gr.TabItem("Healing Loop"):
                with gr.Row():
                    heal_preset_dd = gr.Dropdown(
                        choices=list(HEALING_PRESETS.keys()),
                        value="Healed after retry",
                        label="Preset",
                        scale=2,
                    )
                    heal_btn = gr.Button("Run Healing", variant="primary", scale=1)

                retries_sl = gr.Slider(1, 5, value=3, step=1, label="Max retries")

                heal_banner = gr.HTML(label="Outcome")
                heal_attempts = gr.Markdown(label="Attempt Timeline")
                heal_summary = gr.Markdown(label="Summary")

                heal_preset_dd.change(
                    fn=_load_healing_preset,
                    inputs=[heal_preset_dd],
                    outputs=[retries_sl],
                )

                heal_btn.click(
                    fn=run_healing,
                    inputs=[heal_preset_dd, retries_sl],
                    outputs=[heal_banner, heal_attempts, heal_summary],
                )

            # ── Tab 2: Autophagy ──────────────────────────────────────
            with gr.TabItem("Autophagy"):
                with gr.Row():
                    auto_preset_dd = gr.Dropdown(
                        choices=list(AUTOPHAGY_PRESETS.keys()),
                        value="Critical context",
                        label="Preset",
                        scale=2,
                    )
                    auto_btn = gr.Button("Check & Prune", variant="primary", scale=1)

                auto_context_tb = gr.Textbox(
                    lines=6,
                    label="Context text",
                    placeholder="Enter context text to analyze…",
                )

                with gr.Row():
                    auto_tokens_sl = gr.Slider(100, 2000, value=500, step=50, label="Max tokens")
                    auto_force_cb = gr.Checkbox(label="Force prune", value=False)

                auto_badge = gr.HTML(label="Health Status")
                auto_context_md = gr.Markdown(label="Context")
                auto_stats_md = gr.Markdown(label="Prune Stats")

                auto_preset_dd.change(
                    fn=_load_autophagy_preset,
                    inputs=[auto_preset_dd],
                    outputs=[auto_context_tb, auto_tokens_sl, auto_force_cb],
                )

                auto_btn.click(
                    fn=run_autophagy,
                    inputs=[auto_preset_dd, auto_context_tb, auto_tokens_sl, auto_force_cb],
                    outputs=[auto_badge, auto_context_md, auto_stats_md],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
