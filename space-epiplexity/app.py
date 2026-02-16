"""
Operon Epistemic Stagnation Monitor -- Interactive Gradio Demo
==============================================================

Feed a sequence of messages to the EpiplexityMonitor and watch how
embedding novelty and perplexity combine to detect epistemic stagnation.

Run locally:
    pip install gradio
    python space-epiplexity/app.py

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
    EpiplexityMonitor,
    MockEmbeddingProvider,
    HealthStatus,
    EpiplexityResult,
)

# ── Presets ────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Enter your own messages (one per line).",
        "messages": [],
        "alpha": 0.5,
        "window_size": 10,
        "threshold": 0.2,
    },
    "Healthy exploration": {
        "description": "Diverse analytical messages — the monitor stays HEALTHY/EXPLORING.",
        "messages": [
            "Let's analyze the dataset structure first",
            "The primary key uses UUID v4 format",
            "I notice a many-to-many relationship between users and roles",
            "The indexes on created_at should improve query performance",
            "There's a potential N+1 query issue in the dashboard endpoint",
            "We could add Redis caching for the session store",
            "The authentication flow uses OAuth2 with PKCE",
            "Database migrations need to be idempotent",
            "Let me check the API rate limiting configuration",
            "The webhook retry logic uses exponential backoff",
        ],
        "alpha": 0.5,
        "window_size": 10,
        "threshold": 0.2,
    },
    "Gradual stagnation": {
        "description": "Messages become increasingly repetitive → STAGNANT → CRITICAL.",
        "messages": [
            "Let me investigate the error in the payment module",
            "The error seems related to currency conversion",
            "Looking at the currency conversion code more carefully",
            "The currency conversion is definitely the issue",
            "Still examining the currency conversion logic",
            "The currency conversion code needs fixing",
            "Currency conversion is the problem here",
            "The issue is currency conversion",
            "Currency conversion problem",
            "Currency conversion",
        ],
        "alpha": 0.5,
        "window_size": 5,
        "threshold": 0.2,
    },
    "Sudden loop": {
        "description": "4 diverse messages then 6 identical — sharp STAGNANT transition.",
        "messages": [
            "First, let me review the architecture documentation",
            "The microservices communicate via gRPC",
            "Service discovery uses Consul with health checks",
            "The circuit breaker pattern prevents cascade failures",
            "I need to check the logs",
            "I need to check the logs",
            "I need to check the logs",
            "I need to check the logs",
            "I need to check the logs",
            "I need to check the logs",
        ],
        "alpha": 0.5,
        "window_size": 5,
        "threshold": 0.2,
    },
    "Convergence pattern": {
        "description": "Messages narrow toward an answer — CONVERGING (healthy narrowing).",
        "messages": [
            "The bug could be in the frontend or backend",
            "Narrowing it down to the API layer",
            "The issue is in the authentication middleware",
            "Specifically the JWT validation step",
            "The token expiry check has an off-by-one error",
            "Found it: comparing with <= instead of <",
            "The fix is changing line 42 to use strict less-than",
            "Verified: token expiry now correctly rejects at boundary",
            "All tests pass with the fix applied",
            "Deploying the fix to staging for final verification",
        ],
        "alpha": 0.5,
        "window_size": 10,
        "threshold": 0.15,
    },
    "Recovery from stagnation": {
        "description": "Gets stuck, then a new approach breaks the stagnation → HEALTHY again.",
        "messages": [
            "Let me try to optimize the database query",
            "The query is slow because of the join",
            "The join is slow",
            "Still looking at the slow join",
            "The join is the bottleneck",
            "Wait, let me try a completely different approach",
            "Instead of optimizing the join, I'll denormalize the data",
            "Created a materialized view for the dashboard metrics",
            "The materialized view refreshes every 5 minutes",
            "Query time dropped from 3s to 50ms with the new approach",
        ],
        "alpha": 0.5,
        "window_size": 5,
        "threshold": 0.2,
    },
}

# ── Status styling ─────────────────────────────────────────────────────────

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


# ── Preset loader ─────────────────────────────────────────────────────────


def _load_preset(name: str) -> tuple[str, float, int, float]:
    p = PRESETS.get(name, PRESETS["(custom)"])
    messages_text = "\n".join(p["messages"]) if p["messages"] else ""
    return messages_text, p["alpha"], p["window_size"], p["threshold"]


# ── Core logic ─────────────────────────────────────────────────────────────


def run_epiplexity(
    preset_name: str,
    custom_messages: str,
    alpha: float,
    window_size: int,
    threshold: float,
) -> tuple[str, str, str]:
    """Run the EpiplexityMonitor over a message sequence.

    Returns (status_banner_html, timeline_md, summary_md).
    """
    # Parse messages
    messages = [
        line.strip()
        for line in custom_messages.strip().split("\n")
        if line.strip()
    ]

    if not messages:
        empty = "Enter messages (one per line) to analyze."
        return empty, "", ""

    monitor = EpiplexityMonitor(
        embedding_provider=MockEmbeddingProvider(dim=128),
        alpha=alpha,
        window_size=int(window_size),
        threshold=threshold,
        critical_duration=5,
    )

    results: list[tuple[str, EpiplexityResult]] = []
    for msg in messages:
        result = monitor.measure(msg)
        results.append((msg, result))

    # ── Final status banner ───────────────────────────────────────────
    final_status = results[-1][1].status
    color, label = STATUS_STYLES.get(final_status, ("#888", str(final_status)))

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f"background:{color}20;border:2px solid {color};margin-bottom:8px\">"
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f"Final Status: {label}</span><br>"
        f'<span style="color:#888;font-size:0.9em">'
        f"After {len(messages)} messages — α={alpha}, window={int(window_size)}, "
        f"threshold={threshold}</span></div>"
    )

    # ── Measurement timeline ──────────────────────────────────────────
    lines = [
        "| # | Message | Novelty | Perplexity | Epiplexity | Integral | Status |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for i, (msg, r) in enumerate(results, 1):
        preview = msg[:40] + "…" if len(msg) > 40 else msg
        preview = preview.replace("|", "\\|")
        lines.append(
            f"| {i} | {preview} | {r.embedding_novelty:.3f} "
            f"| {r.normalized_perplexity:.3f} | {r.epiplexity:.3f} "
            f"| {r.epiplexic_integral:.3f} | {_status_badge(r.status)} |"
        )
    timeline_md = "\n".join(lines)

    # ── Summary statistics ────────────────────────────────────────────
    epiplexities = [r.epiplexity for _, r in results]
    novelties = [r.embedding_novelty for _, r in results]

    status_counts: dict[str, int] = {}
    for _, r in results:
        s = r.status.value
        status_counts[s] = status_counts.get(s, 0) + 1

    status_breakdown = " | ".join(f"**{k}**: {v}" for k, v in status_counts.items())

    transitions = sum(
        1
        for i in range(1, len(results))
        if results[i][1].status != results[i - 1][1].status
    )

    summary = f"""### Summary Statistics

| Metric | Value |
| :--- | :--- |
| Total messages | {len(messages)} |
| Mean epiplexity | {sum(epiplexities) / len(epiplexities):.4f} |
| Min epiplexity | {min(epiplexities):.4f} |
| Max epiplexity | {max(epiplexities):.4f} |
| Mean novelty | {sum(novelties) / len(novelties):.4f} |
| Status transitions | {transitions} |
| Final integral | {results[-1][1].epiplexic_integral:.4f} |

**Status distribution**: {status_breakdown}

### How to Read

- **Embedding novelty**: How different each message's embedding is from the running average (0 = identical, 1 = maximally different)
- **Normalized perplexity**: Estimated surprise given the context (higher = more novel content)
- **Epiplexity**: Combined metric = α × novelty + (1-α) × perplexity
- **Integral**: Smoothed running sum — dropping below threshold triggers STAGNANT/CRITICAL
"""

    return banner, timeline_md, summary


# ── Gradio UI ──────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Epistemic Stagnation Monitor") as app:
        gr.Markdown(
            "# 🧠 Epistemic Stagnation Monitor\n"
            "Feed messages to the **EpiplexityMonitor** and watch how "
            "embedding novelty and perplexity detect when an agent loops."
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Gradual stagnation",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Analyze Messages", variant="primary", scale=1)

        messages_tb = gr.Textbox(
            lines=8,
            label="Messages (one per line)",
            placeholder="Enter messages here, one per line…",
        )

        with gr.Row():
            alpha_sl = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Alpha (embedding vs. perplexity)")
            window_sl = gr.Slider(3, 20, value=10, step=1, label="Window size")
            thresh_sl = gr.Slider(0.05, 0.5, value=0.2, step=0.01, label="Stagnation threshold")

        banner_html = gr.HTML(label="Status")
        timeline_md = gr.Markdown(label="Measurement Timeline")
        summary_md = gr.Markdown(label="Summary")

        # ── Event wiring ──────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[messages_tb, alpha_sl, window_sl, thresh_sl],
        )

        run_btn.click(
            fn=run_epiplexity,
            inputs=[preset_dd, messages_tb, alpha_sl, window_sl, thresh_sl],
            outputs=[banner_html, timeline_md, summary_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
