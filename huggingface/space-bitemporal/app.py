"""
Operon Bi-Temporal Memory — Explorer (Gradio Demo)
===================================================

Three-tab demo:
  1. Fact Timeline   — load preset scenarios, inspect facts and corrections
  2. Point-in-Time   — query belief state at any (valid, record) coordinate
  3. Diff & Audit    — compare time points, view full audit trails

Run locally:
    pip install gradio operon-ai
    python space-bitemporal/app.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import BiTemporalMemory


# ── Time helpers ────────────────────────────────────────────────────

_BASE = datetime(2026, 3, 15, 9, 0, 0)


def _t(day: int, hour: int = 12) -> datetime:
    return _BASE + timedelta(days=day - 1, hours=hour - 9)


def _fmt(dt: datetime) -> str:
    return dt.strftime("Day %d %H:%M").replace("Day 0", "Day ").lstrip("0")


def _day_fmt(dt: datetime) -> str:
    delta = dt - _BASE
    day = delta.days + 1
    return f"Day {day} {dt.strftime('%H:%M')}"


# ── Preset scenarios ───────────────────────────────────────────────


def _build_compliance_audit() -> BiTemporalMemory:
    """Financial product approval with retroactive correction."""
    mem = BiTemporalMemory()
    # Day 1: Quantitative model outputs risk score
    mem.record_fact(
        subject="product:BOND-7Y", predicate="risk_score",
        value=0.42, valid_from=_t(1), recorded_from=_t(1),
        source="quant_model", tags=("quantitative",),
    )
    # Day 2: Liquidity assessed
    mem.record_fact(
        subject="product:BOND-7Y", predicate="liquidity_class",
        value="B", valid_from=_t(2), recorded_from=_t(2),
        source="liquidity_engine", tags=("quantitative",),
    )
    # Day 3: Regulatory category assigned
    mem.record_fact(
        subject="product:BOND-7Y", predicate="regulatory_category",
        value="standard", valid_from=_t(1), recorded_from=_t(3),
        source="compliance_team", tags=("regulatory",),
    )
    # Day 5: Post-approval audit reveals risk was higher
    risk_facts = mem.retrieve_valid_at(at=_t(5), subject="product:BOND-7Y",
                                       predicate="risk_score")
    if risk_facts:
        mem.correct_fact(
            old_fact_id=risk_facts[0].fact_id, value=0.68,
            valid_from=_t(1), recorded_from=_t(5),
            source="audit_review", tags=("correction", "audit"),
        )
    return mem


def _build_client_onboarding() -> BiTemporalMemory:
    """Client onboarding with progressive fact discovery."""
    mem = BiTemporalMemory()
    # Day 1: Initial KYC data
    mem.record_fact(
        subject="client:ACME-42", predicate="kyc_status",
        value="pending", valid_from=_t(1), recorded_from=_t(1),
        source="onboarding_portal",
    )
    mem.record_fact(
        subject="client:ACME-42", predicate="revenue_band",
        value="mid-market", valid_from=_t(1), recorded_from=_t(1),
        source="crm_sync",
    )
    # Day 2: KYC approved
    mem.record_fact(
        subject="client:ACME-42", predicate="kyc_status",
        value="approved", valid_from=_t(2), recorded_from=_t(2),
        source="compliance_team",
    )
    # Day 3: Revenue reclassified (retroactive to day 1)
    rev_facts = mem.retrieve_valid_at(at=_t(3), subject="client:ACME-42",
                                       predicate="revenue_band")
    if rev_facts:
        mem.correct_fact(
            old_fact_id=rev_facts[0].fact_id, value="enterprise",
            valid_from=_t(1), recorded_from=_t(3),
            source="finance_team", tags=("correction",),
        )
    # Day 4: Tier assigned based on corrected revenue
    mem.record_fact(
        subject="client:ACME-42", predicate="tier",
        value="gold", valid_from=_t(4), recorded_from=_t(4),
        source="tier_engine",
    )
    return mem


def _build_incident_response() -> BiTemporalMemory:
    """Incident timeline with evolving root cause analysis."""
    mem = BiTemporalMemory()
    # Hour 0: Alert fires
    mem.record_fact(
        subject="incident:INC-1337", predicate="severity",
        value="P2", valid_from=_t(1, 9), recorded_from=_t(1, 9),
        source="alerting_system", tags=("auto",),
    )
    mem.record_fact(
        subject="incident:INC-1337", predicate="root_cause",
        value="database_timeout", valid_from=_t(1, 9), recorded_from=_t(1, 10),
        source="oncall_engineer",
    )
    # Hour 3: Escalated to P1
    sev_facts = mem.retrieve_valid_at(at=_t(1, 12), subject="incident:INC-1337",
                                       predicate="severity")
    if sev_facts:
        mem.correct_fact(
            old_fact_id=sev_facts[0].fact_id, value="P1",
            valid_from=_t(1, 9), recorded_from=_t(1, 12),
            source="incident_commander", tags=("escalation",),
        )
    # Hour 5: Root cause updated
    rc_facts = mem.retrieve_valid_at(at=_t(1, 14), subject="incident:INC-1337",
                                      predicate="root_cause")
    if rc_facts:
        mem.correct_fact(
            old_fact_id=rc_facts[0].fact_id,
            value="network_partition_causing_db_timeout",
            valid_from=_t(1, 9), recorded_from=_t(1, 14),
            source="sre_team", tags=("correction", "root_cause"),
        )
    # Hour 8: Resolved
    mem.record_fact(
        subject="incident:INC-1337", predicate="status",
        value="resolved", valid_from=_t(1, 17), recorded_from=_t(1, 17),
        source="incident_commander",
    )
    return mem


PRESETS = {
    "Compliance Audit (BOND-7Y)": {
        "description": "Financial product approval with post-approval risk correction.",
        "build_fn": _build_compliance_audit,
        "subjects": ["product:BOND-7Y"],
    },
    "Client Onboarding (ACME-42)": {
        "description": "Progressive KYC and revenue reclassification during onboarding.",
        "build_fn": _build_client_onboarding,
        "subjects": ["client:ACME-42"],
    },
    "Incident Response (INC-1337)": {
        "description": "Evolving severity and root cause during an incident.",
        "build_fn": _build_incident_response,
        "subjects": ["incident:INC-1337"],
    },
}

# Module-level state: the currently loaded memory instance
_current_mem: BiTemporalMemory | None = None
_current_subjects: list[str] = []


# ── HTML helpers ────────────────────────────────────────────────────


def _badge(text, color="#6366f1"):
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:5px;'
        f'background:{color};color:#fff;font-weight:600;font-size:0.85em;margin:2px">'
        f'{text}</span>'
    )


def _fact_row_html(f, show_status=True):
    status = ""
    if show_status:
        if f.recorded_to is not None:
            status = _badge("CLOSED", "#ef4444")
        else:
            status = _badge("ACTIVE", "#22c55e")
    supersedes = ""
    if f.supersedes:
        supersedes = f' <span style="color:#9333ea;font-size:0.8em">[corrects {f.supersedes[:8]}]</span>'
    tags = ""
    if f.tags:
        tags = " ".join(_badge(t, "#64748b") for t in f.tags)
    valid_to = _day_fmt(f.valid_to) if f.valid_to else "ongoing"
    rec_to = _day_fmt(f.recorded_to) if f.recorded_to else "current"
    return (
        f'<tr style="border-bottom:1px solid #e5e7eb">'
        f'<td style="padding:6px;font-family:monospace;font-size:0.85em">{f.fact_id[:8]}</td>'
        f'<td style="padding:6px">{status}</td>'
        f'<td style="padding:6px;font-weight:600">{f.subject}</td>'
        f'<td style="padding:6px">{f.predicate}</td>'
        f'<td style="padding:6px;font-weight:600">{f.value}</td>'
        f'<td style="padding:6px;font-size:0.85em">{_day_fmt(f.valid_from)} &mdash; {valid_to}</td>'
        f'<td style="padding:6px;font-size:0.85em">{_day_fmt(f.recorded_from)} &mdash; {rec_to}</td>'
        f'<td style="padding:6px;font-size:0.85em">{f.source}{supersedes}</td>'
        f'<td style="padding:6px">{tags}</td>'
        f'</tr>'
    )


def _fact_table_html(facts, title="Facts", show_status=True):
    if not facts:
        return f'<div style="padding:12px;color:#9ca3af">{title}: No matching facts.</div>'
    header = (
        '<table style="width:100%;border-collapse:collapse">'
        '<tr style="background:#f3f4f6">'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db;text-align:left">ID</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db">Status</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db;text-align:left">Subject</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db;text-align:left">Predicate</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db;text-align:left">Value</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db;text-align:left">Valid Time</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db;text-align:left">Record Time</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db;text-align:left">Source</th>'
        '<th style="padding:6px;border-bottom:2px solid #d1d5db">Tags</th>'
        '</tr>'
    )
    rows = "".join(_fact_row_html(f, show_status) for f in facts)
    return (
        f'<div style="margin-bottom:8px;font-weight:700;font-size:1.05em">{title}</div>'
        + header + rows + '</table>'
    )


def _summary_card(label, value, color="#6366f1"):
    return (
        f'<div style="display:inline-block;padding:12px 20px;border:2px solid {color};'
        f'border-radius:8px;margin:4px;text-align:center;min-width:120px">'
        f'<div style="font-size:0.85em;color:#6b7280">{label}</div>'
        f'<div style="font-size:1.4em;font-weight:700;color:{color}">{value}</div>'
        f'</div>'
    )


# ── Tab 1: Fact Timeline ───────────────────────────────────────────


def _load_preset(preset_name):
    global _current_mem, _current_subjects
    preset = PRESETS.get(preset_name)
    if not preset:
        return "Select a preset.", ""
    _current_mem = preset["build_fn"]()
    _current_subjects = preset["subjects"]

    all_facts = _current_mem._facts
    active = [f for f in all_facts if f.recorded_to is None]
    closed = [f for f in all_facts if f.recorded_to is not None]
    corrections = [f for f in all_facts if f.supersedes is not None]

    summary = (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:16px">'
        + _summary_card("Total Facts", len(all_facts), "#3b82f6")
        + _summary_card("Active", len(active), "#22c55e")
        + _summary_card("Closed", len(closed), "#ef4444")
        + _summary_card("Corrections", len(corrections), "#9333ea")
        + '</div>'
    )
    timeline = _fact_table_html(
        sorted(all_facts, key=lambda f: f.recorded_from),
        title=f"Full Timeline ({preset_name})",
    )
    return summary, timeline


# ── Tab 2: Point-in-Time Query ─────────────────────────────────────


def _run_query(query_type, valid_day, record_day, subject_filter):
    if _current_mem is None:
        return "Load a scenario first (Tab 1)."

    subject = subject_filter.strip() or None
    valid_time = _t(int(valid_day))
    record_time = _t(int(record_day))

    if query_type == "Valid-Time Only":
        facts = _current_mem.retrieve_valid_at(at=valid_time, subject=subject)
        title = f"Valid at {_day_fmt(valid_time)} (active records only)"
        explanation = (
            '<div style="padding:8px;background:#f0fdf4;border-radius:6px;margin-bottom:12px;'
            'font-size:0.9em;color:#166534">'
            '<strong>Valid-time query:</strong> Returns facts whose valid interval contains '
            f'{_day_fmt(valid_time)}, considering only currently active records. '
            'This answers: "What is true in the world at this time?"'
            '</div>'
        )
    elif query_type == "Record-Time Only":
        facts = _current_mem.retrieve_known_at(at=record_time, subject=subject)
        title = f"Known by {_day_fmt(record_time)} (includes closed records)"
        explanation = (
            '<div style="padding:8px;background:#eff6ff;border-radius:6px;margin-bottom:12px;'
            'font-size:0.9em;color:#1e40af">'
            '<strong>Record-time query:</strong> Returns facts the system had recorded by '
            f'{_day_fmt(record_time)}, including later-closed records. '
            'This answers: "What did the system know at this time?"'
            '</div>'
        )
    else:
        facts = _current_mem.retrieve_belief_state(
            at_valid=valid_time, at_record=record_time,
        )
        title = f"Belief state at (valid={_day_fmt(valid_time)}, record={_day_fmt(record_time)})"
        explanation = (
            '<div style="padding:8px;background:#faf5ff;border-radius:6px;margin-bottom:12px;'
            'font-size:0.9em;color:#6b21a8">'
            '<strong>Belief-state query:</strong> Intersects both axes &mdash; returns facts '
            f'valid at {_day_fmt(valid_time)} AND recorded by {_day_fmt(record_time)}. '
            'This answers: "What did the system believe was true at this world-time, '
            'given only what it knew by this record-time?"'
            '</div>'
        )
        if subject:
            facts = [f for f in facts if f.subject == subject]

    return explanation + _fact_table_html(facts, title=title)


# ── Tab 3: Diff & Audit ───────────────────────────────────────────


def _run_diff(axis, t1_day, t2_day):
    if _current_mem is None:
        return "Load a scenario first (Tab 1)."

    t1 = _t(int(t1_day))
    t2 = _t(int(t2_day))
    diff = _current_mem.diff_between(t1, t2, axis=axis.lower().replace("-time", ""))
    axis_label = axis.lower().replace("-time", "")
    return _fact_table_html(
        diff,
        title=f"New facts on {axis_label} axis between {_day_fmt(t1)} and {_day_fmt(t2)}",
    )


def _run_audit(subject):
    if _current_mem is None:
        return "", ""
    subject = subject.strip()
    if not subject:
        subject = _current_subjects[0] if _current_subjects else ""
    if not subject:
        return "Enter a subject.", ""

    history = _current_mem.history(subject)
    timeline = _current_mem.timeline_for(subject)

    hist_html = _fact_table_html(
        history, title=f"History (by record time): {subject}",
    )
    time_html = _fact_table_html(
        timeline, title=f"Timeline (by valid time): {subject}",
    )
    return hist_html, time_html


# ── Gradio UI ──────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Bi-Temporal Memory Explorer") as app:
        gr.Markdown(
            "# Bi-Temporal Memory Explorer\n"
            "Explore **dual time axes**: valid time (when a fact is true) vs "
            "record time (when the system learned it). Corrections are "
            "append-only --- old records are closed, never mutated.\n\n"
            "Start by loading a **preset scenario** in the first tab."
        )

        with gr.Tabs():
            # ── Tab 1: Fact Timeline ─────────────────────────────
            with gr.TabItem("Fact Timeline"):
                gr.Markdown(
                    "Load a preset scenario to populate the bi-temporal "
                    "memory. Each scenario demonstrates corrections and "
                    "retroactive knowledge updates."
                )
                with gr.Row():
                    preset_dd = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Compliance Audit (BOND-7Y)",
                        label="Preset Scenario",
                        scale=3,
                    )
                    load_btn = gr.Button("Load Scenario", variant="primary", scale=1)

                summary_html = gr.HTML(label="Summary")
                timeline_html = gr.HTML(label="Fact Timeline")

                load_btn.click(
                    fn=_load_preset,
                    inputs=[preset_dd],
                    outputs=[summary_html, timeline_html],
                )

            # ── Tab 2: Point-in-Time Query ───────────────────────
            with gr.TabItem("Point-in-Time Query"):
                gr.Markdown(
                    "Query the memory at specific time coordinates. "
                    "Compare how **valid-time**, **record-time**, and "
                    "**belief-state** queries produce different results "
                    "for the same data."
                )
                with gr.Row():
                    query_type = gr.Dropdown(
                        choices=[
                            "Valid-Time Only",
                            "Record-Time Only",
                            "Belief State (both axes)",
                        ],
                        value="Belief State (both axes)",
                        label="Query Type",
                    )
                    subject_input = gr.Textbox(
                        value="",
                        label="Subject Filter (optional)",
                        placeholder="e.g. product:BOND-7Y",
                    )
                with gr.Row():
                    valid_slider = gr.Slider(
                        1, 7, value=2, step=1,
                        label="Valid-Time Day",
                    )
                    record_slider = gr.Slider(
                        1, 7, value=4, step=1,
                        label="Record-Time Day",
                    )
                query_btn = gr.Button("Query", variant="primary")
                query_output = gr.HTML(label="Query Results")

                query_btn.click(
                    fn=_run_query,
                    inputs=[query_type, valid_slider, record_slider, subject_input],
                    outputs=[query_output],
                )

            # ── Tab 3: Diff & Audit ──────────────────────────────
            with gr.TabItem("Diff & Audit"):
                gr.Markdown(
                    "Compare what changed between two time points, or "
                    "inspect the full audit trail for a subject."
                )

                gr.Markdown("### Temporal Diff")
                with gr.Row():
                    diff_axis = gr.Dropdown(
                        choices=["Valid-Time", "Record-Time"],
                        value="Record-Time",
                        label="Axis",
                    )
                    diff_t1 = gr.Slider(1, 7, value=1, step=1, label="From Day")
                    diff_t2 = gr.Slider(1, 7, value=5, step=1, label="To Day")
                diff_btn = gr.Button("Compute Diff", variant="primary")
                diff_output = gr.HTML(label="Diff Results")

                diff_btn.click(
                    fn=_run_diff,
                    inputs=[diff_axis, diff_t1, diff_t2],
                    outputs=[diff_output],
                )

                gr.Markdown("### Subject Audit Trail")
                with gr.Row():
                    audit_subject = gr.Textbox(
                        value="",
                        label="Subject",
                        placeholder="e.g. product:BOND-7Y",
                        scale=3,
                    )
                    audit_btn = gr.Button("Audit", variant="primary", scale=1)

                audit_history = gr.HTML(label="History (Record Order)")
                audit_timeline = gr.HTML(label="Timeline (Valid Order)")

                audit_btn.click(
                    fn=_run_audit,
                    inputs=[audit_subject],
                    outputs=[audit_history, audit_timeline],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
