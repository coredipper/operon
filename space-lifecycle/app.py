"""
Operon Lifecycle Manager -- Telomere & Genome Demo
===================================================

Two-tab demo for agent lifecycle management:

1. Telomere Lifecycle: Watch telomeres shorten as operations execute,
   phase transitions, and optional renewal.
2. Genome: Configure genes, express active config, replicate with mutations.

Run locally:
    pip install gradio
    python space-lifecycle/app.py
"""

import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    Genome,
    Gene,
    GeneType,
    Telomere,
    TelomereStatus,
    LifecyclePhase,
)


# ---------------------------------------------------------------------------
# Telomere presets
# ---------------------------------------------------------------------------

TELOMERE_PRESETS: dict[str, dict] = {
    "(custom)": {
        "max_ops": 100, "error_threshold": 10, "cost": 1, "allow_renewal": False,
        "description": "Configure your own parameters",
    },
    "Long-lived agent": {
        "max_ops": 200, "error_threshold": 20, "cost": 1, "allow_renewal": False,
        "description": "High capacity agent -- slow telomere depletion",
    },
    "Fragile agent": {
        "max_ops": 30, "error_threshold": 3, "cost": 1, "allow_renewal": False,
        "description": "Low capacity -- enters senescence quickly",
    },
    "Error-prone agent": {
        "max_ops": 100, "error_threshold": 5, "cost": 1, "allow_renewal": False,
        "description": "Errors injected every 10 ops -- tests error accumulation",
    },
    "Renewable agent": {
        "max_ops": 50, "error_threshold": 10, "cost": 2, "allow_renewal": True,
        "description": "Renewal enabled -- telomeres extend when senescent",
    },
}


# ---------------------------------------------------------------------------
# Genome presets
# ---------------------------------------------------------------------------

GENOME_PRESETS: dict[str, list[dict]] = {
    "(custom)": [],
    "Worker agent": [
        {"name": "model", "value": "gpt-4", "type": "STRUCTURAL"},
        {"name": "temperature", "value": "0.7", "type": "REGULATORY"},
        {"name": "max_tokens", "value": "4096", "type": "STRUCTURAL"},
        {"name": "retries", "value": "3", "type": "HOUSEKEEPING"},
        {"name": "debug", "value": "False", "type": "DORMANT"},
    ],
    "Creative agent": [
        {"name": "model", "value": "gpt-4", "type": "STRUCTURAL"},
        {"name": "temperature", "value": "1.2", "type": "REGULATORY"},
        {"name": "creativity", "value": "0.9", "type": "REGULATORY"},
        {"name": "max_tokens", "value": "8192", "type": "STRUCTURAL"},
        {"name": "experimental", "value": "True", "type": "CONDITIONAL"},
    ],
    "Safety-first": [
        {"name": "model", "value": "gpt-4", "type": "STRUCTURAL"},
        {"name": "safety_checks", "value": "True", "type": "STRUCTURAL"},
        {"name": "temperature", "value": "0.3", "type": "REGULATORY"},
        {"name": "experimental", "value": "False", "type": "DORMANT"},
        {"name": "audit_log", "value": "True", "type": "HOUSEKEEPING"},
    ],
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

PHASE_STYLES = {
    LifecyclePhase.NASCENT: ("#94a3b8", "NASCENT", "Initializing"),
    LifecyclePhase.ACTIVE: ("#22c55e", "ACTIVE", "Normal operation"),
    LifecyclePhase.SENESCENT: ("#f59e0b", "SENESCENT", "Aging, reduced capability"),
    LifecyclePhase.APOPTOTIC: ("#ef4444", "APOPTOTIC", "Preparing for shutdown"),
    LifecyclePhase.TERMINATED: ("#6b7280", "TERMINATED", "No longer operational"),
}

GENE_TYPE_MAP = {
    "STRUCTURAL": GeneType.STRUCTURAL,
    "REGULATORY": GeneType.REGULATORY,
    "HOUSEKEEPING": GeneType.HOUSEKEEPING,
    "CONDITIONAL": GeneType.CONDITIONAL,
    "DORMANT": GeneType.DORMANT,
}


def _phase_badge(phase: LifecyclePhase) -> str:
    color, label, _ = PHASE_STYLES.get(phase, ("#6b7280", "UNKNOWN", ""))
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def _telomere_bar(current: int, maximum: int) -> str:
    pct = max(0, min(100, int(current / maximum * 100))) if maximum > 0 else 0
    if pct > 50:
        color = "#22c55e"
    elif pct > 20:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return (
        f'<div style="margin:8px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85em;">'
        f'<span>Telomere Length</span><span>{current}/{maximum}</span></div>'
        f'<div style="background:#e5e7eb;border-radius:4px;height:20px;">'
        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;'
        f'transition:width 0.3s;"></div></div></div>'
    )


# ---------------------------------------------------------------------------
# Telomere logic
# ---------------------------------------------------------------------------

def run_telomere(
    preset_name: str,
    max_ops: int,
    error_threshold: int,
    cost_per_op: int,
    allow_renewal: bool,
) -> tuple[str, str, str, str]:
    """Run the telomere lifecycle simulation.

    Returns (summary_html, telomere_bar_html, timeline_md, events_md).
    """
    max_ops = int(max_ops)
    error_threshold = int(error_threshold)
    cost_per_op = int(cost_per_op)
    is_error_prone = preset_name == "Error-prone agent"

    telomere = Telomere(
        max_operations=max_ops,
        error_threshold=error_threshold,
        allow_renewal=allow_renewal,
        silent=True,
    )
    telomere.start()

    timeline_rows = []
    phase_transitions = []
    prev_phase = telomere.get_phase()
    renewed = False

    step = 0
    while telomere.is_operational():
        step += 1

        # Inject errors for error-prone preset
        if is_error_prone and step % 10 == 0:
            telomere.record_error()
            status = telomere.get_status()
            timeline_rows.append({
                "step": step,
                "action": "ERROR",
                "length": status.telomere_length,
                "remaining": status.operations_remaining,
                "health": status.health_score,
                "phase": status.phase,
            })
            new_phase = status.phase
            if new_phase != prev_phase:
                phase_transitions.append((step, prev_phase, new_phase))
                prev_phase = new_phase
            if not telomere.is_operational():
                break
            continue

        can_continue = telomere.tick(cost=cost_per_op)
        status = telomere.get_status()

        new_phase = status.phase
        if new_phase != prev_phase:
            phase_transitions.append((step, prev_phase, new_phase))
            prev_phase = new_phase

        timeline_rows.append({
            "step": step,
            "action": "TICK",
            "length": status.telomere_length,
            "remaining": status.operations_remaining,
            "health": status.health_score,
            "phase": status.phase,
        })

        # Renewal when senescent
        if allow_renewal and not renewed and new_phase == LifecyclePhase.SENESCENT:
            telomere.renew()
            renewed = True
            status = telomere.get_status()
            new_phase = status.phase
            if new_phase != prev_phase:
                phase_transitions.append((step, prev_phase, new_phase))
                prev_phase = new_phase
            timeline_rows.append({
                "step": step,
                "action": "RENEW",
                "length": status.telomere_length,
                "remaining": status.operations_remaining,
                "health": status.health_score,
                "phase": status.phase,
            })

        if not can_continue:
            break

        # Safety cap
        if step > max_ops + 50:
            break

    # Final status
    final_status = telomere.get_status()
    stats = telomere.get_statistics()

    # --- Summary banner ---
    phase_color, _, phase_desc = PHASE_STYLES.get(
        final_status.phase, ("#6b7280", "UNKNOWN", "")
    )
    summary_html = (
        f'<div style="padding:16px;border-radius:8px;border:2px solid {phase_color};background:#f9fafb;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
        f'<span style="font-size:1.2em;font-weight:700;">Final Phase:</span>'
        f'{_phase_badge(final_status.phase)}'
        f'<span style="color:#6b7280;font-size:0.9em;">-- {phase_desc}</span>'
        f'</div>'
        f'<div style="display:flex;gap:20px;font-size:0.9em;flex-wrap:wrap;">'
        f'<span>Operations: <b>{step}</b></span>'
        f'<span>Health: <b>{final_status.health_score:.0%}</b></span>'
        f'<span>Telomere: <b>{final_status.telomere_length}/{final_status.max_telomere_length}</b></span>'
        f'</div>'
        f'</div>'
    )

    # --- Telomere bar ---
    bar_html = _telomere_bar(final_status.telomere_length, final_status.max_telomere_length)

    # --- Timeline table (sample every N rows if large) ---
    sample_interval = max(1, len(timeline_rows) // 30)
    timeline_md = "| Step | Action | Telomere | Remaining | Health | Phase |\n"
    timeline_md += "|------|--------|----------|-----------|--------|-------|\n"
    for i, row in enumerate(timeline_rows):
        if i % sample_interval == 0 or i == len(timeline_rows) - 1 or row["action"] in ("RENEW", "ERROR"):
            timeline_md += (
                f'| {row["step"]} | {row["action"]} | {row["length"]} '
                f'| {row["remaining"]} | {row["health"]:.0%} '
                f'| {_phase_badge(row["phase"])} |\n'
            )

    if phase_transitions:
        timeline_md += "\n**Phase transitions:**\n\n"
        for step_num, old, new in phase_transitions:
            _, old_label, _ = PHASE_STYLES.get(old, ("#6b7280", "?", ""))
            _, new_label, _ = PHASE_STYLES.get(new, ("#6b7280", "?", ""))
            timeline_md += f"- Step {step_num}: {old_label} -> {new_label}\n"

    # --- Events ---
    events_md = "### Lifecycle Events\n\n"
    events = telomere.get_events(limit=20)
    if events:
        events_md += "| Event | Details |\n"
        events_md += "|-------|---------|\n"
        for ev in events:
            details_str = ", ".join(f"{k}={v}" for k, v in ev.details.items()) if ev.details else "--"
            events_md += f"| {ev.event_type} | {details_str} |\n"
    else:
        events_md += "*No events recorded.*\n"

    return summary_html, bar_html, timeline_md, events_md


def load_telomere_preset(name: str):
    preset = TELOMERE_PRESETS.get(name)
    if not preset:
        return 100, 10, 1, False
    return preset["max_ops"], preset["error_threshold"], preset["cost"], preset["allow_renewal"]


# ---------------------------------------------------------------------------
# Genome logic
# ---------------------------------------------------------------------------

def _parse_gene_value(value_str: str):
    """Parse a string into a typed value."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def run_genome_express(
    name1, val1, type1,
    name2, val2, type2,
    name3, val3, type3,
    name4, val4, type4,
    name5, val5, type5,
) -> tuple[str, str]:
    """Express a genome and show active config.

    Returns (config_html, stats_md).
    """
    names = [name1, name2, name3, name4, name5]
    values = [val1, val2, val3, val4, val5]
    types = [type1, type2, type3, type4, type5]

    genes = []
    for name, val, gtype in zip(names, values, types):
        if not name.strip():
            continue
        gene_type = GENE_TYPE_MAP.get(gtype, GeneType.STRUCTURAL)
        genes.append(Gene(
            name=name.strip(),
            value=_parse_gene_value(val.strip()),
            gene_type=gene_type,
        ))

    if not genes:
        return "Add at least one gene.", ""

    genome = Genome(genes=genes, allow_mutations=True, silent=True)
    expressed = genome.express()

    # --- Config display ---
    config_html = (
        '<div style="padding:16px;border-radius:8px;border:2px solid #22c55e;background:#f0fdf4;">'
        '<div style="font-size:1.2em;font-weight:700;color:#16a34a;margin-bottom:8px;">'
        'Expressed Configuration</div>'
    )
    for key, value in expressed.items():
        config_html += (
            f'<div style="font-family:monospace;font-size:0.95em;padding:2px 0;">'
            f'<span style="color:#6b7280;">{key}:</span> '
            f'<span style="color:#15803d;font-weight:600;">{value}</span></div>'
        )
    config_html += f'<div style="margin-top:8px;font-size:0.8em;color:#6b7280;">Genome hash: <code>{genome.get_hash()}</code></div>'
    config_html += '</div>'

    # --- Stats ---
    stats = genome.get_statistics()
    gene_list = genome.list_genes()

    stats_md = "### Gene Details\n\n"
    stats_md += "| Name | Value | Type | Expression |\n"
    stats_md += "|------|-------|------|------------|\n"
    for g in gene_list:
        stats_md += f"| {g['name']} | {g['value']} | {g['type']} | {g['expression']} |\n"

    stats_md += f"\n**Total genes:** {stats['total_genes']}\n\n"
    stats_md += f"**Generation:** {stats['generation']}\n\n"
    stats_md += f"**Genome hash:** `{genome.get_hash()}`\n"

    return config_html, stats_md


def run_genome_replicate(
    name1, val1, type1,
    name2, val2, type2,
    name3, val3, type3,
    name4, val4, type4,
    name5, val5, type5,
) -> tuple[str, str]:
    """Replicate genome with mutations and show diff.

    Returns (diff_html, details_md).
    """
    names = [name1, name2, name3, name4, name5]
    values = [val1, val2, val3, val4, val5]
    types = [type1, type2, type3, type4, type5]

    genes = []
    for name, val, gtype in zip(names, values, types):
        if not name.strip():
            continue
        gene_type = GENE_TYPE_MAP.get(gtype, GeneType.STRUCTURAL)
        genes.append(Gene(
            name=name.strip(),
            value=_parse_gene_value(val.strip()),
            gene_type=gene_type,
        ))

    if not genes:
        return "Add at least one gene.", ""

    parent = Genome(genes=genes, allow_mutations=True, silent=True)

    # Create mutations: modify first REGULATORY gene's value
    mutations = {}
    for g in genes:
        if g.gene_type == GeneType.REGULATORY:
            if isinstance(g.value, (int, float)):
                mutations[g.name] = round(g.value * 1.5, 2)
            elif isinstance(g.value, bool):
                mutations[g.name] = not g.value
            else:
                mutations[g.name] = g.value + "_mutated"
            break

    if not mutations:
        # Mutate first gene if no regulatory found
        g = genes[0]
        if isinstance(g.value, (int, float)):
            mutations[g.name] = round(g.value * 2, 2)
        else:
            mutations[g.name] = str(g.value) + "_v2"

    child = parent.replicate(mutations=mutations)

    diff = parent.diff(child)

    # --- Diff display ---
    diff_html = (
        '<div style="padding:16px;border-radius:8px;border:2px solid #8b5cf6;background:#f5f3ff;">'
        '<div style="font-size:1.2em;font-weight:700;color:#7c3aed;margin-bottom:8px;">'
        'Replication Diff</div>'
    )
    if diff:
        for gene_name, (parent_val, child_val) in diff.items():
            diff_html += (
                f'<div style="font-family:monospace;font-size:0.95em;padding:4px 0;">'
                f'<span style="color:#6b7280;">{gene_name}:</span> '
                f'<span style="color:#dc2626;text-decoration:line-through;">{parent_val}</span> '
                f'-> <span style="color:#16a34a;font-weight:600;">{child_val}</span></div>'
            )
    else:
        diff_html += '<div style="color:#6b7280;">No differences found.</div>'

    diff_html += (
        f'<div style="margin-top:8px;font-size:0.8em;color:#6b7280;">'
        f'Parent hash: <code>{parent.get_hash()}</code> | '
        f'Child hash: <code>{child.get_hash()}</code></div>'
    )
    diff_html += '</div>'

    # --- Details ---
    parent_expressed = parent.express()
    child_expressed = child.express()

    details_md = "### Comparison\n\n"
    details_md += "| Gene | Parent | Child | Changed |\n"
    details_md += "|------|--------|-------|---------|\n"
    all_keys = set(list(parent_expressed.keys()) + list(child_expressed.keys()))
    for key in sorted(all_keys):
        pv = parent_expressed.get(key, "--")
        cv = child_expressed.get(key, "--")
        changed = "Yes" if pv != cv else ""
        details_md += f"| {key} | {pv} | {cv} | {changed} |\n"

    details_md += f"\n**Mutations applied:** {mutations}\n"

    return diff_html, details_md


def load_genome_preset(name: str):
    """Load a genome preset into the gene fields."""
    preset = GENOME_PRESETS.get(name, [])
    result = []
    for i in range(5):
        if i < len(preset):
            result.extend([preset[i]["name"], preset[i]["value"], preset[i]["type"]])
        else:
            result.extend(["", "", "STRUCTURAL"])
    return result


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    gene_type_choices = list(GENE_TYPE_MAP.keys())

    with gr.Blocks(title="Operon Lifecycle Manager") as app:
        gr.Markdown(
            "# Operon Lifecycle Manager\n"
            "Agent lifecycle management with biological **telomere shortening** "
            "and **genome configuration**.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Tabs():
            # --- Telomere Tab ---
            with gr.TabItem("Telomere Lifecycle"):
                gr.Markdown(
                    "### Telomere Shortening Simulation\n\n"
                    "Watch how an agent's telomeres shorten with each operation. "
                    "When telomeres deplete, the agent enters senescence. "
                    "With renewal enabled, telomeres can be extended."
                )

                with gr.Row():
                    telo_preset = gr.Dropdown(
                        choices=list(TELOMERE_PRESETS.keys()),
                        value="(custom)",
                        label="Load Preset",
                        scale=2,
                    )
                    telo_run_btn = gr.Button("Run Lifecycle", variant="primary", scale=1)

                with gr.Row():
                    max_ops_slider = gr.Slider(
                        minimum=10, maximum=300, value=100, step=10,
                        label="Max Operations",
                    )
                    error_thresh_slider = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="Error Threshold",
                    )
                    cost_slider = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1,
                        label="Cost per Operation",
                    )
                    renewal_check = gr.Checkbox(
                        label="Allow Renewal",
                        value=False,
                    )

                telo_summary = gr.HTML(label="Summary")
                telo_bar = gr.HTML(label="Telomere")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Timeline")
                        telo_timeline = gr.Markdown()
                    with gr.Column(scale=1):
                        telo_events = gr.Markdown()

                telo_run_btn.click(
                    fn=run_telomere,
                    inputs=[telo_preset, max_ops_slider, error_thresh_slider, cost_slider, renewal_check],
                    outputs=[telo_summary, telo_bar, telo_timeline, telo_events],
                )
                telo_preset.change(
                    fn=load_telomere_preset,
                    inputs=[telo_preset],
                    outputs=[max_ops_slider, error_thresh_slider, cost_slider, renewal_check],
                )

            # --- Genome Tab ---
            with gr.TabItem("Genome"):
                gr.Markdown(
                    "### Genome Configuration\n\n"
                    "Configure agent genes with types: STRUCTURAL (core), "
                    "REGULATORY (controls), HOUSEKEEPING (essential), "
                    "CONDITIONAL (context-dependent), DORMANT (inactive)."
                )

                genome_preset = gr.Dropdown(
                    choices=list(GENOME_PRESETS.keys()),
                    value="(custom)",
                    label="Load Preset",
                )

                gene_components = []
                for i in range(5):
                    with gr.Row():
                        gname = gr.Textbox(label=f"Gene {i+1} Name", value="", scale=2)
                        gval = gr.Textbox(label="Value", value="", scale=2)
                        gtype = gr.Dropdown(
                            choices=gene_type_choices,
                            value="STRUCTURAL",
                            label="Type",
                            scale=1,
                        )
                        gene_components.extend([gname, gval, gtype])

                with gr.Row():
                    express_btn = gr.Button("Express", variant="primary")
                    replicate_btn = gr.Button("Replicate with Mutations", variant="secondary")

                genome_config = gr.HTML(label="Configuration")
                genome_stats = gr.Markdown()

                express_btn.click(
                    fn=run_genome_express,
                    inputs=gene_components,
                    outputs=[genome_config, genome_stats],
                )
                replicate_btn.click(
                    fn=run_genome_replicate,
                    inputs=gene_components,
                    outputs=[genome_config, genome_stats],
                )
                genome_preset.change(
                    fn=load_genome_preset,
                    inputs=[genome_preset],
                    outputs=gene_components,
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
