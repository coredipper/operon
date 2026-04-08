"""Operon DNA Repair -- State Integrity Checking and Recovery.

Run locally:  pip install gradio && python space-dna-repair/app.py
"""

import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    Gene,
    GeneType,
    Genome,
    ExpressionLevel,
    HistoneStore,
    DNARepair,
    DamageSeverity,
)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

SEVERITY_STYLES: dict[DamageSeverity, tuple[str, str]] = {
    DamageSeverity.LOW: ("#3b82f6", "LOW"),
    DamageSeverity.MODERATE: ("#eab308", "MODERATE"),
    DamageSeverity.HIGH: ("#f97316", "HIGH"),
    DamageSeverity.CRITICAL: ("#ef4444", "CRITICAL"),
}


def _severity_badge(severity: DamageSeverity) -> str:
    color, label = SEVERITY_STYLES.get(severity, ("#6b7280", "UNKNOWN"))
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{text}</span>'
    )

def _ok_badge(text: str = "OK") -> str:
    return _badge(text, "#22c55e")

def _fail_badge(text: str = "FAIL") -> str:
    return _badge(text, "#ef4444")


def _section(title: str, body: str) -> str:
    return (f'<div style="margin:8px 0;padding:12px;border-radius:6px;'
            f'border:1px solid #e5e7eb;background:#f9fafb;">'
            f'<div style="font-weight:700;margin-bottom:6px;">{title}</div>'
            f'{body}</div>')


def _format_damage(damage_list: list) -> str:
    if not damage_list:
        return _section("Scan Result", _ok_badge("No damage detected"))
    rows = ""
    for d in damage_list:
        lc = SEVERITY_STYLES.get(d.severity, ("#6b7280", "?"))[0]
        rows += (f'<div style="margin:4px 0;padding:6px;border-left:3px solid {lc};'
                 f'padding-left:10px;">{_severity_badge(d.severity)} '
                 f'<b>{d.corruption_type.value}</b> at <code>{d.location}</code>'
                 f'<br/><span style="color:#6b7280;font-size:0.9em;">{d.description}</span>'
                 f'<br/><span style="font-size:0.85em;">Strategy: '
                 f'<code>{d.recommended_strategy.value}</code></span></div>')
    return _section(f"Scan: {len(damage_list)} damage site(s)", rows)


# ---------------------------------------------------------------------------
# Shared state factory
# ---------------------------------------------------------------------------

def _make_genome() -> Genome:
    return Genome(
        genes=[
            Gene("model", "gpt-4", gene_type=GeneType.STRUCTURAL, required=True),
            Gene("temperature", 0.7),
            Gene("max_tokens", 4096),
            Gene("retry_count", 3),
        ],
        allow_mutations=True,
        silent=True,
    )


# ---------------------------------------------------------------------------
# Tab 1: Checkpoint & Scan
# ---------------------------------------------------------------------------

CORRUPTION_CHOICES = [
    "Mutate temperature gene",
    "Silence required gene (model)",
    "Change expression (max_tokens -> HIGH)",
    "Add extra gene",
]


def tab1_init(state: dict) -> tuple[str, dict]:
    genome = _make_genome()
    histones = HistoneStore(silent=True)
    repair = DNARepair(histone_store=histones, silent=True)
    state = {"genome": genome, "histones": histones, "repair": repair, "checkpoint": None}

    genes_html = ""
    for name, gene in sorted(genome._genes.items()):
        expr = genome._expression[name].level.name
        req = " (required)" if gene.required else ""
        genes_html += (f'<div style="margin:2px 0;"><code>{name}</code> = '
                       f'<b>{gene.value}</b> [{gene.gene_type.value}] '
                       f'expression={expr}{req}</div>')
    return _section("Genome (4 genes)", genes_html), state


def tab1_checkpoint(state: dict) -> tuple[str, dict]:
    genome = state.get("genome")
    repair = state.get("repair")
    if genome is None or repair is None:
        return "Initialize the genome first.", state
    cp = repair.checkpoint(genome)
    state["checkpoint"] = cp
    body = (f'<div>ID: <code>{cp.checkpoint_id}</code> | '
            f'Hash: <code>{cp.genome_hash}</code> | '
            f'Genes: <b>{cp.gene_count}</b></div><div>Expression:</div>')
    for name, level in cp.expression_snapshot:
        body += f'<div style="margin-left:12px;"><code>{name}</code> = {ExpressionLevel(level).name}</div>'
    return _section("Checkpoint Taken", body), state


def tab1_inject(corruption_type: str, state: dict) -> tuple[str, dict]:
    genome = state.get("genome")
    if genome is None:
        return "Initialize the genome first.", state

    actions = {
        "Mutate temperature gene": (
            lambda: genome.mutate("temperature", 0.95, reason="simulated drift"),
            "Mutated <code>temperature</code>: 0.7 &rarr; 0.95"),
        "Silence required gene (model)": (
            lambda: genome.set_expression("model", ExpressionLevel.SILENCED, "corruption"),
            "Silenced required gene <code>model</code>"),
        "Change expression (max_tokens -> HIGH)": (
            lambda: genome.set_expression("max_tokens", ExpressionLevel.HIGH, ""),
            "Changed <code>max_tokens</code> expression to HIGH (no modifier)"),
        "Add extra gene": (
            lambda: genome.add_gene(Gene("rogue_param", "unexpected", gene_type=GeneType.CONDITIONAL)),
            "Added rogue gene <code>rogue_param</code>"),
    }
    if corruption_type not in actions:
        return "Unknown corruption type.", state
    fn, desc = actions[corruption_type]
    fn()
    return _section("Corruption Injected", desc), state


def tab1_scan(state: dict) -> str:
    genome = state.get("genome")
    repair = state.get("repair")
    checkpoint = state.get("checkpoint")
    if genome is None or repair is None or checkpoint is None:
        return "Take a checkpoint first, then inject corruption."
    damage = repair.scan(genome, checkpoint)
    return _format_damage(damage)


# ---------------------------------------------------------------------------
# Tab 2: Repair & Certify
# ---------------------------------------------------------------------------

PIPELINE_PRESETS = {
    "Temperature drift": lambda g: g.mutate("temperature", 0.95, reason="drift"),
    "Required gene silenced": lambda g: g.set_expression("model", ExpressionLevel.SILENCED, "corruption"),
    "Multiple corruptions": lambda g: (
        g.mutate("temperature", 0.95, reason="drift"),
        g.set_expression("max_tokens", ExpressionLevel.HIGH, ""),
        g.add_gene(Gene("debug_flag", True, gene_type=GeneType.CONDITIONAL)),
    ),
}


def tab2_pipeline(preset_name: str) -> str:
    if preset_name not in PIPELINE_PRESETS:
        return "Select a preset scenario."

    genome = _make_genome()
    histones = HistoneStore(silent=True)
    repair = DNARepair(histone_store=histones, silent=True)
    html = ""

    # Step 1: Checkpoint
    checkpoint = repair.checkpoint(genome)
    html += _section(
        "Step 1: Checkpoint",
        f'Hash: <code>{checkpoint.genome_hash}</code> | '
        f'Genes: {checkpoint.gene_count} | '
        f'ID: <code>{checkpoint.checkpoint_id}</code>',
    )

    # Step 2: Inject corruption
    PIPELINE_PRESETS[preset_name](genome)
    html += _section("Step 2: Inject Corruption", f'Scenario: <b>{preset_name}</b>')

    # Step 3: Scan
    damage = repair.scan(genome, checkpoint)
    html += _format_damage(damage)

    # Step 4: Repair
    repair_html = ""
    for d in damage:
        result = repair.repair(genome, d)
        badge = _ok_badge("SUCCESS") if result.success else _fail_badge("FAILED")
        repair_html += (
            f'<div style="margin:4px 0;">{badge} '
            f'<code>{result.strategy_used.value}</code>: {result.details}</div>'
        )
    html += _section("Step 4: Repair", repair_html or "No damage to repair.")

    # Step 5: Re-scan
    post_damage = repair.scan(genome, checkpoint)
    if post_damage:
        html += _format_damage(post_damage)
    else:
        html += _section("Step 5: Re-scan", _ok_badge("Clean -- no remaining damage"))

    # Step 6: Certify
    cert = repair.certify(genome, checkpoint)
    v = cert.verify()
    status = _ok_badge("HOLDS") if v.holds else _fail_badge("DOES NOT HOLD")
    evidence = "".join(f'<div><code>{k}</code>: {val}</div>' for k, val in v.evidence.items())
    html += _section("Step 6: Certificate Verification",
                      f'{status}<div style="margin-top:6px;font-size:0.9em;">{evidence}</div>')

    # Statistics
    stats = repair.get_statistics()
    stats_body = (
        f'Checkpoints: {stats["checkpoints"]} | '
        f'Scans: {stats["scans_performed"]} | '
        f'Repairs: {stats["repairs_successful"]}/{stats["repairs_attempted"]}'
    )
    html += _section("Statistics", stats_body)

    return html


# ---------------------------------------------------------------------------
# Tab 3: Repair Memory
# ---------------------------------------------------------------------------

def tab3_markers(tag_filter: str) -> str:
    # Run a scenario that generates histone markers
    genome = _make_genome()
    histones = HistoneStore(silent=True)
    repair = DNARepair(histone_store=histones, silent=True)
    checkpoint = repair.checkpoint(genome)

    # Inject multiple corruptions to generate markers
    genome.mutate("temperature", 0.95, reason="drift")
    genome.set_expression("max_tokens", ExpressionLevel.HIGH, "")
    damage = repair.scan(genome, checkpoint)
    for d in damage:
        repair.repair(genome, d)

    # Retrieve markers
    tags = None if tag_filter == "all" else [tag_filter]
    context = histones.retrieve_context(tags=tags)

    if not context.markers:
        return _section("Histone Markers", "No markers found for this filter.")

    _str_colors = {"WEAK": "#93c5fd", "MODERATE": "#fcd34d",
                    "STRONG": "#fb923c", "PERMANENT": "#f87171"}
    rows = ""
    for m in context.markers:
        sc = _str_colors.get(m.strength.name, "#d1d5db")
        tags = " ".join(f'<span style="background:#e5e7eb;padding:1px 6px;'
                        f'border-radius:3px;font-size:0.8em;">{t}</span>' for t in m.tags)
        rows += (f'<div style="margin:6px 0;padding:8px;border:1px solid #e5e7eb;'
                 f'border-radius:4px;"><div style="font-size:0.9em;">{m.content}</div>'
                 f'<div style="margin-top:4px;display:flex;gap:8px;align-items:center;">'
                 f'<span style="background:{sc};padding:1px 6px;border-radius:3px;'
                 f'font-size:0.8em;font-weight:600;">{m.strength.name}</span>'
                 f'<span style="font-size:0.8em;color:#6b7280;">{m.marker_type.value}</span>'
                 f'{tags}</div></div>')
    return _section(
        f"Histone Markers ({len(context.markers)} found, "
        f"{context.active_markers} active)",
        rows,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon DNA Repair") as app:
        gr.Markdown(
            "# Operon DNA Repair\n"
            "Checkpoint genome state, inject corruptions, scan for damage, "
            "repair mutations, and verify integrity certificates.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        # Session-scoped state for Tab 1
        t1_state = gr.State(value={})

        # ------ Tab 1: Checkpoint & Scan ------
        with gr.Tab("Checkpoint & Scan"):
            gr.Markdown(
                "Initialize a genome with 4 genes, take a checkpoint, "
                "inject a corruption, and scan for damage."
            )
            init_btn = gr.Button("Initialize Genome", variant="primary")
            genome_html = gr.HTML()

            checkpoint_btn = gr.Button("Take Checkpoint")
            checkpoint_html = gr.HTML()

            with gr.Row():
                corruption_dd = gr.Dropdown(
                    choices=CORRUPTION_CHOICES,
                    value=CORRUPTION_CHOICES[0],
                    label="Corruption Type",
                    scale=2,
                )
                inject_btn = gr.Button("Inject Corruption", scale=1)
            inject_html = gr.HTML()

            scan_btn = gr.Button("Scan for Damage", variant="primary")
            scan_html = gr.HTML()

            init_btn.click(fn=tab1_init, inputs=[t1_state], outputs=[genome_html, t1_state])
            checkpoint_btn.click(fn=tab1_checkpoint, inputs=[t1_state], outputs=[checkpoint_html, t1_state])
            inject_btn.click(fn=tab1_inject, inputs=[corruption_dd, t1_state], outputs=[inject_html, t1_state])
            scan_btn.click(fn=tab1_scan, inputs=[t1_state], outputs=[scan_html])

        # ------ Tab 2: Repair & Certify ------
        with gr.Tab("Repair & Certify"):
            gr.Markdown(
                "Run the full pipeline: checkpoint, corrupt, scan, repair, "
                "re-scan, and certify. Select a preset scenario."
            )
            with gr.Row():
                preset_dd = gr.Dropdown(
                    choices=list(PIPELINE_PRESETS.keys()),
                    value="Temperature drift",
                    label="Preset Scenario",
                    scale=2,
                )
                pipeline_btn = gr.Button("Run Full Pipeline", variant="primary", scale=1)
            pipeline_html = gr.HTML()

            pipeline_btn.click(
                fn=tab2_pipeline,
                inputs=[preset_dd],
                outputs=[pipeline_html],
            )

        # ------ Tab 3: Repair Memory ------
        with gr.Tab("Repair Memory"):
            gr.Markdown(
                "Inspect HistoneStore markers generated during repair operations. "
                "Each successful repair is stored as an epigenetic marker."
            )
            with gr.Row():
                tag_dd = gr.Dropdown(
                    choices=["all", "repair", "genome_drift", "expression_drift"],
                    value="all",
                    label="Tag Filter",
                    scale=2,
                )
                markers_btn = gr.Button("Retrieve Markers", variant="primary", scale=1)
            markers_html = gr.HTML()

            markers_btn.click(
                fn=tab3_markers,
                inputs=[tag_dd],
                outputs=[markers_html],
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
