"""
Operon Security Audit -- Unified Security Dashboard
====================================================

Run DNA Repair integrity scans and verify categorical certificates
across all Operon subsystems from a single dashboard.

Run locally:
    pip install gradio
    python space-security-audit/app.py
"""

import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    ATP_Store, DNARepair, Gene, GeneType, Genome, HistoneStore, QuorumSensingBio,
)
from operon_ai.state.mtor import MTORScaler

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "Healthy agent (budget=1000, 3 genes)": {
        "budget": 1000, "num_genes": 3, "corrupt": False, "silence_required": False,
    },
    "Low budget agent (budget=50)": {
        "budget": 50, "num_genes": 3, "corrupt": False, "silence_required": False,
    },
    "Corrupted agent (safety_level mutated)": {
        "budget": 1000, "num_genes": 3, "corrupt": True, "silence_required": False,
    },
    "Silenced required gene": {
        "budget": 1000, "num_genes": 3, "corrupt": False, "silence_required": True,
    },
}

_GENE_POOL = [
    Gene("model", "gpt-4", gene_type=GeneType.STRUCTURAL, required=True),
    Gene("safety_level", "high"),
    Gene("temperature", "0.7"),
    Gene("max_tokens", "4096"),
    Gene("context_window", "128k"),
    Gene("retry_policy", "exponential"),
]

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _badge(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;font-weight:600;">{text}</span>')

def _severity_badge(severity) -> str:
    c = {1: "#3b82f6", 2: "#f59e0b", 3: "#ef4444", 4: "#991b1b"}.get(int(severity), "#6b7280")
    l = {1: "LOW", 2: "MODERATE", 3: "HIGH", 4: "CRITICAL"}.get(int(severity), "?")
    return _badge(l, c)

def _health_badge(label: str) -> str:
    return _badge(label, {"HEALTHY": "#22c55e", "DEGRADED": "#f59e0b", "CRITICAL": "#ef4444"}.get(label, "#6b7280"))

def _cert_badge(holds: bool) -> str:
    return _badge("HOLDS", "#22c55e") if holds else _badge("FAILS", "#ef4444")

def _card(content: str, border: str = "#e5e7eb", thick: bool = False) -> str:
    bw = "2px" if thick else "1px"
    return (f'<div style="padding:16px;border-radius:8px;border:{bw} solid {border};'
            f'background:#f9fafb;margin-bottom:16px;">{content}</div>')

# ---------------------------------------------------------------------------
# Tab 1: Audit Pipeline
# ---------------------------------------------------------------------------

def run_audit(preset_name: str, budget: int, num_genes: int) -> str:
    budget, num_genes = int(budget), int(num_genes)
    preset = PRESETS.get(preset_name, {})

    genes = _GENE_POOL[:num_genes]
    genome = Genome(genes=genes, allow_mutations=True, silent=True)
    atp = ATP_Store(budget=budget, silent=True)
    repair = DNARepair(histone_store=HistoneStore(silent=True), silent=True)
    cp = repair.checkpoint(genome)

    # Apply preset mutations after checkpoint
    if preset.get("corrupt") and "safety_level" in genome._genes:
        genome.mutate("safety_level", "none", reason="preset: corrupt")
    if preset.get("silence_required"):
        for name, g in genome._genes.items():
            if g.required:
                genome.silence_gene(name, reason="preset: silence required")
                break

    damages = repair.scan(genome, cp)
    cert = repair.certify(genome, cp)
    verification = cert.verify()

    # --- DNA Repair scan ---
    if damages:
        rows = "".join(
            f"<tr style='border-bottom:1px solid #e5e7eb;'>"
            f"<td style='padding:6px 8px;'>{d.corruption_type.value}</td>"
            f"<td style='padding:6px 8px;'>{_severity_badge(d.severity)}</td>"
            f"<td style='padding:6px 8px;'><code>{d.location}</code></td>"
            f"<td style='padding:6px 8px;'>{d.description}</td></tr>"
            for d in damages
        )
        scan_html = (
            f"<h3 style='margin:0 0 12px 0;'>DNA Repair Scan</h3>"
            f"<p>Detected <b>{len(damages)}</b> damage report(s):</p>"
            f"<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"<tr style='background:#e5e7eb;'>"
            f"<th style='padding:6px 8px;text-align:left;'>Type</th>"
            f"<th style='padding:6px 8px;text-align:left;'>Severity</th>"
            f"<th style='padding:6px 8px;text-align:left;'>Location</th>"
            f"<th style='padding:6px 8px;text-align:left;'>Description</th>"
            f"</tr>{rows}</table>"
        )
    else:
        scan_html = (
            "<h3 style='margin:0 0 12px 0;'>DNA Repair Scan</h3>"
            "<p style='color:#22c55e;font-weight:600;'>No damage detected. "
            "Genome integrity intact.</p>"
        )

    # --- Certificate status ---
    cert_html = (
        f"<h3 style='margin:0 0 12px 0;'>Certificate Status</h3>"
        f"<p><b>Theorem:</b> {cert.theorem} &mdash; {_cert_badge(verification.holds)}</p>"
        f"<p><b>Conclusion:</b> {cert.conclusion}</p>"
    )

    # --- Overall health ---
    has_critical = any(int(d.severity) >= 4 for d in damages)
    if has_critical or not verification.holds:
        health = "CRITICAL"
    elif damages:
        health = "DEGRADED"
    else:
        health = "HEALTHY"

    bc = {"HEALTHY": "#22c55e", "DEGRADED": "#f59e0b", "CRITICAL": "#ef4444"}[health]
    health_html = (
        f'<span style="font-size:1.3em;font-weight:700;">Overall Health: </span>'
        f'{_health_badge(health)}'
        f'<div style="margin-top:8px;font-size:0.9em;color:#6b7280;">'
        f'Damage reports: {len(damages)} | Certificate: '
        f'{"holds" if verification.holds else "fails"} | Budget: {budget}</div>'
    )

    return _card(scan_html) + _card(cert_html) + _card(health_html, border=bc, thick=True)


def load_audit_preset(name: str):
    p = PRESETS.get(name)
    return (p["budget"], p["num_genes"]) if p else (1000, 3)


# ---------------------------------------------------------------------------
# Tab 2: Certificate Dashboard
# ---------------------------------------------------------------------------

def verify_all_certificates(budget: int) -> str:
    budget = int(budget)

    atp = ATP_Store(budget=budget, silent=True)
    qs = QuorumSensingBio(population_size=10)
    qs.calibrate()
    mtor = MTORScaler(atp_store=atp)

    genes = [
        Gene("model", "gpt-4", gene_type=GeneType.STRUCTURAL, required=True),
        Gene("safety_level", "high"),
    ]
    genome = Genome(genes=genes, allow_mutations=True, silent=True)
    repair = DNARepair(histone_store=HistoneStore(silent=True), silent=True)
    cp = repair.checkpoint(genome)

    certs = [
        ("ATP", atp.certify()),
        ("QuorumSensing", qs.certify()),
        ("MTOR", mtor.certify()),
        ("DNARepair", repair.certify(genome, cp)),
    ]

    all_hold = True
    rows = ""
    for label, cert in certs:
        v = cert.verify()
        if not v.holds:
            all_hold = False

        row_bg = "#f0fdf4" if v.holds else "#fef2f2"
        evidence_parts = []
        for k, val in v.evidence.items():
            evidence_parts.append(f"{k}={val:.4g}" if isinstance(val, float) else f"{k}={val}")
        evidence_str = ", ".join(evidence_parts) or "--"

        rows += (
            f"<tr style='background:{row_bg};border-bottom:1px solid #e5e7eb;'>"
            f"<td style='padding:8px;font-weight:600;'>{cert.theorem}</td>"
            f"<td style='padding:8px;'><code>{cert.source}</code></td>"
            f"<td style='padding:8px;'>{_cert_badge(v.holds)}</td>"
            f"<td style='padding:8px;font-size:0.85em;'>{evidence_str}</td></tr>"
        )

    summary_label = "ALL CERTIFICATES HOLD" if all_hold else "SOME CERTIFICATES FAIL"
    summary_color = "#22c55e" if all_hold else "#ef4444"

    inner = (
        f"<h3 style='margin:0 0 12px 0;'>Certificate Verification Matrix</h3>"
        f"<p style='font-size:0.9em;color:#6b7280;'>ATP budget: {budget}</p>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
        f"<tr style='background:#e5e7eb;'>"
        f"<th style='padding:8px;text-align:left;'>Theorem</th>"
        f"<th style='padding:8px;text-align:left;'>Source</th>"
        f"<th style='padding:8px;text-align:left;'>Status</th>"
        f"<th style='padding:8px;text-align:left;'>Evidence</th>"
        f"</tr>{rows}</table>"
        f"<div style='margin-top:12px;text-align:center;'>"
        f"{_badge(summary_label, summary_color)}</div>"
    )
    return _card(inner)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Security Audit") as app:
        gr.Markdown(
            "# Operon Security Audit\n"
            "Unified security dashboard combining DNA Repair integrity scans "
            "with categorical certificate verification across all Operon "
            "subsystems.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Tab("Audit Pipeline"):
            gr.Markdown(
                "Select an agent preset to simulate different integrity "
                "scenarios, then run a full DNA Repair scan and certificate check."
            )
            with gr.Row():
                audit_preset = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value="Healthy agent (budget=1000, 3 genes)",
                    label="Agent Configuration", scale=2,
                )
                audit_btn = gr.Button("Run Audit", variant="primary", scale=1)
            with gr.Row():
                audit_budget = gr.Slider(
                    minimum=0, maximum=2000, value=1000, step=10, label="ATP Budget",
                )
                audit_genes = gr.Slider(
                    minimum=2, maximum=6, value=3, step=1, label="Number of Genes",
                )
            audit_output = gr.HTML(label="Audit Report")
            audit_btn.click(
                fn=run_audit,
                inputs=[audit_preset, audit_budget, audit_genes],
                outputs=[audit_output],
            )
            audit_preset.change(
                fn=load_audit_preset, inputs=[audit_preset],
                outputs=[audit_budget, audit_genes],
            )

        with gr.Tab("Certificate Dashboard"):
            gr.Markdown(
                "Collect and verify all four categorical certificates. "
                "Adjust the ATP budget to see how it affects the ATP and MTOR certificates."
            )
            with gr.Row():
                cert_budget = gr.Slider(
                    minimum=0, maximum=2000, value=1000, step=10, label="ATP Budget",
                )
                cert_btn = gr.Button("Verify All", variant="primary")
            cert_output = gr.HTML(label="Certificate Matrix")
            cert_btn.click(
                fn=verify_all_certificates, inputs=[cert_budget], outputs=[cert_output],
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
