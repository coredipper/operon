"""Operon Certificate Framework -- self-verifiable structural guarantees.

Run locally: pip install gradio && python space-certificates/app.py
"""

import json, sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import ATP_Store, QuorumSensingBio
from operon_ai.state.mtor import MTORScaler
from operon_ai.core.certificate import certificate_to_dict, certificate_from_dict


# -- Presets ----------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "(custom)": {"component": "ATP_Store", "atp_budget": 1000, "qs_pop": 10, "mtor_hyst": 0.05},
    "ATP healthy": {"component": "ATP_Store", "atp_budget": 1000, "qs_pop": 10, "mtor_hyst": 0.05},
    "ATP empty (fails)": {"component": "ATP_Store", "atp_budget": 0, "qs_pop": 10, "mtor_hyst": 0.05},
    "QuorumSensing standard": {"component": "QuorumSensingBio", "atp_budget": 1000, "qs_pop": 10, "mtor_hyst": 0.05},
    "QuorumSensing large population": {"component": "QuorumSensingBio", "atp_budget": 1000, "qs_pop": 50, "mtor_hyst": 0.05},
    "MTOR default hysteresis": {"component": "MTORScaler", "atp_budget": 1000, "qs_pop": 10, "mtor_hyst": 0.05},
    "MTOR zero hysteresis (fails)": {"component": "MTORScaler", "atp_budget": 1000, "qs_pop": 10, "mtor_hyst": 0.0},
}

COMPONENTS = ["ATP_Store", "QuorumSensingBio", "MTORScaler"]


# -- Helpers ----------------------------------------------------------------

def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{text}</span>'
    )


def _holds_badge(holds: bool) -> str:
    return _badge("HOLDS", "#22c55e") if holds else _badge("FAILS", "#ef4444")


def _kv_table(d: dict) -> str:
    rows = "".join(
        f"<tr><td style='padding:4px 12px;font-weight:600;'>{k}</td>"
        f"<td style='padding:4px 12px;'>{v}</td></tr>"
        for k, v in d.items()
    )
    return f"<table style='border-collapse:collapse;margin-top:8px;'>{rows}</table>"


def _create_component(name: str, atp_budget: int, qs_pop: int, mtor_hyst: float):
    """Instantiate a component and return (component, certificate)."""
    if name == "ATP_Store":
        comp = ATP_Store(budget=int(atp_budget), silent=True)
        cert = comp.certify()
        return comp, cert
    elif name == "QuorumSensingBio":
        comp = QuorumSensingBio(population_size=int(qs_pop))
        comp.calibrate()
        cert = comp.certify()
        return comp, cert
    else:  # MTORScaler
        atp = ATP_Store(budget=int(atp_budget), silent=True)
        comp = MTORScaler(atp_store=atp, hysteresis=float(mtor_hyst))
        cert = comp.certify()
        return comp, cert


# -- Tab 1: Issue & Verify --------------------------------------------------

def certify_and_verify(component: str, atp_budget, qs_pop, mtor_hyst) -> str:
    try:
        _, cert = _create_component(component, atp_budget, qs_pop, mtor_hyst)
    except Exception as e:
        return f"<p style='color:#ef4444;'>Error: {e}</p>"

    v = cert.verify()
    evidence = dict(v.evidence)

    bc = "#22c55e" if v.holds else "#ef4444"
    ev_fmt = {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in evidence.items()}
    return (
        f'<div style="padding:16px;border-radius:8px;border:2px solid {bc};background:#f9fafb;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        f'<span style="font-size:1.1em;font-weight:700;">Verification:</span>{_holds_badge(v.holds)}</div>'
        f'<div style="margin-bottom:8px;"><b>Theorem:</b> <code>{cert.theorem}</code></div>'
        f'<div style="margin-bottom:8px;"><b>Source:</b> <code>{cert.source}</code></div>'
        f'<div style="margin-bottom:8px;"><b>Conclusion:</b> {cert.conclusion}</div>'
        f'<div style="margin-bottom:4px;"><b>Parameters:</b></div>{_kv_table(dict(cert.parameters))}'
        f'<div style="margin-top:12px;margin-bottom:4px;"><b>Evidence:</b></div>{_kv_table(ev_fmt)}'
        f'</div>'
    )


# -- Tab 2: Tamper Detection ------------------------------------------------

def issue_certificate(component: str, atp_budget, qs_pop, mtor_hyst):
    try:
        _, cert = _create_component(component, atp_budget, qs_pop, mtor_hyst)
    except Exception as e:
        return f"<p style='color:#ef4444;'>Error: {e}</p>", "{}"

    v = cert.verify()
    d = certificate_to_dict(cert)

    html = (
        f'<div style="padding:16px;border-radius:8px;border:2px solid #3b82f6;background:#f9fafb;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        f'<span style="font-size:1.1em;font-weight:700;">Certificate Issued</span>{_holds_badge(v.holds)}</div>'
        f'<div><b>Theorem:</b> <code>{cert.theorem}</code></div>'
        f'<div><b>Conclusion:</b> {cert.conclusion}</div>'
        f'<div style="margin-top:8px;"><b>Parameters:</b></div>{_kv_table(dict(cert.parameters))}</div>'
    )
    return html, json.dumps(d, indent=2)


def tamper_and_reverify(cert_json: str, tamper_value: float) -> str:
    try:
        d = json.loads(cert_json)
    except (json.JSONDecodeError, TypeError):
        return "<p style='color:#ef4444;'>No certificate issued yet. Click 'Issue Certificate' first.</p>"

    # Apply tamper based on theorem type
    theorem = d.get("theorem", "")
    original_params = json.dumps(d["parameters"], indent=2)

    if theorem == "priority_gating":
        d["parameters"]["budget"] = int(tamper_value)
        tamper_desc = f"budget changed to {int(tamper_value)}"
    elif theorem == "no_false_activation":
        d["parameters"]["N"] = int(tamper_value)
        tamper_desc = f"N (population) changed to {int(tamper_value)}"
    elif theorem == "no_oscillation":
        d["parameters"]["hysteresis"] = tamper_value
        tamper_desc = f"hysteresis changed to {tamper_value}"
    else:
        return f"<p style='color:#ef4444;'>Unknown theorem: {theorem}</p>"

    tampered_params = json.dumps(d["parameters"], indent=2)

    try:
        cert2 = certificate_from_dict(d)
        v2 = cert2.verify()
    except Exception as e:
        return f"<p style='color:#ef4444;'>Re-verification error: {e}</p>"

    evidence = dict(v2.evidence)
    bc = "#22c55e" if v2.holds else "#ef4444"
    ev_fmt = {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in evidence.items()}
    pre_orig = f'<pre style="background:#e5e7eb;padding:8px;border-radius:4px;font-size:0.85em;">{original_params}</pre>'
    pre_tamp = f'<pre style="background:#fef2f2;padding:8px;border-radius:4px;font-size:0.85em;">{tampered_params}</pre>'
    return (
        f'<div style="padding:16px;border-radius:8px;border:2px solid {bc};background:#f9fafb;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        f'<span style="font-size:1.1em;font-weight:700;">Re-verification After Tamper:</span>{_holds_badge(v2.holds)}</div>'
        f'<div style="margin-bottom:8px;"><b>Tamper applied:</b> {tamper_desc}</div>'
        f'<div style="display:flex;gap:20px;margin-bottom:12px;">'
        f'<div><b>Original parameters:</b>{pre_orig}</div>'
        f'<div><b>Tampered parameters:</b>{pre_tamp}</div></div>'
        f'<div style="margin-bottom:4px;"><b>Evidence from re-derivation:</b></div>{_kv_table(ev_fmt)}</div>'
    )


# -- Preset loader ----------------------------------------------------------

def load_preset(name: str):
    preset = PRESETS.get(name)
    if not preset:
        return "ATP_Store", 1000, 10, 0.05
    return preset["component"], preset["atp_budget"], preset["qs_pop"], preset["mtor_hyst"]


# -- Gradio UI --------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Certificate Framework") as app:
        gr.Markdown(
            "# Operon Certificate Framework\n"
            "Issue, verify, and tamper-detect **categorical certificates** -- "
            "self-verifiable structural guarantees from ATP_Store, "
            "QuorumSensingBio, and MTORScaler. Certificates use derivation "
            "replay: `verify()` re-derives the guarantee from parameters "
            "rather than trusting a stored boolean.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Tabs():
            # ----- Tab 1: Issue & Verify -----
            with gr.Tab("Issue & Verify"):
                with gr.Row():
                    t1_preset = gr.Dropdown(choices=list(PRESETS.keys()), value="(custom)",
                                            label="Load Preset", scale=2)

                with gr.Row():
                    t1_component = gr.Dropdown(choices=COMPONENTS, value="ATP_Store", label="Component")
                    t1_atp = gr.Slider(minimum=0, maximum=2000, value=1000, step=10, label="ATP Budget")
                    t1_qs = gr.Slider(minimum=2, maximum=50, value=10, step=1, label="QS Population Size")
                    t1_hyst = gr.Slider(minimum=0.0, maximum=0.2, value=0.05, step=0.01, label="MTOR Hysteresis")

                t1_btn = gr.Button("Certify & Verify", variant="primary")
                t1_output = gr.HTML(label="Verification Result")

                t1_btn.click(
                    fn=certify_and_verify,
                    inputs=[t1_component, t1_atp, t1_qs, t1_hyst],
                    outputs=[t1_output],
                )
                t1_preset.change(
                    fn=load_preset,
                    inputs=[t1_preset],
                    outputs=[t1_component, t1_atp, t1_qs, t1_hyst],
                )

            # ----- Tab 2: Tamper Detection -----
            with gr.Tab("Tamper Detection"):
                gr.Markdown(
                    "Issue a certificate, then modify a parameter and re-verify. "
                    "The certificate detects the change via derivation replay."
                )

                with gr.Row():
                    t2_component = gr.Dropdown(choices=COMPONENTS, value="ATP_Store", label="Component")
                    t2_atp = gr.Slider(minimum=0, maximum=2000, value=1000, step=10, label="ATP Budget")
                    t2_qs = gr.Slider(minimum=2, maximum=50, value=10, step=1, label="QS Population Size")
                    t2_hyst = gr.Slider(minimum=0.0, maximum=0.2, value=0.05, step=0.01, label="MTOR Hysteresis")

                t2_issue_btn = gr.Button("Issue Certificate", variant="primary")
                t2_issue_output = gr.HTML(label="Issued Certificate")
                t2_cert_json = gr.Textbox(label="Certificate (JSON)", lines=8, interactive=False)

                t2_issue_btn.click(
                    fn=issue_certificate,
                    inputs=[t2_component, t2_atp, t2_qs, t2_hyst],
                    outputs=[t2_issue_output, t2_cert_json],
                )

                gr.Markdown("### Tamper & Re-verify")
                gr.Markdown(
                    "Move the slider to inject a new parameter value into the "
                    "serialized certificate, then click **Re-verify**. For ATP, "
                    "this drains the budget; for MTOR, sets hysteresis to zero; "
                    "for QS, changes population size."
                )
                t2_tamper = gr.Slider(minimum=0.0, maximum=100.0, value=0.0, step=1.0,
                                     label="Tamper Value (budget/population/hysteresis)")
                t2_reverify_btn = gr.Button("Re-verify After Tamper", variant="secondary")
                t2_tamper_output = gr.HTML(label="Tamper Detection Result")

                t2_reverify_btn.click(
                    fn=tamper_and_reverify,
                    inputs=[t2_cert_json, t2_tamper],
                    outputs=[t2_tamper_output],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
