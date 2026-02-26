"""
Operon Coalgebra — State Machine Explorer (Gradio Demo)
=======================================================

Three-tab demo:
  1. Step-by-Step   — drive a counter coalgebra, inspect trace
  2. Composition    — parallel & sequential machines
  3. Bisimulation   — compare two machines for observational equivalence

Run locally:
    pip install gradio
    python space-coalgebra/app.py
"""

import sys
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    FunctionalCoalgebra,
    ParallelCoalgebra,
    SequentialCoalgebra,
    StateMachine,
    BisimulationResult,
    check_bisimulation,
)

# ── Coalgebra factories ──────────────────────────────────────────────────

COALGEBRA_TYPES = {
    "Counter (state + input)": lambda: FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: s + i,
    ),
    "Doubler (state * 2, ignores input)": lambda: FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: s * 2,
    ),
    "Modular (state + input) mod 10": lambda: FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: (s + i) % 10,
    ),
    "Saturating (clamps at 100)": lambda: FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: min(s + i, 100),
    ),
}

# ── HTML helpers ─────────────────────────────────────────────────────────


def _trace_table_html(trace) -> str:
    """Render a transition trace as an HTML table."""
    if not trace:
        return "<p style='color:#888'>No steps recorded yet.</p>"
    rows = []
    for t in trace:
        rows.append(
            f"<tr>"
            f"<td style='text-align:center'>{t.step}</td>"
            f"<td style='text-align:center'>{t.state_before}</td>"
            f"<td style='text-align:center'>{t.input}</td>"
            f"<td style='text-align:center;font-weight:600'>{t.output}</td>"
            f"<td style='text-align:center'>{t.state_after}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr style='background:#f3f4f6'>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Step</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>State Before</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Input</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Output</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>State After</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )


def _state_badge(label: str, value, color: str = "#6366f1") -> str:
    return (
        f'<span style="display:inline-block;padding:4px 12px;border-radius:6px;'
        f'background:{color};color:#fff;font-weight:600;margin:2px">'
        f"{label}: {value}</span>"
    )


def _bisim_result_html(result: BisimulationResult) -> str:
    if result.equivalent:
        icon = '<span style="color:#22c55e;font-size:1.5em">&#10003;</span>'
        verdict = "Equivalent"
        color = "#22c55e"
    else:
        icon = '<span style="color:#ef4444;font-size:1.5em">&#10007;</span>'
        verdict = "Not Equivalent"
        color = "#ef4444"

    html = (
        f'<div style="padding:12px;border:2px solid {color};border-radius:8px">'
        f'<div style="font-size:1.2em;font-weight:700;margin-bottom:8px">'
        f'{icon} {verdict}</div>'
        f'<div style="color:#666">{result.message}</div>'
        f'<div style="margin-top:6px;color:#888">States explored: {result.states_explored}</div>'
    )
    if result.witness:
        inp, out_a, out_b = result.witness
        html += (
            f'<div style="margin-top:8px;padding:8px;background:#fef2f2;border-radius:4px">'
            f'<strong>Witness:</strong> input={inp}, '
            f'output_a={out_a}, output_b={out_b}</div>'
        )
    html += "</div>"
    return html


# ── Persistent state for Tab 1 ──────────────────────────────────────────

_step_machine: StateMachine | None = None


def _reset_machine(initial_state: int) -> tuple[str, str]:
    """Reset the state machine and return (state_badge, trace_html)."""
    global _step_machine
    coalgebra = FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: s + i,
    )
    _step_machine = StateMachine(state=int(initial_state), coalgebra=coalgebra)
    badge = _state_badge("Current State", _step_machine.state)
    return badge, "<p style='color:#888'>Trace cleared. Ready to step.</p>"


def _do_step(input_val: int) -> tuple[str, str, str]:
    """Step the machine once. Returns (readout, state_badge, trace_html)."""
    global _step_machine
    if _step_machine is None:
        _reset_machine(0)

    output = _step_machine.step(int(input_val))
    readout = _state_badge("Readout (output)", output, "#3b82f6")
    badge = _state_badge("Current State", _step_machine.state)
    trace_html = _trace_table_html(_step_machine.trace)
    return readout, badge, trace_html


def _run_sequence(seq_text: str) -> tuple[str, str, str]:
    """Run comma-separated inputs. Returns (readout, state_badge, trace)."""
    global _step_machine
    if _step_machine is None:
        _reset_machine(0)

    try:
        inputs = [int(x.strip()) for x in seq_text.split(",") if x.strip()]
    except ValueError:
        return (
            '<span style="color:#ef4444">Invalid input — use comma-separated integers</span>',
            _state_badge("Current State", _step_machine.state),
            _trace_table_html(_step_machine.trace),
        )

    if not inputs:
        return (
            '<span style="color:#888">No inputs provided</span>',
            _state_badge("Current State", _step_machine.state),
            _trace_table_html(_step_machine.trace),
        )

    outputs = _step_machine.run(inputs)
    readout = _state_badge("Last Readout", outputs[-1], "#3b82f6")
    badge = _state_badge("Current State", _step_machine.state)
    trace_html = _trace_table_html(_step_machine.trace)
    return readout, badge, trace_html


# ── Tab 2: Composition ──────────────────────────────────────────────────


def _run_composition(
    mode: str,
    init_s1: int,
    init_s2: int,
    seq_text: str,
) -> tuple[str, str]:
    """Run parallel or sequential composition.

    Returns (state_html, trace_html).
    """
    counter = FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: s + i,
    )
    doubler = FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: s * 2,
    )

    try:
        inputs = [int(x.strip()) for x in seq_text.split(",") if x.strip()]
    except ValueError:
        return (
            '<span style="color:#ef4444">Invalid input — use comma-separated integers</span>',
            "",
        )

    if not inputs:
        return ('<span style="color:#888">No inputs provided</span>', "")

    init_s1 = int(init_s1)
    init_s2 = int(init_s2)

    if mode == "Parallel":
        composed = ParallelCoalgebra(first=counter, second=doubler)
        machine = StateMachine(state=(init_s1, init_s2), coalgebra=composed)
    else:
        composed = SequentialCoalgebra(first=counter, second=doubler)
        machine = StateMachine(state=(init_s1, init_s2), coalgebra=composed)

    machine.run(inputs)

    # Build trace table with dual states
    rows = []
    for t in machine.trace:
        s1_before, s2_before = t.state_before
        s1_after, s2_after = t.state_after
        if isinstance(t.output, tuple):
            o1, o2 = t.output
            out_str = f"({o1}, {o2})"
        else:
            out_str = str(t.output)

        rows.append(
            f"<tr>"
            f"<td style='text-align:center'>{t.step}</td>"
            f"<td style='text-align:center'>({s1_before}, {s2_before})</td>"
            f"<td style='text-align:center'>{t.input}</td>"
            f"<td style='text-align:center;font-weight:600'>{out_str}</td>"
            f"<td style='text-align:center'>({s1_after}, {s2_after})</td>"
            f"</tr>"
        )

    trace_html = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr style='background:#f3f4f6'>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Step</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>(S1, S2) Before</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Input</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Output</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>(S1, S2) After</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )

    final_s1, final_s2 = machine.state
    mode_desc = (
        "Both machines receive the same input and evolve independently."
        if mode == "Parallel"
        else "Counter output feeds as input to Doubler."
    )
    state_html = (
        f'<div style="padding:8px">'
        f'<div style="margin-bottom:8px;color:#666">{mode_desc}</div>'
        f'{_state_badge("Counter (S1)", final_s1, "#6366f1")} '
        f'{_state_badge("Doubler (S2)", final_s2, "#f97316")}'
        f"</div>"
    )
    return state_html, trace_html


# ── Tab 3: Bisimulation ─────────────────────────────────────────────────

BISIM_PRESETS = {
    "(custom)": {
        "desc": "Configure machines and inputs manually.",
        "type_a": "Counter (state + input)",
        "init_a": 0,
        "type_b": "Counter (state + input)",
        "init_b": 0,
        "inputs": "1,2,3",
    },
    "Identical machines": {
        "desc": "Two counters starting at 0 — should be equivalent.",
        "type_a": "Counter (state + input)",
        "init_a": 0,
        "type_b": "Counter (state + input)",
        "init_b": 0,
        "inputs": "1,2,3,4,5",
    },
    "Different initial state": {
        "desc": "Same coalgebra but different starting state — diverges immediately.",
        "type_a": "Counter (state + input)",
        "init_a": 0,
        "type_b": "Counter (state + input)",
        "init_b": 5,
        "inputs": "1,2,3",
    },
    "Counter vs Doubler": {
        "desc": "Different coalgebras — diverges when outputs differ.",
        "type_a": "Counter (state + input)",
        "init_a": 0,
        "type_b": "Doubler (state * 2, ignores input)",
        "init_b": 0,
        "inputs": "1,2,3,4",
    },
    "Counter vs Modular": {
        "desc": "Counter vs mod-10 counter — equivalent until state exceeds 10.",
        "type_a": "Counter (state + input)",
        "init_a": 0,
        "type_b": "Modular (state + input) mod 10",
        "init_b": 0,
        "inputs": "1,2,3,4,5",
    },
}


def _load_bisim_preset(name: str) -> tuple[str, int, str, int, str]:
    p = BISIM_PRESETS.get(name, BISIM_PRESETS["(custom)"])
    return p["type_a"], p["init_a"], p["type_b"], p["init_b"], p["inputs"]


def _run_bisimulation(
    type_a: str,
    init_a: int,
    type_b: str,
    init_b: int,
    seq_text: str,
) -> str:
    """Check bisimulation and return result HTML."""
    coal_a = COALGEBRA_TYPES.get(type_a, list(COALGEBRA_TYPES.values())[0])()
    coal_b = COALGEBRA_TYPES.get(type_b, list(COALGEBRA_TYPES.values())[0])()

    machine_a = StateMachine(state=int(init_a), coalgebra=coal_a)
    machine_b = StateMachine(state=int(init_b), coalgebra=coal_b)

    try:
        inputs = [int(x.strip()) for x in seq_text.split(",") if x.strip()]
    except ValueError:
        return '<span style="color:#ef4444">Invalid input — use comma-separated integers</span>'

    result = check_bisimulation(machine_a, machine_b, inputs)
    return _bisim_result_html(result)


# ── Gradio UI ────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Coalgebra Explorer") as app:
        gr.Markdown(
            "# ⚙️ Coalgebra — State Machine Explorer\n"
            "Build, compose, and compare **coalgebraic state machines** "
            "with full transition tracing."
        )

        with gr.Tabs():
            # ── Tab 1: Step-by-Step ──────────────────────────────────
            with gr.TabItem("Step-by-Step"):
                gr.Markdown(
                    "Drive a **counter coalgebra** (`state + input`) one step "
                    "at a time or run a whole sequence. The trace table records "
                    "every transition."
                )

                with gr.Row():
                    init_sl = gr.Slider(
                        -50, 50, value=0, step=1,
                        label="Initial State",
                    )
                    reset_btn = gr.Button("Reset", variant="secondary")

                state_html = gr.HTML(
                    value=_state_badge("Current State", 0),
                    label="Current State",
                )

                with gr.Row():
                    input_sl = gr.Slider(
                        -10, 10, value=1, step=1,
                        label="Input Value",
                    )
                    step_btn = gr.Button("Step", variant="primary")

                readout_html = gr.HTML(label="Readout")

                with gr.Row():
                    seq_tb = gr.Textbox(
                        label="Run Sequence (comma-separated)",
                        placeholder="1,2,3,-1,5",
                        scale=3,
                    )
                    seq_btn = gr.Button("Run Sequence", variant="primary", scale=1)

                trace_html = gr.HTML(
                    value="<p style='color:#888'>Press Reset to initialize, then Step or Run Sequence.</p>",
                    label="Transition Trace",
                )

                # Wire events
                reset_btn.click(
                    fn=_reset_machine,
                    inputs=[init_sl],
                    outputs=[state_html, trace_html],
                )
                step_btn.click(
                    fn=_do_step,
                    inputs=[input_sl],
                    outputs=[readout_html, state_html, trace_html],
                )
                seq_btn.click(
                    fn=_run_sequence,
                    inputs=[seq_tb],
                    outputs=[readout_html, state_html, trace_html],
                )

            # ── Tab 2: Composition ───────────────────────────────────
            with gr.TabItem("Composition"):
                gr.Markdown(
                    "Compose a **Counter** (`state + input`) and a **Doubler** "
                    "(`state * 2`) in parallel or sequentially.\n\n"
                    "- **Parallel**: both receive the same input, evolve independently\n"
                    "- **Sequential**: counter's readout feeds as the doubler's input"
                )

                with gr.Row():
                    comp_mode = gr.Dropdown(
                        choices=["Parallel", "Sequential"],
                        value="Parallel",
                        label="Composition Mode",
                    )
                    comp_s1 = gr.Slider(-20, 20, value=0, step=1, label="Counter Initial State (S1)")
                    comp_s2 = gr.Slider(-20, 20, value=1, step=1, label="Doubler Initial State (S2)")

                with gr.Row():
                    comp_seq = gr.Textbox(
                        label="Input Sequence (comma-separated)",
                        placeholder="1,2,3,4,5",
                        value="1,2,3,4,5",
                        scale=3,
                    )
                    comp_btn = gr.Button("Run", variant="primary", scale=1)

                comp_state_html = gr.HTML(label="Final States")
                comp_trace_html = gr.HTML(label="Transition Trace")

                comp_btn.click(
                    fn=_run_composition,
                    inputs=[comp_mode, comp_s1, comp_s2, comp_seq],
                    outputs=[comp_state_html, comp_trace_html],
                )

            # ── Tab 3: Bisimulation ──────────────────────────────────
            with gr.TabItem("Bisimulation"):
                gr.Markdown(
                    "Check whether two machines produce **identical outputs** "
                    "over a given input sequence. If they diverge, the first "
                    "diverging input is shown as a *witness*."
                )

                with gr.Row():
                    bisim_preset = gr.Dropdown(
                        choices=list(BISIM_PRESETS.keys()),
                        value="Identical machines",
                        label="Preset Scenario",
                        scale=2,
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Machine A")
                        bisim_type_a = gr.Dropdown(
                            choices=list(COALGEBRA_TYPES.keys()),
                            value="Counter (state + input)",
                            label="Coalgebra Type",
                        )
                        bisim_init_a = gr.Slider(
                            -50, 50, value=0, step=1,
                            label="Initial State",
                        )
                    with gr.Column():
                        gr.Markdown("### Machine B")
                        bisim_type_b = gr.Dropdown(
                            choices=list(COALGEBRA_TYPES.keys()),
                            value="Counter (state + input)",
                            label="Coalgebra Type",
                        )
                        bisim_init_b = gr.Slider(
                            -50, 50, value=0, step=1,
                            label="Initial State",
                        )

                with gr.Row():
                    bisim_seq = gr.Textbox(
                        label="Input Sequence (comma-separated)",
                        placeholder="1,2,3,4,5",
                        value="1,2,3,4,5",
                        scale=3,
                    )
                    bisim_btn = gr.Button("Check Bisimulation", variant="primary", scale=1)

                bisim_result_html = gr.HTML(label="Result")

                bisim_preset.change(
                    fn=_load_bisim_preset,
                    inputs=[bisim_preset],
                    outputs=[bisim_type_a, bisim_init_a, bisim_type_b, bisim_init_b, bisim_seq],
                )
                bisim_btn.click(
                    fn=_run_bisimulation,
                    inputs=[bisim_type_a, bisim_init_a, bisim_type_b, bisim_init_b, bisim_seq],
                    outputs=[bisim_result_html],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
