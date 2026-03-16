"""
Operon Epistemic Topology — Explorer (Gradio Demo)
===================================================

Three-tab demo:
  1. Topology Explorer — preset diagrams, classification, observation profiles
  2. Theorem Dashboard — error amplification, sequential penalty, speedup, density
  3. Pattern Advisor   — recommend a practical coordination pattern from task constraints

Run locally:
    pip install gradio operon-ai
    python space-epistemic/app.py
"""

import sys
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import advise_topology
from operon_ai.core.denature import SummarizeFilter
from operon_ai.core.epistemic import (
    TopologyClass,
    classify_topology,
    epistemic_partition,
    error_amplification_bound,
    observation_profiles,
    parallel_speedup,
    sequential_penalty,
    tool_density,
)
from operon_ai.core.optics import PrismOptic
from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram


# ── Port helper ──────────────────────────────────────────────────────


def _pt(dt=DataType.JSON, il=IntegrityLabel.VALIDATED):
    return PortType(dt, il)


# ── Preset diagram builders ─────────────────────────────────────────


def _build_independent():
    """3 Independent Workers — no wires, fully parallel."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Summarizer", outputs={"out": _pt()},
        cost=ResourceCost(atp=15, latency_ms=30.0),
        capabilities={Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Translator", outputs={"out": _pt()},
        cost=ResourceCost(atp=20, latency_ms=40.0),
        capabilities={Capability.NET},
    ))
    d.add_module(ModuleSpec(
        "Formatter", outputs={"out": _pt()},
        cost=ResourceCost(atp=10, latency_ms=15.0),
    ))
    return d


def _build_pipeline():
    """4-Stage Pipeline — A -> B -> C -> D sequential chain."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Ingestor", outputs={"out": _pt()},
        cost=ResourceCost(atp=5, latency_ms=10.0),
    ))
    d.add_module(ModuleSpec(
        "Parser", inputs={"in": _pt()}, outputs={"out": _pt()},
        cost=ResourceCost(atp=10, latency_ms=20.0),
    ))
    d.add_module(ModuleSpec(
        "Validator", inputs={"in": _pt()}, outputs={"out": _pt()},
        cost=ResourceCost(atp=15, latency_ms=25.0),
        capabilities={Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Writer", inputs={"in": _pt()},
        cost=ResourceCost(atp=8, latency_ms=12.0),
        capabilities={Capability.WRITE_FS},
    ))
    d.connect("Ingestor", "out", "Parser", "in")
    d.connect("Parser", "out", "Validator", "in")
    d.connect("Validator", "out", "Writer", "in")
    return d


def _build_fan_in():
    """Fan-In Hub (3->1) — three workers feed one aggregator."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Worker1", outputs={"out": _pt()},
        cost=ResourceCost(atp=12, latency_ms=20.0),
        capabilities={Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Worker2", outputs={"out": _pt()},
        cost=ResourceCost(atp=12, latency_ms=20.0),
        capabilities={Capability.NET},
    ))
    d.add_module(ModuleSpec(
        "Worker3", outputs={"out": _pt()},
        cost=ResourceCost(atp=12, latency_ms=20.0),
        capabilities={Capability.EXEC_CODE},
    ))
    d.add_module(ModuleSpec(
        "Aggregator", inputs={"i1": _pt(), "i2": _pt(), "i3": _pt()},
        cost=ResourceCost(atp=8, latency_ms=10.0),
    ))
    d.connect("Worker1", "out", "Aggregator", "i1")
    d.connect("Worker2", "out", "Aggregator", "i2")
    d.connect("Worker3", "out", "Aggregator", "i3")
    return d


def _build_diamond():
    """Diamond (Dispatch->{A,B}->Merge) — hybrid with optic + denature."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Dispatcher", outputs={"o1": _pt(), "o2": _pt()},
        cost=ResourceCost(atp=5, latency_ms=8.0),
    ))
    d.add_module(ModuleSpec(
        "Analyzer", inputs={"in": _pt(DataType.JSON, IntegrityLabel.VALIDATED)},
        outputs={"out": _pt()},
        cost=ResourceCost(atp=25, latency_ms=50.0),
        capabilities={Capability.READ_FS, Capability.EXEC_CODE},
    ))
    d.add_module(ModuleSpec(
        "Enricher", inputs={"in": _pt()}, outputs={"out": _pt()},
        cost=ResourceCost(atp=18, latency_ms=35.0),
        capabilities={Capability.NET},
    ))
    d.add_module(ModuleSpec(
        "Synthesizer", inputs={"i1": _pt(), "i2": _pt()},
        cost=ResourceCost(atp=10, latency_ms=15.0),
    ))
    prism = PrismOptic(accept=frozenset({DataType.JSON, DataType.ERROR}))
    d.connect("Dispatcher", "o1", "Analyzer", "in", optic=prism)
    denature = SummarizeFilter(max_length=200, prefix="[enriched]")
    d.connect("Dispatcher", "o2", "Enricher", "in", denature=denature)
    d.connect("Analyzer", "out", "Synthesizer", "i1")
    d.connect("Enricher", "out", "Synthesizer", "i2")
    return d


def _build_complex():
    """6-Module Complex — richer hybrid with multiple capabilities."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Gateway", outputs={"o1": _pt(), "o2": _pt(), "o3": _pt()},
        cost=ResourceCost(atp=4, latency_ms=5.0),
    ))
    d.add_module(ModuleSpec(
        "Fetcher", inputs={"in": _pt()}, outputs={"out": _pt()},
        cost=ResourceCost(atp=20, latency_ms=60.0),
        capabilities={Capability.NET},
    ))
    d.add_module(ModuleSpec(
        "Executor", inputs={"in": _pt()}, outputs={"out": _pt()},
        cost=ResourceCost(atp=30, latency_ms=80.0),
        capabilities={Capability.EXEC_CODE, Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Reader", inputs={"in": _pt()}, outputs={"out": _pt()},
        cost=ResourceCost(atp=10, latency_ms=15.0),
        capabilities={Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Merger", inputs={"i1": _pt(), "i2": _pt(), "i3": _pt()},
        outputs={"out": _pt()},
        cost=ResourceCost(atp=8, latency_ms=10.0),
    ))
    d.add_module(ModuleSpec(
        "Reporter", inputs={"in": _pt()},
        cost=ResourceCost(atp=5, latency_ms=8.0),
        capabilities={Capability.NET},
    ))
    d.connect("Gateway", "o1", "Fetcher", "in")
    d.connect("Gateway", "o2", "Executor", "in")
    d.connect("Gateway", "o3", "Reader", "in")
    d.connect("Fetcher", "out", "Merger", "i1")
    d.connect("Executor", "out", "Merger", "i2")
    d.connect("Reader", "out", "Merger", "i3")
    d.connect("Merger", "out", "Reporter", "in")
    return d


TOPOLOGY_PRESETS = {
    "3 Independent Workers": {
        "description": "No wires — fully parallel, no coordination.",
        "build_fn": _build_independent,
    },
    "4-Stage Pipeline": {
        "description": "Linear chain: A -> B -> C -> D.",
        "build_fn": _build_pipeline,
    },
    "Fan-In Hub (3->1)": {
        "description": "Three workers feed one aggregator.",
        "build_fn": _build_fan_in,
    },
    "Diamond (Dispatch->{A,B}->Merge)": {
        "description": "Hybrid with optic + denature on parallel branches.",
        "build_fn": _build_diamond,
    },
    "6-Module Complex": {
        "description": "Fan-out -> 3 parallel -> fan-in -> reporter.",
        "build_fn": _build_complex,
    },
}


# ── HTML helpers ─────────────────────────────────────────────────────

_TOPOLOGY_COLORS = {
    TopologyClass.INDEPENDENT: "#22c55e",
    TopologyClass.SEQUENTIAL: "#3b82f6",
    TopologyClass.CENTRALIZED: "#a855f7",
    TopologyClass.HYBRID: "#f97316",
}


def _badge(text, color="#6366f1"):
    return (
        f'<span style="display:inline-block;padding:4px 12px;border-radius:6px;'
        f'background:{color};color:#fff;font-weight:600;margin:2px">'
        f'{text}</span>'
    )


def _topology_badge(tc):
    color = _TOPOLOGY_COLORS.get(tc, "#888")
    return _badge(tc.value.upper(), color)


_PATTERN_COLORS = {
    "single_worker": "#64748b",
    "single_worker_with_reviewer": "#16a34a",
    "specialist_swarm": "#7c3aed",
}


def _pattern_badge(pattern_name):
    label = pattern_name.replace("_", " ").upper()
    color = _PATTERN_COLORS.get(pattern_name, "#6366f1")
    return _badge(label, color)


def _progress_bar(value, max_val, label, color="#6366f1"):
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div style="margin:6px 0">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85em;margin-bottom:2px">'
        f'<span>{label}</span><span style="font-weight:600">{value:.3f}</span></div>'
        f'<div style="background:#e5e7eb;border-radius:4px;height:16px;overflow:hidden">'
        f'<div style="background:{color};height:100%;width:{pct:.1f}%;border-radius:4px">'
        f'</div></div></div>'
    )


def _profile_table_html(profiles):
    rows = []
    for name in sorted(profiles):
        p = profiles[name]
        direct = ", ".join(sorted(p.direct_sources)) or "-"
        trans = ", ".join(sorted(p.transitive_sources)) or "-"
        optic = _badge("Yes", "#8b5cf6") if p.has_optic_filter else _badge("No", "#d1d5db")
        denat = _badge("Yes", "#ec4899") if p.has_denature_filter else _badge("No", "#d1d5db")
        rows.append(
            f"<tr>"
            f"<td style='padding:6px;font-weight:600'>{name}</td>"
            f"<td style='padding:6px'>{direct}</td>"
            f"<td style='padding:6px'>{trans}</td>"
            f"<td style='padding:6px;text-align:center'>{p.observation_width}</td>"
            f"<td style='padding:6px;text-align:center'>{optic}</td>"
            f"<td style='padding:6px;text-align:center'>{denat}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr style='background:#f3f4f6'>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Module</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Direct Sources</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Transitive Sources</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Width</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Optic</th>"
        "<th style='padding:6px;border-bottom:2px solid #d1d5db'>Denature</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )


def _partition_html(partition):
    parts = []
    for i, eq_class in enumerate(partition.equivalence_classes, 1):
        members = ", ".join(sorted(eq_class))
        parts.append(
            f'<div style="display:inline-block;padding:6px 14px;border-radius:8px;'
            f'background:#f0f9ff;border:1px solid #bae6fd;margin:4px">'
            f'<span style="color:#0369a1;font-weight:600">Class {i}:</span> {members}'
            f'</div>'
        )
    return (
        f'<div style="padding:8px">'
        f'<div style="margin-bottom:6px;font-weight:600">'
        f'{len(partition.equivalence_classes)} equivalence class(es)</div>'
        + "".join(parts)
        + '</div>'
    )


def _classification_html(cls):
    tb = _topology_badge(cls.topology_class)
    hub = cls.hub_module or "-"
    return (
        f'<div style="padding:12px;border:2px solid #e5e7eb;border-radius:8px">'
        f'<div style="font-size:1.3em;margin-bottom:10px">{tb}</div>'
        f'<table style="border-collapse:collapse">'
        f'<tr><td style="padding:4px 12px;color:#666">Hub module</td>'
        f'<td style="padding:4px 12px;font-weight:600">{hub}</td></tr>'
        f'<tr><td style="padding:4px 12px;color:#666">Chain length</td>'
        f'<td style="padding:4px 12px;font-weight:600">{cls.chain_length}</td></tr>'
        f'<tr><td style="padding:4px 12px;color:#666">Parallelism width</td>'
        f'<td style="padding:4px 12px;font-weight:600">{cls.parallelism_width}</td></tr>'
        f'<tr><td style="padding:4px 12px;color:#666">Source modules</td>'
        f'<td style="padding:4px 12px;font-weight:600">{cls.num_sources}</td></tr>'
        f'</table></div>'
    )


def _theorem_card_html(title, color, content):
    return (
        f'<div style="padding:14px;border:2px solid {color};border-radius:8px;'
        f'margin:6px;flex:1;min-width:220px">'
        f'<div style="font-weight:700;color:{color};margin-bottom:8px">{title}</div>'
        f'{content}</div>'
    )


def _diagram_text(diagram):
    lines = [f"**Modules:** {len(diagram.modules)} | **Wires:** {len(diagram.wires)}"]
    for name, spec in diagram.modules.items():
        caps = ""
        if spec.capabilities:
            caps = f" [{', '.join(c.value for c in spec.capabilities)}]"
        cost = ""
        if spec.cost:
            cost = f" (ATP={spec.cost.atp}, {spec.cost.latency_ms}ms)"
        lines.append(f"- **{name}**{cost}{caps}")
    for w in diagram.wires:
        extras = []
        if w.optic is not None:
            extras.append("optic")
        if w.denature is not None:
            extras.append("denature")
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        lines.append(f"  {w.src_module}.{w.src_port} -> {w.dst_module}.{w.dst_port}{extra_str}")
    return "\n".join(lines)


# ── Tab 1: Topology Explorer ────────────────────────────────────────


def _run_explorer(preset_name):
    preset = TOPOLOGY_PRESETS.get(preset_name)
    if not preset:
        return "Select a preset.", "", "", ""
    diagram = preset["build_fn"]()
    overview = _diagram_text(diagram)
    cls = classify_topology(diagram)
    cls_html = _classification_html(cls)
    profiles = observation_profiles(diagram)
    prof_html = _profile_table_html(profiles)
    part = epistemic_partition(diagram)
    part_html = _partition_html(part)
    return overview, cls_html, prof_html, part_html


# ── Tab 2: Theorem Dashboard ────────────────────────────────────────


def _run_theorems(preset_name, detection_rate, comm_cost_ratio):
    preset = TOPOLOGY_PRESETS.get(preset_name)
    if not preset:
        return "Select a preset."
    diagram = preset["build_fn"]()

    # Theorem 1: Error Amplification
    eb = error_amplification_bound(diagram, detection_rate=detection_rate)
    t1 = (
        f'<div style="font-size:0.9em;color:#666;margin-bottom:6px">'
        f'n={eb.n_agents} agents, d={eb.detection_rate:.2f}</div>'
        + _progress_bar(eb.independent_bound, eb.independent_bound + 1,
                        "Independent bound", "#ef4444")
        + _progress_bar(eb.centralized_bound, eb.independent_bound + 1,
                        "Centralized bound", "#22c55e")
        + f'<div style="margin-top:6px;font-size:0.85em">'
        f'Amplification ratio: <strong>{eb.amplification_ratio:.2f}x</strong></div>'
    )

    # Theorem 2: Sequential Penalty
    sp = sequential_penalty(diagram, comm_cost_ratio=comm_cost_ratio)
    t2 = (
        f'<div style="font-size:0.9em;color:#666;margin-bottom:6px">'
        f'chain={sp.chain_length}, handoffs={sp.num_handoffs}</div>'
        + _progress_bar(sp.overhead_ratio, 1.0, "Overhead ratio", "#f97316")
        + f'<div style="margin-top:6px;font-size:0.85em">'
        f'Comm cost ratio: <strong>{sp.comm_cost_ratio:.2f}</strong></div>'
    )

    # Theorem 3: Parallel Speedup
    ps = parallel_speedup(diagram)
    t3 = (
        f'<div style="font-size:0.9em;color:#666;margin-bottom:6px">'
        f'{ps.num_subtasks} modules</div>'
        + _progress_bar(ps.speedup, ps.num_subtasks, "Speedup", "#3b82f6")
        + f'<div style="margin-top:6px;font-size:0.85em">'
        f'Total cost: {ps.total_cost.atp} ATP | '
        f'Bottleneck: {ps.max_layer_cost.atp} ATP</div>'
    )

    # Theorem 4: Tool Density
    td = tool_density(diagram)
    t4 = (
        f'<div style="font-size:0.9em;color:#666;margin-bottom:6px">'
        f'{td.total_tools} tools across {td.num_modules} modules</div>'
        + _progress_bar(td.remote_fraction, 1.0, "Remote fraction", "#a855f7")
        + f'<div style="margin-top:6px;font-size:0.85em">'
        f'Tools/module: {td.tools_per_module:.2f} | '
        f'Planning cost: <strong>{td.planning_cost_ratio:.1f}x</strong></div>'
    )

    cards = (
        '<div style="display:flex;flex-wrap:wrap;gap:8px">'
        + _theorem_card_html("Theorem 1: Error Amplification", "#ef4444", t1)
        + _theorem_card_html("Theorem 2: Sequential Penalty", "#f97316", t2)
        + _theorem_card_html("Theorem 3: Parallel Speedup", "#3b82f6", t3)
        + _theorem_card_html("Theorem 4: Tool Density", "#a855f7", t4)
        + '</div>'
    )
    return cards


# ── Tab 3: Pattern Advisor ──────────────────────────────────────────


def _run_advisor(task_shape, num_subtasks, num_tools, error_tolerance):
    task_shape_map = {
        "Sequential / dependent": "sequential",
        "Parallel / decomposable": "parallel",
        "Mixed": "mixed",
    }
    advice = advise_topology(
        task_shape=task_shape_map[task_shape],
        subtask_count=int(num_subtasks),
        tool_count=int(num_tools),
        error_tolerance=float(error_tolerance),
    )
    pattern = _pattern_badge(advice.recommended_pattern)
    tb = _topology_badge(advice.topology)
    return (
        f'<div style="padding:16px;border:2px solid #e5e7eb;border-radius:10px">'
        f'<div style="font-size:1.3em;margin-bottom:12px">Recommended Pattern: {pattern}</div>'
        f'<div style="margin-bottom:8px;color:#374151;font-size:0.95em">'
        f'Suggested API: <code>{advice.suggested_api}</code></div>'
        f'<div style="margin-bottom:12px;color:#374151;line-height:1.6">{advice.rationale}</div>'
        f'<div style="margin-bottom:12px;color:#6b7280;font-size:0.9em">'
        f'Underlying topology class: {tb}</div>'
        f'<div style="margin-top:12px;padding:10px;background:#f9fafb;border-radius:6px;'
        f'font-size:0.85em;color:#666">'
        f'Inputs: {task_shape.lower()}, {int(num_subtasks)} subtasks, '
        f'{int(num_tools)} tools, '
        f'error tolerance={error_tolerance:.2f}'
        f'</div></div>'
    )


# ── Gradio UI ────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Epistemic Topology Explorer") as app:
        gr.Markdown(
            "# Epistemic Topology Explorer\n"
            "Analyze **wiring diagram topology** to predict error amplification, "
            "coordination overhead, and parallelism bounds.\n\n"
            "If you care about a practical answer first, start in **Pattern Advisor**."
        )

        with gr.Tabs():
            # ── Tab 1: Topology Explorer ─────────────────────────────
            with gr.TabItem("Topology Explorer"):
                gr.Markdown(
                    "Select a preset diagram and click **Analyze** to see "
                    "its topology classification, observation profiles, and "
                    "epistemic partition."
                )

                with gr.Row():
                    explorer_preset = gr.Dropdown(
                        choices=list(TOPOLOGY_PRESETS.keys()),
                        value="Diamond (Dispatch->{A,B}->Merge)",
                        label="Preset Diagram",
                        scale=3,
                    )
                    explorer_btn = gr.Button("Analyze", variant="primary", scale=1)

                explorer_overview = gr.Markdown(label="Diagram Overview")
                explorer_cls = gr.HTML(label="Classification")
                explorer_profiles = gr.HTML(label="Observation Profiles")
                explorer_partition = gr.HTML(label="Epistemic Partition")

                explorer_btn.click(
                    fn=_run_explorer,
                    inputs=[explorer_preset],
                    outputs=[explorer_overview, explorer_cls,
                             explorer_profiles, explorer_partition],
                )

            # ── Tab 2: Theorem Dashboard ─────────────────────────────
            with gr.TabItem("Theorem Dashboard"):
                gr.Markdown(
                    "Compute all four theorem bounds for a preset diagram. "
                    "Adjust **detection rate** (Theorem 1) and "
                    "**communication cost ratio** (Theorem 2) to see how "
                    "parameters affect the predictions."
                )

                with gr.Row():
                    theorem_preset = gr.Dropdown(
                        choices=list(TOPOLOGY_PRESETS.keys()),
                        value="Diamond (Dispatch->{A,B}->Merge)",
                        label="Preset Diagram",
                        scale=2,
                    )
                with gr.Row():
                    theorem_det = gr.Slider(
                        0.0, 0.99, value=0.75, step=0.01,
                        label="Detection Rate (d)",
                    )
                    theorem_comm = gr.Slider(
                        0.0, 1.0, value=0.4, step=0.01,
                        label="Comm Cost Ratio",
                    )
                    theorem_btn = gr.Button("Compute Bounds", variant="primary")

                theorem_output = gr.HTML(label="Theorem Results")

                theorem_btn.click(
                    fn=_run_theorems,
                    inputs=[theorem_preset, theorem_det, theorem_comm],
                    outputs=[theorem_output],
                )

            # ── Tab 3: Pattern Advisor ───────────────────────────────
            with gr.TabItem("Pattern Advisor"):
                gr.Markdown(
                    "Describe your task constraints and get a **recommended "
                    "coordination pattern** with rationale. No diagram needed — "
                    "just task properties.\n\n"
                    "This is the shortest path if you want a practical answer "
                    "before digging into observation profiles or theorem cards."
                )

                with gr.Row():
                    adv_shape = gr.Dropdown(
                        choices=[
                            "Sequential / dependent",
                            "Parallel / decomposable",
                            "Mixed",
                        ],
                        value="Sequential / dependent",
                        label="Task Shape",
                    )
                    adv_subtasks = gr.Slider(
                        1, 20, value=5, step=1,
                        label="Number of Subtasks",
                    )
                with gr.Row():
                    adv_tools = gr.Slider(
                        0, 30, value=3, step=1,
                        label="Number of Tools",
                    )
                    adv_error = gr.Slider(
                        0.01, 0.5, value=0.1, step=0.01,
                        label="Error Tolerance",
                    )

                adv_btn = gr.Button("Recommend", variant="primary")
                adv_output = gr.HTML(label="Recommendation")

                adv_btn.click(
                    fn=_run_advisor,
                    inputs=[adv_shape, adv_subtasks, adv_tools, adv_error],
                    outputs=[adv_output],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
