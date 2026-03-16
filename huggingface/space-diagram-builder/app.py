"""
Operon Diagram Builder — Custom Wiring Diagram Analyzer (Gradio Demo)
=====================================================================

Two-tab demo:
  1. Build & Analyze — text-defined diagrams with full epistemic analysis
  2. Compare Topologies — side-by-side comparison of two diagrams

Module format:  Name:atp:latency[:cap1,cap2]
Wire format:    Src.port -> Dst.port[:denature|:optic]

Run locally:
    pip install gradio operon-ai
    python space-diagram-builder/app.py
"""

import sys
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

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


# ── Constants ────────────────────────────────────────────────────────

_DEFAULT_PT = PortType(DataType.JSON, IntegrityLabel.VALIDATED)

_CAP_MAP = {
    "read_fs": Capability.READ_FS,
    "write_fs": Capability.WRITE_FS,
    "net": Capability.NET,
    "exec_code": Capability.EXEC_CODE,
    "money": Capability.MONEY,
    "email_send": Capability.EMAIL_SEND,
}

_TOPOLOGY_COLORS = {
    TopologyClass.INDEPENDENT: "#22c55e",
    TopologyClass.SEQUENTIAL: "#3b82f6",
    TopologyClass.CENTRALIZED: "#a855f7",
    TopologyClass.HYBRID: "#f97316",
}


# ── Presets ──────────────────────────────────────────────────────────

PRESETS = {
    "Reviewer Gate": {
        "modules": (
            "Executor:10:25\n"
            "Reviewer:8:20\n"
            "Sink:2:5"
        ),
        "wires": (
            "Executor.out -> Sink.i1\n"
            "Reviewer.out -> Sink.i2"
        ),
    },
    "Specialist Swarm": {
        "modules": (
            "Legal:10:20\n"
            "Security:10:20\n"
            "Finance:10:20\n"
            "Coordinator:6:10"
        ),
        "wires": (
            "Legal.out -> Coordinator.i1\n"
            "Security.out -> Coordinator.i2\n"
            "Finance.out -> Coordinator.i3"
        ),
    },
    "Diamond Pipeline": {
        "modules": (
            "Dispatcher:5:8\n"
            "Analyzer:25:50:read_fs,exec_code\n"
            "Enricher:18:35:net\n"
            "Synthesizer:10:15"
        ),
        "wires": (
            "Dispatcher.o1 -> Analyzer.in:optic\n"
            "Dispatcher.o2 -> Enricher.in:denature\n"
            "Analyzer.out -> Synthesizer.i1\n"
            "Enricher.out -> Synthesizer.i2"
        ),
    },
    "Microservices Fan-In": {
        "modules": (
            "AuthService:10:20:net\n"
            "DataService:15:30:read_fs\n"
            "CacheService:5:5\n"
            "Gateway:8:10"
        ),
        "wires": (
            "AuthService.out -> Gateway.i1\n"
            "DataService.out -> Gateway.i2\n"
            "CacheService.out -> Gateway.i3"
        ),
    },
    "Linear Chain": {
        "modules": (
            "Ingest:5:10\n"
            "Parse:10:20\n"
            "Validate:15:25:read_fs\n"
            "Store:8:12:write_fs"
        ),
        "wires": (
            "Ingest.out -> Parse.in\n"
            "Parse.out -> Validate.in\n"
            "Validate.out -> Store.in"
        ),
    },
    "(Custom)": {
        "modules": "",
        "wires": "",
    },
}


# ── Parser ───────────────────────────────────────────────────────────


def _parse_modules(text):
    """Parse module definitions.

    Format: Name:atp:latency[:cap1,cap2]
    Returns dict of {name: ModuleSpec} or raises ValueError.
    """
    modules = {}
    for line_num, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"Line {line_num}: expected 'Name:atp:latency[:caps]', got '{line}'"
            )
        name = parts[0].strip()
        try:
            atp = int(parts[1].strip())
            latency = float(parts[2].strip())
        except ValueError:
            raise ValueError(
                f"Line {line_num}: atp must be int, latency must be number in '{line}'"
            )
        caps = set()
        if len(parts) >= 4 and parts[3].strip():
            for cap_str in parts[3].strip().split(","):
                cap_str = cap_str.strip().lower()
                if cap_str in _CAP_MAP:
                    caps.add(_CAP_MAP[cap_str])
                else:
                    raise ValueError(
                        f"Line {line_num}: unknown capability '{cap_str}'. "
                        f"Valid: {', '.join(_CAP_MAP.keys())}"
                    )
        modules[name] = {
            "atp": atp, "latency": latency,
            "capabilities": caps if caps else None,
        }
    return modules


def _parse_wires(text, module_names):
    """Parse wire definitions.

    Format: Src.port -> Dst.port[:denature|:optic]
    Returns list of (src_mod, src_port, dst_mod, dst_port, modifier).
    """
    wires = []
    for line_num, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Check for modifier suffix
        modifier = None
        if line.endswith(":denature"):
            modifier = "denature"
            line = line[:-9].strip()
        elif line.endswith(":optic"):
            modifier = "optic"
            line = line[:-6].strip()

        if " -> " not in line:
            raise ValueError(
                f"Line {line_num}: expected 'Src.port -> Dst.port', got '{line}'"
            )
        src_part, dst_part = line.split(" -> ", 1)
        if "." not in src_part or "." not in dst_part:
            raise ValueError(
                f"Line {line_num}: ports must be 'Module.port', got '{src_part} -> {dst_part}'"
            )
        src_mod, src_port = src_part.rsplit(".", 1)
        dst_mod, dst_port = dst_part.rsplit(".", 1)
        src_mod, dst_mod = src_mod.strip(), dst_mod.strip()
        src_port, dst_port = src_port.strip(), dst_port.strip()

        if src_mod not in module_names:
            raise ValueError(f"Line {line_num}: unknown source module '{src_mod}'")
        if dst_mod not in module_names:
            raise ValueError(f"Line {line_num}: unknown destination module '{dst_mod}'")

        wires.append((src_mod, src_port, dst_mod, dst_port, modifier))
    return wires


def _build_diagram(modules_text, wires_text):
    """Build a WiringDiagram from text definitions.

    Returns (diagram, None) on success or (None, error_html) on failure.
    """
    try:
        if not modules_text.strip():
            raise ValueError("No modules defined.")
        mod_defs = _parse_modules(modules_text)
        wire_defs = _parse_wires(wires_text, set(mod_defs.keys())) if wires_text.strip() else []
    except ValueError as e:
        return None, (
            f'<div style="padding:12px;background:#fef2f2;border:2px solid #fca5a5;'
            f'border-radius:8px;color:#991b1b">'
            f'<strong>Parse error:</strong> {e}</div>'
        )

    # Collect ports needed per module
    outputs_needed: dict[str, set[str]] = {name: set() for name in mod_defs}
    inputs_needed: dict[str, set[str]] = {name: set() for name in mod_defs}
    for src_mod, src_port, dst_mod, dst_port, _ in wire_defs:
        outputs_needed[src_mod].add(src_port)
        inputs_needed[dst_mod].add(dst_port)

    d = WiringDiagram()
    for name, info in mod_defs.items():
        inputs = {p: _DEFAULT_PT for p in inputs_needed[name]}
        outputs = {p: _DEFAULT_PT for p in outputs_needed[name]}
        d.add_module(ModuleSpec(
            name,
            inputs=inputs or {},
            outputs=outputs or {},
            cost=ResourceCost(atp=info["atp"], latency_ms=info["latency"]),
            capabilities=info["capabilities"],
        ))

    for src_mod, src_port, dst_mod, dst_port, modifier in wire_defs:
        kwargs = {}
        if modifier == "optic":
            kwargs["optic"] = PrismOptic(accept=frozenset({DataType.JSON, DataType.ERROR}))
        elif modifier == "denature":
            kwargs["denature"] = SummarizeFilter(max_length=200, prefix="[filtered]")
        d.connect(src_mod, src_port, dst_mod, dst_port, **kwargs)

    return d, None


# ── HTML helpers ─────────────────────────────────────────────────────


def _badge(text, color="#6366f1"):
    return (
        f'<span style="display:inline-block;padding:4px 12px;border-radius:6px;'
        f'background:{color};color:#fff;font-weight:600;margin:2px">'
        f'{text}</span>'
    )


def _topology_badge(tc):
    color = _TOPOLOGY_COLORS.get(tc, "#888")
    return _badge(tc.value.upper(), color)


def _progress_bar(value, max_val, label, color="#6366f1"):
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div style="margin:4px 0">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85em;margin-bottom:2px">'
        f'<span>{label}</span><span style="font-weight:600">{value:.3f}</span></div>'
        f'<div style="background:#e5e7eb;border-radius:4px;height:14px;overflow:hidden">'
        f'<div style="background:{color};height:100%;width:{pct:.1f}%;border-radius:4px">'
        f'</div></div></div>'
    )


def _profile_table_html(profiles):
    rows = []
    for name in sorted(profiles):
        p = profiles[name]
        direct = ", ".join(sorted(p.direct_sources)) or "-"
        trans = ", ".join(sorted(p.transitive_sources)) or "-"
        optic = "Yes" if p.has_optic_filter else "-"
        denat = "Yes" if p.has_denature_filter else "-"
        rows.append(
            f"<tr>"
            f"<td style='padding:4px 8px;font-weight:600'>{name}</td>"
            f"<td style='padding:4px 8px'>{direct}</td>"
            f"<td style='padding:4px 8px'>{trans}</td>"
            f"<td style='padding:4px 8px;text-align:center'>{p.observation_width}</td>"
            f"<td style='padding:4px 8px;text-align:center'>{optic}</td>"
            f"<td style='padding:4px 8px;text-align:center'>{denat}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:0.9em'>"
        "<tr style='background:#f3f4f6'>"
        "<th style='padding:4px 8px;border-bottom:2px solid #d1d5db'>Module</th>"
        "<th style='padding:4px 8px;border-bottom:2px solid #d1d5db'>Direct</th>"
        "<th style='padding:4px 8px;border-bottom:2px solid #d1d5db'>Transitive</th>"
        "<th style='padding:4px 8px;border-bottom:2px solid #d1d5db'>Width</th>"
        "<th style='padding:4px 8px;border-bottom:2px solid #d1d5db'>Optic</th>"
        "<th style='padding:4px 8px;border-bottom:2px solid #d1d5db'>Denature</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )


def _full_analysis_html(diagram, detection_rate, comm_cost_ratio):
    """Run full analysis and return combined HTML."""
    cls = classify_topology(diagram)
    tb = _topology_badge(cls.topology_class)
    hub = cls.hub_module or "-"

    profiles = observation_profiles(diagram)
    prof_html = _profile_table_html(profiles)

    part = epistemic_partition(diagram)
    part_parts = []
    for i, eq_class in enumerate(part.equivalence_classes, 1):
        members = ", ".join(sorted(eq_class))
        part_parts.append(f'<span style="margin-right:12px">Class {i}: [{members}]</span>')

    eb = error_amplification_bound(diagram, detection_rate=detection_rate)
    sp = sequential_penalty(diagram, comm_cost_ratio=comm_cost_ratio)
    ps = parallel_speedup(diagram)
    td = tool_density(diagram)

    return (
        f'<div style="padding:12px">'
        # Classification
        f'<div style="margin-bottom:16px;padding:12px;border:2px solid #e5e7eb;border-radius:8px">'
        f'<div style="font-size:1.2em;margin-bottom:8px">Topology: {tb}</div>'
        f'<div style="font-size:0.9em;color:#666">'
        f'Hub: {hub} | Chain: {cls.chain_length} | '
        f'Width: {cls.parallelism_width} | Sources: {cls.num_sources}</div>'
        f'</div>'
        # Profiles
        f'<div style="margin-bottom:16px">'
        f'<div style="font-weight:700;margin-bottom:6px">Observation Profiles</div>'
        f'{prof_html}</div>'
        # Partition
        f'<div style="margin-bottom:16px">'
        f'<div style="font-weight:700;margin-bottom:6px">'
        f'Epistemic Partition ({len(part.equivalence_classes)} classes)</div>'
        f'<div>{"".join(part_parts)}</div></div>'
        # Theorems
        f'<div style="display:flex;flex-wrap:wrap;gap:8px">'
        f'<div style="flex:1;min-width:200px;padding:10px;border:1px solid #fca5a5;border-radius:6px">'
        f'<div style="font-weight:700;color:#ef4444;font-size:0.9em">T1: Error Amplification</div>'
        f'<div style="font-size:0.85em;margin-top:4px">'
        f'Independent: {eb.independent_bound} | Centralized: {eb.centralized_bound:.2f} | '
        f'Ratio: {eb.amplification_ratio:.2f}x</div></div>'
        f'<div style="flex:1;min-width:200px;padding:10px;border:1px solid #fdba74;border-radius:6px">'
        f'<div style="font-weight:700;color:#f97316;font-size:0.9em">T2: Sequential Penalty</div>'
        f'<div style="font-size:0.85em;margin-top:4px">'
        f'Chain: {sp.chain_length} | Handoffs: {sp.num_handoffs} | '
        f'Overhead: {sp.overhead_ratio:.4f}</div></div>'
        f'<div style="flex:1;min-width:200px;padding:10px;border:1px solid #93c5fd;border-radius:6px">'
        f'<div style="font-weight:700;color:#3b82f6;font-size:0.9em">T3: Parallel Speedup</div>'
        f'<div style="font-size:0.85em;margin-top:4px">'
        f'Speedup: {ps.speedup:.2f}x | Total: {ps.total_cost.atp} ATP | '
        f'Bottleneck: {ps.max_layer_cost.atp} ATP</div></div>'
        f'<div style="flex:1;min-width:200px;padding:10px;border:1px solid #c4b5fd;border-radius:6px">'
        f'<div style="font-weight:700;color:#a855f7;font-size:0.9em">T4: Tool Density</div>'
        f'<div style="font-size:0.85em;margin-top:4px">'
        f'Tools: {td.total_tools} | Modules: {td.num_modules} | '
        f'Planning: {td.planning_cost_ratio:.1f}x</div></div>'
        f'</div></div>'
    )


# ── Tab 1: Build & Analyze ──────────────────────────────────────────


def _load_preset(preset_name):
    p = PRESETS.get(preset_name, PRESETS["(Custom)"])
    return p["modules"], p["wires"]


def _run_build(modules_text, wires_text, detection_rate, comm_cost_ratio):
    diagram, error = _build_diagram(modules_text, wires_text)
    if error:
        return error
    return _full_analysis_html(diagram, detection_rate, comm_cost_ratio)


# ── Tab 2: Compare Topologies ────────────────────────────────────────


def _run_compare(
    modules_a, wires_a, modules_b, wires_b,
    detection_rate, comm_cost_ratio,
):
    diag_a, err_a = _build_diagram(modules_a, wires_a)
    diag_b, err_b = _build_diagram(modules_b, wires_b)

    if err_a or err_b:
        parts = []
        if err_a:
            parts.append(f'<div style="margin-bottom:8px"><strong>Diagram A:</strong> {err_a}</div>')
        if err_b:
            parts.append(f'<div><strong>Diagram B:</strong> {err_b}</div>')
        return "".join(parts)

    cls_a = classify_topology(diag_a)
    cls_b = classify_topology(diag_b)
    eb_a = error_amplification_bound(diag_a, detection_rate=detection_rate)
    eb_b = error_amplification_bound(diag_b, detection_rate=detection_rate)
    sp_a = sequential_penalty(diag_a, comm_cost_ratio=comm_cost_ratio)
    sp_b = sequential_penalty(diag_b, comm_cost_ratio=comm_cost_ratio)
    ps_a = parallel_speedup(diag_a)
    ps_b = parallel_speedup(diag_b)
    td_a = tool_density(diag_a)
    td_b = tool_density(diag_b)

    def _row(label, val_a, val_b, fmt="{}", highlight=False):
        a_str = fmt.format(val_a)
        b_str = fmt.format(val_b)
        style = "font-weight:600;" if highlight else ""
        return (
            f"<tr>"
            f"<td style='padding:4px 10px;{style}'>{label}</td>"
            f"<td style='padding:4px 10px;text-align:center;{style}'>{a_str}</td>"
            f"<td style='padding:4px 10px;text-align:center;{style}'>{b_str}</td>"
            f"</tr>"
        )

    tb_a = _topology_badge(cls_a.topology_class)
    tb_b = _topology_badge(cls_b.topology_class)

    rows = [
        f"<tr><td style='padding:4px 10px;font-weight:700'>Topology</td>"
        f"<td style='padding:4px 10px;text-align:center'>{tb_a}</td>"
        f"<td style='padding:4px 10px;text-align:center'>{tb_b}</td></tr>",
        _row("Modules", len(diag_a.modules), len(diag_b.modules)),
        _row("Wires", len(diag_a.wires), len(diag_b.wires)),
        _row("Chain length", cls_a.chain_length, cls_b.chain_length),
        _row("Parallelism width", cls_a.parallelism_width, cls_b.parallelism_width),
        _row("Hub", cls_a.hub_module or "-", cls_b.hub_module or "-"),
        _row("T1: Error (indep)", eb_a.independent_bound, eb_b.independent_bound),
        _row("T1: Error (central)", eb_a.centralized_bound, eb_b.centralized_bound, "{:.2f}"),
        _row("T2: Overhead", sp_a.overhead_ratio, sp_b.overhead_ratio, "{:.4f}"),
        _row("T3: Speedup", ps_a.speedup, ps_b.speedup, "{:.2f}x", highlight=True),
        _row("T4: Planning cost", td_a.planning_cost_ratio, td_b.planning_cost_ratio, "{:.1f}x"),
    ]

    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr style='background:#f3f4f6'>"
        "<th style='padding:6px 10px;border-bottom:2px solid #d1d5db'>Metric</th>"
        "<th style='padding:6px 10px;border-bottom:2px solid #d1d5db;text-align:center'>Diagram A</th>"
        "<th style='padding:6px 10px;border-bottom:2px solid #d1d5db;text-align:center'>Diagram B</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )


# ── Gradio UI ────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Diagram Builder") as app:
        gr.Markdown(
            "# Operon Diagram Builder\n"
            "Build **custom wiring diagrams** via text and get full "
            "structural + epistemic analysis.\n\n"
            "If you want to start from something practical, pick a pattern preset "
            "like **Reviewer Gate** or **Specialist Swarm**, then inspect or edit "
            "the generated diagram."
        )

        with gr.Tabs():
            # ── Tab 1: Build & Analyze ───────────────────────────────
            with gr.TabItem("Build & Analyze"):
                gr.Markdown(
                    "Define modules as `Name:atp:latency[:cap1,cap2]` and "
                    "wires as `Src.port -> Dst.port[:denature|:optic]`.\n\n"
                    "Practical path: start from a pattern preset, see the result, "
                    "then edit the text only if you need more control."
                )

                with gr.Row():
                    build_preset = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Reviewer Gate",
                        label="Preset / Starting Pattern",
                        scale=1,
                    )

                with gr.Row():
                    build_modules = gr.Textbox(
                        label="Module Definitions (one per line)",
                        lines=6,
                        value=PRESETS["Reviewer Gate"]["modules"],
                        placeholder="Name:atp:latency[:cap1,cap2]",
                    )
                    build_wires = gr.Textbox(
                        label="Wire Definitions (one per line)",
                        lines=6,
                        value=PRESETS["Reviewer Gate"]["wires"],
                        placeholder="Src.port -> Dst.port[:denature|:optic]",
                    )

                with gr.Row():
                    build_det = gr.Slider(
                        0.0, 0.99, value=0.75, step=0.01,
                        label="Detection Rate (Theorem 1)",
                    )
                    build_comm = gr.Slider(
                        0.0, 1.0, value=0.4, step=0.01,
                        label="Comm Cost Ratio (Theorem 2)",
                    )
                    build_btn = gr.Button("Build & Analyze", variant="primary")

                build_output = gr.HTML(label="Analysis Results")

                build_preset.change(
                    fn=_load_preset,
                    inputs=[build_preset],
                    outputs=[build_modules, build_wires],
                )
                build_btn.click(
                    fn=_run_build,
                    inputs=[build_modules, build_wires, build_det, build_comm],
                    outputs=[build_output],
                )

            # ── Tab 2: Compare Topologies ────────────────────────────
            with gr.TabItem("Compare Topologies"):
                gr.Markdown(
                    "Define two diagrams side by side and compare their "
                    "classification and theorem values. A useful starting point "
                    "is to compare a practical pattern preset against a more "
                    "naive structure."
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Diagram A")
                        cmp_modules_a = gr.Textbox(
                            label="Modules",
                            lines=5,
                            value=PRESETS["Reviewer Gate"]["modules"],
                        )
                        cmp_wires_a = gr.Textbox(
                            label="Wires",
                            lines=4,
                            value=PRESETS["Reviewer Gate"]["wires"],
                        )
                    with gr.Column():
                        gr.Markdown("### Diagram B")
                        cmp_modules_b = gr.Textbox(
                            label="Modules",
                            lines=5,
                            value=PRESETS["Linear Chain"]["modules"],
                        )
                        cmp_wires_b = gr.Textbox(
                            label="Wires",
                            lines=4,
                            value=PRESETS["Linear Chain"]["wires"],
                        )

                with gr.Row():
                    cmp_det = gr.Slider(
                        0.0, 0.99, value=0.75, step=0.01,
                        label="Detection Rate",
                    )
                    cmp_comm = gr.Slider(
                        0.0, 1.0, value=0.4, step=0.01,
                        label="Comm Cost Ratio",
                    )
                    cmp_btn = gr.Button("Compare", variant="primary")

                cmp_output = gr.HTML(label="Comparison")

                cmp_btn.click(
                    fn=_run_compare,
                    inputs=[cmp_modules_a, cmp_wires_a,
                            cmp_modules_b, cmp_wires_b,
                            cmp_det, cmp_comm],
                    outputs=[cmp_output],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
