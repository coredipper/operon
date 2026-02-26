"""
Operon Diffusion — Morphogen Gradient Visualizer (Gradio Demo)
==============================================================

Three-tab demo:
  1. Linear Chain     — gradient formation on a line graph
  2. Topologies       — preset graph shapes (star, ring, grid, tree)
  3. Competing Sources — two morphogens diffusing simultaneously

Run locally:
    pip install gradio
    python space-diffusion/app.py
"""

import sys
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    MorphogenType,
    MorphogenSource,
    DiffusionParams,
    DiffusionField,
)

# ── Topology builders ────────────────────────────────────────────────────

NODE_COLORS = [
    "#6366f1", "#3b82f6", "#22c55e", "#eab308",
    "#f97316", "#ef4444", "#ec4899", "#8b5cf6",
]


def _build_linear(n: int) -> dict[str, list[str]]:
    """A → B → C → ... linear chain."""
    nodes = [chr(65 + i) for i in range(n)]
    adj: dict[str, list[str]] = {nd: [] for nd in nodes}
    for i in range(n - 1):
        adj[nodes[i]].append(nodes[i + 1])
        adj[nodes[i + 1]].append(nodes[i])
    return adj


def _build_star(n: int) -> dict[str, list[str]]:
    """Hub 'A' connected to B, C, D, ..."""
    nodes = [chr(65 + i) for i in range(n)]
    adj: dict[str, list[str]] = {nd: [] for nd in nodes}
    hub = nodes[0]
    for spoke in nodes[1:]:
        adj[hub].append(spoke)
        adj[spoke].append(hub)
    return adj


def _build_ring(n: int) -> dict[str, list[str]]:
    """A → B → C → ... → A cycle."""
    nodes = [chr(65 + i) for i in range(n)]
    adj: dict[str, list[str]] = {nd: [] for nd in nodes}
    for i in range(n):
        nxt = (i + 1) % n
        adj[nodes[i]].append(nodes[nxt])
        adj[nodes[nxt]].append(nodes[i])
    # Deduplicate
    for nd in adj:
        adj[nd] = list(dict.fromkeys(adj[nd]))
    return adj


def _build_grid() -> dict[str, list[str]]:
    """2x3 grid: A-B-C / D-E-F."""
    adj: dict[str, list[str]] = {
        "A": ["B", "D"], "B": ["A", "C", "E"], "C": ["B", "F"],
        "D": ["A", "E"], "E": ["B", "D", "F"], "F": ["C", "E"],
    }
    return adj


def _build_binary_tree() -> dict[str, list[str]]:
    """A root, B/C children, D/E under B, F/G under C."""
    adj: dict[str, list[str]] = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"], "E": ["B"],
        "F": ["C"], "G": ["C"],
    }
    return adj


TOPOLOGIES = {
    "Linear": lambda: _build_linear(5),
    "Star": lambda: _build_star(6),
    "Ring": lambda: _build_ring(5),
    "Grid (2x3)": _build_grid,
    "Binary Tree": _build_binary_tree,
}

# ── Visualization helpers ────────────────────────────────────────────────


def _concentration_bars(snapshot: dict[str, dict[str, float]], morphogen_filter: str | None = None) -> str:
    """Render concentration bars per node."""
    if not snapshot:
        return "<p style='color:#888'>No data yet.</p>"

    rows = []
    for i, (node_id, concs) in enumerate(sorted(snapshot.items())):
        color = NODE_COLORS[i % len(NODE_COLORS)]
        node_rows = []
        if morphogen_filter:
            items = [(morphogen_filter, concs.get(morphogen_filter, 0.0))]
        else:
            items = sorted(concs.items()) if concs else [("(none)", 0.0)]

        for mtype, val in items:
            pct = max(0, min(100, val * 100))
            node_rows.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                f'<span style="width:80px;font-size:0.8em;color:#888">{mtype}</span>'
                f'<div style="flex:1;background:#e5e7eb;border-radius:4px;height:16px">'
                f'<div style="width:{pct:.1f}%;background:{color};height:100%;'
                f'border-radius:4px;transition:width 0.3s"></div></div>'
                f'<span style="width:50px;text-align:right;font-size:0.8em">{val:.4f}</span>'
                f'</div>'
            )

        rows.append(
            f'<div style="margin:6px 0;padding:6px;border-left:3px solid {color}">'
            f'<strong style="font-size:0.9em">Node {node_id}</strong>'
            + "".join(node_rows)
            + "</div>"
        )

    return '<div style="padding:4px">' + "".join(rows) + "</div>"


def _snapshot_table(snapshot: dict[str, dict[str, float]]) -> str:
    """Render snapshot as a markdown table."""
    if not snapshot:
        return "No data."

    # Collect all morphogen types
    all_types = sorted({mt for concs in snapshot.values() for mt in concs})
    if not all_types:
        all_types = ["(empty)"]

    header = "| Node | " + " | ".join(all_types) + " |"
    sep = "| :--- | " + " | ".join("---:" for _ in all_types) + " |"
    rows = [header, sep]
    for node_id in sorted(snapshot.keys()):
        concs = snapshot[node_id]
        vals = " | ".join(f"{concs.get(mt, 0.0):.4f}" for mt in all_types)
        rows.append(f"| {node_id} | {vals} |")

    return "\n".join(rows)


def _gradient_level_table(field: DiffusionField, nodes: list[str]) -> str:
    """Show MorphogenGradient level per node via get_local_gradient()."""
    rows = ["| Node | Morphogen | Concentration | Level |", "| :--- | :--- | ---: | :--- |"]
    for nd in sorted(nodes):
        gradient = field.get_local_gradient(nd)
        for mt in MorphogenType:
            val = gradient.get(mt)
            if val > 0:
                level = gradient.get_level(mt)
                rows.append(f"| {nd} | {mt.value} | {val:.4f} | {level} |")
    if len(rows) == 2:
        return "No non-zero concentrations to display."
    return "\n".join(rows)


# ── Tab 1: Linear Chain ─────────────────────────────────────────────────


def run_linear_chain(
    num_nodes: int,
    source_pos: str,
    emission_rate: float,
    diffusion_rate: float,
    decay_rate: float,
    num_steps: int,
) -> tuple[str, str]:
    """Run diffusion on a linear chain.

    Returns (bars_html, step_snapshots_md).
    """
    n = int(num_nodes)
    nodes = [chr(65 + i) for i in range(n)]
    adj = _build_linear(n)

    params = DiffusionParams(
        diffusion_rate=float(diffusion_rate),
        decay_rate=float(decay_rate),
    )
    field = DiffusionField.from_adjacency(adj, params=params)

    source_node = source_pos if source_pos in nodes else nodes[0]
    field.add_source(MorphogenSource(
        node_id=source_node,
        morphogen_type=MorphogenType.COMPLEXITY,
        emission_rate=float(emission_rate),
    ))

    steps = int(num_steps)
    step_rows = ["| Step | " + " | ".join(nodes) + " |"]
    step_rows.append("| ---: | " + " | ".join("---:" for _ in nodes) + " |")

    for s in range(1, steps + 1):
        field.step()
        snap = field.snapshot()
        vals = " | ".join(
            f"{snap.get(nd, {}).get('complexity', 0.0):.4f}" for nd in nodes
        )
        step_rows.append(f"| {s} | {vals} |")

    bars_html = _concentration_bars(field.snapshot(), morphogen_filter="complexity")
    steps_md = "\n".join(step_rows)

    return bars_html, steps_md


# ── Tab 2: Topologies ───────────────────────────────────────────────────


def run_topology(
    topo_name: str,
    source_node: str,
    emission_rate: float,
    diffusion_rate: float,
    decay_rate: float,
    num_steps: int,
) -> tuple[str, str, str]:
    """Run diffusion on a preset topology.

    Returns (bars_html, snapshot_md, gradient_md).
    """
    builder = TOPOLOGIES.get(topo_name, TOPOLOGIES["Linear"])
    adj = builder()

    params = DiffusionParams(
        diffusion_rate=float(diffusion_rate),
        decay_rate=float(decay_rate),
    )
    field = DiffusionField.from_adjacency(adj, params=params)

    nodes = sorted(adj.keys())
    src = source_node if source_node in nodes else nodes[0]
    field.add_source(MorphogenSource(
        node_id=src,
        morphogen_type=MorphogenType.COMPLEXITY,
        emission_rate=float(emission_rate),
    ))

    field.run(int(num_steps))

    bars_html = _concentration_bars(field.snapshot(), morphogen_filter="complexity")
    snap_md = "### Concentration Snapshot\n\n" + _snapshot_table(field.snapshot())
    grad_md = "### Local Gradient Levels\n\n" + _gradient_level_table(field, nodes)

    return bars_html, snap_md, grad_md


def _get_topology_nodes(topo_name: str) -> dict:
    """Return dropdown update with nodes for the selected topology."""
    builder = TOPOLOGIES.get(topo_name, TOPOLOGIES["Linear"])
    adj = builder()
    nodes = sorted(adj.keys())
    return gr.update(choices=nodes, value=nodes[0])


# ── Tab 3: Competing Sources ────────────────────────────────────────────


def run_competing(
    num_nodes: int,
    src1_pos: str,
    src1_type: str,
    src1_rate: float,
    src2_pos: str,
    src2_type: str,
    src2_rate: float,
    diffusion_rate: float,
    decay_rate: float,
    num_steps: int,
) -> tuple[str, str]:
    """Two morphogens competing on a linear chain.

    Returns (bars_html, table_md).
    """
    n = int(num_nodes)
    nodes = [chr(65 + i) for i in range(n)]
    adj = _build_linear(n)

    params = DiffusionParams(
        diffusion_rate=float(diffusion_rate),
        decay_rate=float(decay_rate),
    )
    field = DiffusionField.from_adjacency(adj, params=params)

    mt_map = {mt.value: mt for mt in MorphogenType}
    mt1 = mt_map.get(src1_type, MorphogenType.COMPLEXITY)
    mt2 = mt_map.get(src2_type, MorphogenType.CONFIDENCE)

    s1 = src1_pos if src1_pos in nodes else nodes[0]
    s2 = src2_pos if src2_pos in nodes else nodes[-1]

    field.add_source(MorphogenSource(node_id=s1, morphogen_type=mt1, emission_rate=float(src1_rate)))
    field.add_source(MorphogenSource(node_id=s2, morphogen_type=mt2, emission_rate=float(src2_rate)))

    field.run(int(num_steps))

    bars_html = _concentration_bars(field.snapshot())

    # Build per-morphogen table
    snap = field.snapshot()
    types_present = sorted({mt for concs in snap.values() for mt in concs})
    header = "| Node | " + " | ".join(types_present) + " |"
    sep = "| :--- | " + " | ".join("---:" for _ in types_present) + " |"
    rows = [header, sep]
    for nd in nodes:
        concs = snap.get(nd, {})
        vals = " | ".join(f"{concs.get(t, 0.0):.4f}" for t in types_present)
        rows.append(f"| {nd} | {vals} |")

    table_md = "### Concentration per Morphogen\n\n" + "\n".join(rows)
    return bars_html, table_md


# ── Gradio UI ────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Diffusion Visualizer") as app:
        gr.Markdown(
            "# 🌊 Diffusion — Morphogen Gradient Visualizer\n"
            "Simulate **morphogen diffusion** on graph topologies and watch "
            "concentration gradients form step by step."
        )

        with gr.Tabs():
            # ── Tab 1: Linear Chain ──────────────────────────────────
            with gr.TabItem("Linear Chain"):
                gr.Markdown(
                    "Emit morphogen from one node in a **linear chain** and "
                    "watch the gradient decay with distance."
                )

                with gr.Row():
                    lc_nodes = gr.Slider(3, 8, value=5, step=1, label="Number of Nodes")
                    lc_source = gr.Dropdown(
                        choices=[chr(65 + i) for i in range(8)],
                        value="A",
                        label="Source Node",
                    )

                with gr.Row():
                    lc_emission = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Emission Rate")
                    lc_diffusion = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Diffusion Rate")
                    lc_decay = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="Decay Rate")

                with gr.Row():
                    lc_steps = gr.Slider(1, 50, value=10, step=1, label="Number of Steps")
                    lc_btn = gr.Button("Run Diffusion", variant="primary")

                lc_bars = gr.HTML(label="Final Concentrations")
                lc_timeline = gr.Markdown(label="Step-by-Step Snapshot")

                lc_btn.click(
                    fn=run_linear_chain,
                    inputs=[lc_nodes, lc_source, lc_emission, lc_diffusion, lc_decay, lc_steps],
                    outputs=[lc_bars, lc_timeline],
                )

            # ── Tab 2: Topologies ────────────────────────────────────
            with gr.TabItem("Topologies"):
                gr.Markdown(
                    "Choose a **graph topology** and see how shape affects "
                    "gradient formation. The local gradient level bridges to "
                    "the `MorphogenGradient` API."
                )

                with gr.Row():
                    tp_topo = gr.Dropdown(
                        choices=list(TOPOLOGIES.keys()),
                        value="Star",
                        label="Topology",
                    )
                    tp_source = gr.Dropdown(
                        choices=sorted(_build_star(6).keys()),
                        value="A",
                        label="Source Node",
                    )

                with gr.Row():
                    tp_emission = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Emission Rate")
                    tp_diffusion = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Diffusion Rate")
                    tp_decay = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="Decay Rate")

                with gr.Row():
                    tp_steps = gr.Slider(1, 50, value=15, step=1, label="Number of Steps")
                    tp_btn = gr.Button("Run Diffusion", variant="primary")

                tp_bars = gr.HTML(label="Concentration Bars")
                with gr.Row():
                    with gr.Column():
                        tp_snap = gr.Markdown(label="Snapshot Table")
                    with gr.Column():
                        tp_grad = gr.Markdown(label="Gradient Levels")

                tp_topo.change(
                    fn=_get_topology_nodes,
                    inputs=[tp_topo],
                    outputs=[tp_source],
                )
                tp_btn.click(
                    fn=run_topology,
                    inputs=[tp_topo, tp_source, tp_emission, tp_diffusion, tp_decay, tp_steps],
                    outputs=[tp_bars, tp_snap, tp_grad],
                )

            # ── Tab 3: Competing Sources ─────────────────────────────
            with gr.TabItem("Competing Sources"):
                gr.Markdown(
                    "Place **two different morphogens** at different nodes "
                    "and observe overlapping gradients. Each morphogen type "
                    "diffuses independently."
                )

                with gr.Row():
                    cs_nodes = gr.Slider(3, 8, value=5, step=1, label="Chain Length")

                morphogen_choices = [mt.value for mt in MorphogenType]

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Source 1")
                        cs_src1 = gr.Dropdown(
                            choices=[chr(65 + i) for i in range(8)],
                            value="A",
                            label="Node",
                        )
                        cs_type1 = gr.Dropdown(
                            choices=morphogen_choices,
                            value="complexity",
                            label="Morphogen Type",
                        )
                        cs_rate1 = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Emission Rate")

                    with gr.Column():
                        gr.Markdown("#### Source 2")
                        cs_src2 = gr.Dropdown(
                            choices=[chr(65 + i) for i in range(8)],
                            value="E",
                            label="Node",
                        )
                        cs_type2 = gr.Dropdown(
                            choices=morphogen_choices,
                            value="confidence",
                            label="Morphogen Type",
                        )
                        cs_rate2 = gr.Slider(0.05, 1.0, value=0.3, step=0.05, label="Emission Rate")

                with gr.Row():
                    cs_diffusion = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Diffusion Rate")
                    cs_decay = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="Decay Rate")
                    cs_steps = gr.Slider(1, 50, value=15, step=1, label="Steps")

                cs_btn = gr.Button("Run Competing Diffusion", variant="primary")
                cs_bars = gr.HTML(label="Concentration Bars")
                cs_table = gr.Markdown(label="Per-Morphogen Table")

                cs_btn.click(
                    fn=run_competing,
                    inputs=[
                        cs_nodes, cs_src1, cs_type1, cs_rate1,
                        cs_src2, cs_type2, cs_rate2,
                        cs_diffusion, cs_decay, cs_steps,
                    ],
                    outputs=[cs_bars, cs_table],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
