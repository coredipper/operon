"""
Operon Plasmid Registry -- Horizontal Gene Transfer Demo
========================================================

Dynamic tool acquisition from a searchable registry with capability gating.
Agents can discover, acquire, execute, and release tools at runtime -- like
bacteria exchanging plasmids via horizontal gene transfer.

Run locally:
    pip install gradio
    python space-plasmid/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import sys
from pathlib import Path

import gradio as gr

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai.core.types import Capability
from operon_ai.organelles.mitochondria import Mitochondria
from operon_ai.organelles.plasmid import Plasmid, PlasmidRegistry


# ---------------------------------------------------------------------------
# Preset capability sets
# ---------------------------------------------------------------------------

CAPABILITY_PRESETS: dict[str, set[Capability]] = {
    "No restrictions": set(),
    "READ_FS only": {Capability.READ_FS},
    "NET only": {Capability.NET},
    "NET + READ_FS": {Capability.NET, Capability.READ_FS},
    "EXEC_CODE only": {Capability.EXEC_CODE},
    "All capabilities": {Capability.NET, Capability.READ_FS, Capability.EXEC_CODE, Capability.WRITE_FS},
}

# ---------------------------------------------------------------------------
# Build default registry
# ---------------------------------------------------------------------------

def _build_registry() -> PlasmidRegistry:
    registry = PlasmidRegistry()

    registry.register(Plasmid(
        name="reverse",
        description="Reverse a string",
        func=lambda s: s[::-1],
        tags=frozenset({"text", "utility"}),
        version="1.0.0",
    ))
    registry.register(Plasmid(
        name="word_count",
        description="Count words in a string",
        func=lambda s: len(s.split()),
        tags=frozenset({"text", "analysis"}),
        version="1.0.0",
    ))
    registry.register(Plasmid(
        name="uppercase",
        description="Convert string to uppercase",
        func=lambda s: s.upper(),
        tags=frozenset({"text", "transform"}),
        version="1.0.0",
    ))
    registry.register(Plasmid(
        name="char_count",
        description="Count characters in a string",
        func=lambda s: len(s),
        tags=frozenset({"text", "analysis"}),
        version="1.2.0",
    ))
    registry.register(Plasmid(
        name="fetch_url",
        description="Fetch a URL (simulated)",
        func=lambda url: f"<html>Content from {url}</html>",
        tags=frozenset({"network", "io"}),
        version="1.0.0",
        required_capabilities=frozenset({Capability.NET}),
    ))
    registry.register(Plasmid(
        name="read_file",
        description="Read a file (simulated)",
        func=lambda path: f"Contents of {path}",
        tags=frozenset({"filesystem", "io"}),
        version="1.0.0",
        required_capabilities=frozenset({Capability.READ_FS}),
    ))
    registry.register(Plasmid(
        name="run_script",
        description="Execute a script (simulated)",
        func=lambda code: f"Executed: {code}",
        tags=frozenset({"execution", "code"}),
        version="2.0.0",
        required_capabilities=frozenset({Capability.EXEC_CODE}),
    ))
    return registry


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_registry = _build_registry()
_mito: Mitochondria | None = None
_event_log: list[str] = []


def _reset_agent(caps_label: str) -> str:
    global _mito, _event_log
    _event_log = []

    if caps_label == "No restrictions":
        _mito = Mitochondria(silent=True)
        _event_log.append("Agent created with NO capability restrictions.")
    else:
        caps = CAPABILITY_PRESETS.get(caps_label, set())
        _mito = Mitochondria(allowed_capabilities=caps, silent=True)
        cap_names = sorted(c.value for c in caps) if caps else ["none"]
        _event_log.append(f"Agent created with capabilities: {cap_names}")

    return _render_state()


def _render_state() -> str:
    if _mito is None:
        return '<div style="padding:16px;color:#6b7280;">No agent created yet. Select capabilities and click Create.</div>'

    tools = list(_mito.tools.keys())
    tool_badges = ""
    if tools:
        tool_badges = " ".join(
            f'<span style="background:#8b5cf6;color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;">{t}</span>'
            for t in tools
        )
    else:
        tool_badges = '<span style="color:#9ca3af;font-style:italic;">none</span>'

    caps = _mito.allowed_capabilities
    if caps is None:
        cap_text = "unrestricted"
    elif caps:
        cap_text = ", ".join(sorted(c.value for c in caps))
    else:
        cap_text = "none (empty set)"

    html = (
        f'<div style="padding:16px;border-radius:8px;border:1px solid #d1d5db;background:#f9fafb;">'
        f'<div style="font-weight:700;margin-bottom:8px;">Agent State</div>'
        f'<div style="margin-bottom:4px;"><strong>Capabilities:</strong> {cap_text}</div>'
        f'<div style="margin-bottom:4px;"><strong>Tools ({len(tools)}):</strong> {tool_badges}</div>'
        f'<div style="font-size:0.85em;color:#6b7280;">ROS: {_mito.get_ros_level():.2f} | '
        f'Operations: {_mito._operations_count}</div>'
        f'</div>'
    )
    return html


def _render_log() -> str:
    if not _event_log:
        return ""
    rows = ""
    for i, entry in enumerate(reversed(_event_log[-20:]), 1):
        bg = "#f0fdf4" if "OK" in entry or "Success" in entry else (
            "#fef2f2" if "BLOCKED" in entry or "Error" in entry or "Failed" in entry
            else "#f9fafb"
        )
        rows += f'<div style="padding:6px 10px;background:{bg};border-bottom:1px solid #e5e7eb;font-size:0.9em;">{entry}</div>'
    return f'<div style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;max-height:400px;overflow-y:auto;">{rows}</div>'


# ---------------------------------------------------------------------------
# Registry browsing
# ---------------------------------------------------------------------------

def browse_registry(search_query: str) -> str:
    if search_query.strip():
        results = _registry.search(search_query)
    else:
        results = [_registry.get(item["name"]) for item in _registry.list_available()]

    if not results:
        return '<div style="padding:12px;color:#6b7280;">No plasmids found.</div>'

    rows = ""
    for p in results:
        caps = sorted(c.value for c in p.required_capabilities) if p.required_capabilities else ["none"]
        tags = sorted(p.tags)
        cap_color = "#22c55e" if caps == ["none"] else "#ef4444"
        rows += (
            f'<tr>'
            f'<td style="padding:8px;font-weight:600;">{p.name}</td>'
            f'<td style="padding:8px;">{p.description}</td>'
            f'<td style="padding:8px;">{p.version}</td>'
            f'<td style="padding:8px;">{", ".join(tags)}</td>'
            f'<td style="padding:8px;"><span style="color:{cap_color};">{", ".join(caps)}</span></td>'
            f'</tr>'
        )

    return (
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<tr style="background:#f3f4f6;"><th style="padding:8px;text-align:left;">Name</th>'
        f'<th style="padding:8px;text-align:left;">Description</th>'
        f'<th style="padding:8px;text-align:left;">Version</th>'
        f'<th style="padding:8px;text-align:left;">Tags</th>'
        f'<th style="padding:8px;text-align:left;">Required Caps</th></tr>'
        f'{rows}</table>'
    )


# ---------------------------------------------------------------------------
# Acquire / Release / Execute
# ---------------------------------------------------------------------------

def acquire_plasmid(plasmid_name: str) -> tuple[str, str]:
    if _mito is None:
        _event_log.append("Error: Create an agent first.")
        return _render_state(), _render_log()

    name = plasmid_name.strip()
    if not name:
        _event_log.append("Error: Enter a plasmid name.")
        return _render_state(), _render_log()

    result = _mito.acquire(name, _registry)
    if result.success:
        _event_log.append(f"OK -- Acquired '{name}' successfully.")
    else:
        _event_log.append(f"BLOCKED -- Acquire '{name}' failed: {result.error}")

    return _render_state(), _render_log()


def release_plasmid(plasmid_name: str) -> tuple[str, str]:
    if _mito is None:
        _event_log.append("Error: Create an agent first.")
        return _render_state(), _render_log()

    name = plasmid_name.strip()
    if not name:
        _event_log.append("Error: Enter a tool name to release.")
        return _render_state(), _render_log()

    try:
        _mito.release(name)
        _event_log.append(f"OK -- Released '{name}' (plasmid curing).")
    except ValueError as e:
        _event_log.append(f"Failed -- Release '{name}': {e}")

    return _render_state(), _render_log()


def execute_tool(expression: str) -> tuple[str, str, str]:
    if _mito is None:
        _event_log.append("Error: Create an agent first.")
        return "", _render_state(), _render_log()

    expr = expression.strip()
    if not expr:
        return "", _render_state(), _render_log()

    result = _mito.metabolize(expr)
    if result.success and result.atp:
        result_html = (
            f'<div style="padding:12px;border-radius:8px;border:2px solid #22c55e;background:#f0fdf4;">'
            f'<span style="font-weight:700;color:#16a34a;">Result:</span> '
            f'<span style="font-family:monospace;font-size:1.2em;">{result.atp.value!r}</span>'
            f'</div>'
        )
        _event_log.append(f"OK -- `{expr}` = {result.atp.value!r}")
    else:
        result_html = (
            f'<div style="padding:12px;border-radius:8px;border:2px solid #ef4444;background:#fef2f2;">'
            f'<span style="font-weight:700;color:#dc2626;">Error:</span> '
            f'<span style="font-family:monospace;">{result.error}</span>'
            f'</div>'
        )
        _event_log.append(f"Failed -- `{expr}`: {result.error}")

    return result_html, _render_state(), _render_log()


# ---------------------------------------------------------------------------
# Full scenario demo
# ---------------------------------------------------------------------------

def run_scenario() -> tuple[str, str]:
    global _mito, _event_log
    _event_log = []
    lines: list[str] = []

    lines.append("### Scenario: Capability-Gated Tool Acquisition\n")

    # Step 1: Create restricted agent
    _mito = Mitochondria(allowed_capabilities={Capability.READ_FS}, silent=True)
    _event_log.append("Agent created with capabilities: ['read_fs']")
    lines.append("**Step 1:** Create agent with `READ_FS` capability only.\n")

    # Step 2: Acquire safe tools
    r1 = _mito.acquire("reverse", _registry)
    _event_log.append(f"Acquire 'reverse': {'OK' if r1.success else 'BLOCKED -- ' + str(r1.error)}")
    lines.append(f"**Step 2:** Acquire `reverse` (no caps required): **{'Success' if r1.success else 'Blocked'}**\n")

    r2 = _mito.acquire("word_count", _registry)
    _event_log.append(f"Acquire 'word_count': {'OK' if r2.success else 'BLOCKED -- ' + str(r2.error)}")
    lines.append(f"**Step 3:** Acquire `word_count` (no caps required): **{'Success' if r2.success else 'Blocked'}**\n")

    # Step 3: Try to acquire NET tool
    r3 = _mito.acquire("fetch_url", _registry)
    _event_log.append(f"Acquire 'fetch_url': {'OK' if r3.success else 'BLOCKED -- ' + str(r3.error)}")
    lines.append(f"**Step 4:** Acquire `fetch_url` (requires `NET`): **{'Success' if r3.success else 'Blocked'}**\n")
    if not r3.success:
        lines.append(f"> *{r3.error}*\n")

    # Step 4: Acquire FS tool (allowed)
    r4 = _mito.acquire("read_file", _registry)
    _event_log.append(f"Acquire 'read_file': {'OK' if r4.success else 'BLOCKED -- ' + str(r4.error)}")
    lines.append(f"**Step 5:** Acquire `read_file` (requires `READ_FS`): **{'Success' if r4.success else 'Blocked'}**\n")

    # Step 5: Execute acquired tools
    lines.append("**Step 6:** Execute acquired tools:\n")

    res = _mito.metabolize('reverse("hello world")')
    val = res.atp.value if res.success and res.atp else res.error
    _event_log.append(f"Execute reverse('hello world'): {val}")
    lines.append(f"- `reverse(\"hello world\")` = `{val}`\n")

    res = _mito.metabolize('word_count("the quick brown fox jumps")')
    val = res.atp.value if res.success and res.atp else res.error
    _event_log.append(f"Execute word_count: {val}")
    lines.append(f"- `word_count(\"the quick brown fox jumps\")` = `{val}`\n")

    # Step 6: Release and verify
    _mito.release("reverse")
    _event_log.append("OK -- Released 'reverse' (plasmid curing).")
    lines.append("**Step 7:** Release `reverse` (plasmid curing).\n")

    res = _mito.metabolize('reverse("test")')
    _event_log.append(f"Execute after release: {'Success' if res.success else 'Failed -- ' + str(res.error)}")
    lines.append(f"**Step 8:** Execute `reverse(\"test\")` after release: **{'Success' if res.success else 'Blocked'}** (tool removed)\n")

    # Step 7: Re-acquire
    r5 = _mito.acquire("reverse", _registry)
    _event_log.append(f"Re-acquire 'reverse': {'OK' if r5.success else 'BLOCKED'}")
    lines.append(f"**Step 9:** Re-acquire `reverse`: **{'Success' if r5.success else 'Blocked'}**\n")

    lines.append(f"\n**Final tools:** {list(_mito.tools.keys())}")

    return "\n".join(lines), _render_log()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon Plasmid Registry") as app:
        gr.Markdown(
            "# Operon Plasmid Registry -- Horizontal Gene Transfer\n"
            "Dynamic tool acquisition with capability gating (Paper §6.2, Eq. 12). "
            "Agents discover and absorb tools from a registry, subject to capability "
            "restrictions that prevent privilege escalation.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Tabs():
            # ── Tab 1: Interactive ────────────────────────────────
            with gr.TabItem("Interactive"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Create Agent")
                        caps_dropdown = gr.Dropdown(
                            choices=list(CAPABILITY_PRESETS.keys()),
                            value="READ_FS only",
                            label="Allowed Capabilities",
                        )
                        create_btn = gr.Button("Create Agent", variant="primary")
                        agent_state = gr.HTML(label="Agent State")

                    with gr.Column(scale=2):
                        gr.Markdown("### 2. Registry")
                        search_input = gr.Textbox(
                            label="Search plasmids",
                            placeholder="e.g. text, reverse, network...",
                        )
                        registry_html = gr.HTML(value=browse_registry(""))

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 3. Acquire / Release")
                        plasmid_input = gr.Textbox(
                            label="Plasmid name",
                            placeholder="e.g. reverse",
                        )
                        with gr.Row():
                            acquire_btn = gr.Button("Acquire", variant="primary")
                            release_btn = gr.Button("Release", variant="secondary")
                    with gr.Column():
                        gr.Markdown("### 4. Execute")
                        expr_input = gr.Textbox(
                            label="Expression",
                            placeholder='e.g. reverse("hello")',
                        )
                        exec_btn = gr.Button("Execute", variant="primary")
                        exec_result = gr.HTML()

                gr.Markdown("### Event Log")
                log_html = gr.HTML()

                # Bindings
                create_btn.click(fn=_reset_agent, inputs=[caps_dropdown], outputs=[agent_state])
                search_input.change(fn=browse_registry, inputs=[search_input], outputs=[registry_html])
                search_input.submit(fn=browse_registry, inputs=[search_input], outputs=[registry_html])
                acquire_btn.click(fn=acquire_plasmid, inputs=[plasmid_input], outputs=[agent_state, log_html])
                release_btn.click(fn=release_plasmid, inputs=[plasmid_input], outputs=[agent_state, log_html])
                exec_btn.click(fn=execute_tool, inputs=[expr_input], outputs=[exec_result, agent_state, log_html])
                expr_input.submit(fn=execute_tool, inputs=[expr_input], outputs=[exec_result, agent_state, log_html])

            # ── Tab 2: Full scenario ──────────────────────────────
            with gr.TabItem("Guided Scenario"):
                gr.Markdown(
                    "### Capability-Gated Tool Acquisition\n\n"
                    "Run a complete scenario demonstrating:\n"
                    "- Agent creation with limited capabilities\n"
                    "- Safe tool acquisition (no caps required)\n"
                    "- Blocked acquisition (insufficient capabilities)\n"
                    "- Tool execution, release (plasmid curing), and re-acquisition"
                )
                scenario_btn = gr.Button("Run Scenario", variant="primary")
                scenario_md = gr.Markdown()
                scenario_log = gr.HTML()
                scenario_btn.click(fn=run_scenario, outputs=[scenario_md, scenario_log])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
