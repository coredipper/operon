"""Operon Adaptive Assembly Dashboard — template selection, assembly, and experience pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gradio as gr


# ---------------------------------------------------------------------------
# Inline simulation types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskFingerprint:
    task_shape: str
    tool_count: int
    subtask_count: int
    required_roles: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatternTemplate:
    template_id: str
    name: str
    topology: str
    stage_count: int
    fingerprint: TaskFingerprint
    tags: tuple[str, ...] = ()
    success_rate: float | None = None


@dataclass
class ExperienceRecord:
    stage_name: str
    signal_category: str
    intervention_kind: str
    outcome_success: bool


# ---------------------------------------------------------------------------
# Preset library
# ---------------------------------------------------------------------------

TEMPLATES = [
    PatternTemplate("t1", "Sequential Review Pipeline", "reviewer_gate", 2,
                    TaskFingerprint("sequential", 2, 2, ("writer", "reviewer")),
                    ("content", "review")),
    PatternTemplate("t2", "Enterprise Analysis Organism", "skill_organism", 3,
                    TaskFingerprint("sequential", 4, 3, ("researcher", "strategist")),
                    ("enterprise", "analysis")),
    PatternTemplate("t3", "Research Swarm", "specialist_swarm", 4,
                    TaskFingerprint("parallel", 5, 4, ("analyst", "writer")),
                    ("research",)),
    PatternTemplate("t4", "Single Worker", "single_worker", 1,
                    TaskFingerprint("sequential", 1, 1, ("worker",)),
                    ("simple",)),
]


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def _score(tmpl: PatternTemplate, fp: TaskFingerprint) -> float:
    shape = 1.0 if tmpl.fingerprint.task_shape == fp.task_shape else 0.0
    tool_prox = 1.0 / (1.0 + abs(tmpl.fingerprint.tool_count - fp.tool_count))
    sub_prox = 1.0 / (1.0 + abs(tmpl.fingerprint.subtask_count - fp.subtask_count))
    role_sim = _jaccard(tmpl.fingerprint.required_roles, fp.required_roles)
    tag_sim = _jaccard(tmpl.tags, fp.tags)
    sr = tmpl.success_rate if tmpl.success_rate is not None else 0.5
    return 0.30 * shape + 0.15 * tool_prox + 0.15 * sub_prox + 0.20 * role_sim + 0.10 * tag_sim + 0.10 * sr


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "Enterprise Analysis": TaskFingerprint("sequential", 3, 3, ("researcher", "strategist"), ("enterprise",)),
    "Content Review": TaskFingerprint("sequential", 2, 2, ("writer", "reviewer"), ("content",)),
    "Parallel Research": TaskFingerprint("parallel", 5, 4, ("analyst", "writer"), ("research",)),
    "Simple Task": TaskFingerprint("sequential", 1, 1, ("worker",), ("simple",)),
}

EXPERIENCE_PRESETS = {
    "Fresh (no experience)": [],
    "Warm (escalate works for epistemic)": [
        ExperienceRecord("router", "epistemic", "escalate", True),
        ExperienceRecord("router", "epistemic", "escalate", True),
        ExperienceRecord("router", "epistemic", "retry", False),
    ],
    "Mixed (retry and escalate both tried)": [
        ExperienceRecord("planner", "epistemic", "escalate", True),
        ExperienceRecord("planner", "epistemic", "retry", True),
        ExperienceRecord("planner", "somatic", "halt", True),
        ExperienceRecord("router", "epistemic", "retry", False),
    ],
}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _run_selection(preset_name):
    if preset_name not in PRESETS:
        return "Select a preset."
    fp = PRESETS[preset_name]
    scored = [(t, round(_score(t, fp), 4)) for t in TEMPLATES]
    scored.sort(key=lambda x: x[1], reverse=True)
    rows = ""
    for i, (t, s) in enumerate(scored):
        bg = "#1a2a1a" if i == 0 else "transparent"
        badge = ' <span style="color:#6a6;font-size:12px">SELECTED</span>' if i == 0 else ""
        rows += f"""<tr style="background:{bg}">
            <td style="padding:10px;font-weight:{'600' if i==0 else '400'}">{t.name}{badge}</td>
            <td style="padding:10px">{t.topology}</td>
            <td style="padding:10px">{t.stage_count} stages</td>
            <td style="padding:10px;font-weight:600">{s:.4f}</td>
        </tr>"""
    fp_html = f"""<div style="margin-bottom:16px;padding:12px;background:#111;border-radius:6px;font-size:14px">
        <strong>Fingerprint:</strong> shape={fp.task_shape}, tools={fp.tool_count}, subtasks={fp.subtask_count}, roles={fp.required_roles}, tags={fp.tags}
    </div>"""
    table = f"""<table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Template</th>
            <th style="text-align:left;padding:8px">Topology</th>
            <th style="text-align:left;padding:8px">Stages</th>
            <th style="text-align:left;padding:8px">Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""
    return fp_html + table


def _run_experience(exp_preset):
    if exp_preset not in EXPERIENCE_PRESETS:
        return "Select a preset."
    records = EXPERIENCE_PRESETS[exp_preset]
    if not records:
        return '<p style="color:#888">Empty experience pool — watcher uses rule-based decisions only.</p>'
    rows = ""
    for r in records:
        color = "#4a4" if r.outcome_success else "#a44"
        rows += f"""<tr>
            <td style="padding:8px">{r.stage_name}</td>
            <td style="padding:8px">{r.signal_category}</td>
            <td style="padding:8px">{r.intervention_kind}</td>
            <td style="padding:8px;color:{color}">{'success' if r.outcome_success else 'failed'}</td>
        </tr>"""
    # Compute recommendation
    success_by_kind: dict[str, int] = {}
    for r in records:
        if r.outcome_success:
            success_by_kind[r.intervention_kind] = success_by_kind.get(r.intervention_kind, 0) + 1
    rec = max(success_by_kind, key=lambda k: success_by_kind[k]) if success_by_kind else "none"
    table = f"""<table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:16px">
        <thead><tr style="border-bottom:2px solid #333">
            <th style="text-align:left;padding:8px">Stage</th>
            <th style="text-align:left;padding:8px">Signal</th>
            <th style="text-align:left;padding:8px">Intervention</th>
            <th style="text-align:left;padding:8px">Outcome</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""
    rec_html = f'<div style="padding:12px;background:#0a1a2a;border:1px solid #1a2a3a;border-radius:6px;font-size:14px"><strong style="color:#6af">Recommendation:</strong> {rec.upper()}</div>'
    return table + rec_html


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="Operon Adaptive Assembly", theme=gr.themes.Base()) as demo:
        gr.Markdown("# Operon Adaptive Assembly\nTemplate selection, organism construction, and experience-driven intervention.")

        with gr.Tab("Template Selection"):
            preset = gr.Dropdown(list(PRESETS.keys()), label="Task Preset", value="Enterprise Analysis")
            btn = gr.Button("Select Template")
            out = gr.HTML()
            btn.click(_run_selection, inputs=[preset], outputs=[out])

        with gr.Tab("Experience Pool"):
            exp_preset = gr.Dropdown(list(EXPERIENCE_PRESETS.keys()), label="Experience Preset", value="Fresh (no experience)")
            btn2 = gr.Button("Show Pool & Recommendation")
            out2 = gr.HTML()
            btn2.click(_run_experience, inputs=[exp_preset], outputs=[out2])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
