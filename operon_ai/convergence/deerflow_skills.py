"""Bidirectional conversion between DeerFlow Markdown skills and Operon PatternTemplates.

DeerFlow skills are Markdown files with optional YAML frontmatter describing
a workflow as numbered steps.  This module parses those into Operon
:class:`PatternTemplate` instances and exports templates back to the DeerFlow
Markdown format.

No external Markdown or YAML libraries are required beyond the stdlib ``yaml``
is **not** used; frontmatter is parsed with simple string splitting to keep
Operon dependency-free.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from ..patterns.repository import PatternTemplate, TaskFingerprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_STEP_RE = re.compile(r"^\s*\d+\.\s+(.+)$", re.MULTILINE)
_YAML_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$")


def parse_skill_frontmatter(skill_md: str) -> dict[str, str]:
    """Extract YAML frontmatter from a DeerFlow skill Markdown string.

    Returns a flat dict of string key-value pairs.  If the document has no
    ``---`` delimited frontmatter, returns an empty dict.
    """
    if not skill_md:
        return {}

    match = _FRONTMATTER_RE.search(skill_md)
    if match is None:
        return {}

    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        m = _YAML_LINE_RE.match(line)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def extract_workflow_steps(skill_md: str) -> list[str]:
    """Extract numbered workflow steps from the Markdown body.

    Returns a list of step description strings (without the leading number).
    """
    if not skill_md:
        return []
    return _STEP_RE.findall(skill_md)


# ---------------------------------------------------------------------------
# Skill -> PatternTemplate
# ---------------------------------------------------------------------------


def skill_to_template(skill_md: str) -> PatternTemplate:
    """Parse a DeerFlow Markdown skill into an Operon :class:`PatternTemplate`.

    Parameters
    ----------
    skill_md:
        Full Markdown content of a DeerFlow skill file, optionally including
        YAML frontmatter between ``---`` delimiters.

    Returns
    -------
    PatternTemplate
        A template whose ``stage_specs`` mirror the numbered steps found in
        the skill body.
    """
    frontmatter = parse_skill_frontmatter(skill_md)
    steps = extract_workflow_steps(skill_md)

    name = frontmatter.get("name", "unnamed_skill")
    category = frontmatter.get("category", "worker")

    # Use per-step roles from frontmatter if available (roundtrip fidelity).
    roles_str = frontmatter.get("roles", "")
    per_step_roles = [r.strip() for r in roles_str.split(",") if r.strip()] if roles_str else []

    # Build stage specs — one per numbered step.
    stage_specs: list[dict[str, Any]] = []
    for idx, step_text in enumerate(steps, start=1):
        role = per_step_roles[idx - 1] if idx - 1 < len(per_step_roles) else category
        stage_specs.append({
            "name": f"step_{idx}",
            "role": role,
            "instructions": step_text,
        })

    # Reject empty skills — a template with zero stages is invalid.
    step_count = len(steps)
    if step_count == 0:
        raise ValueError("Skill has no workflow steps — cannot create a valid template")
    if step_count >= 4:
        task_shape = "mixed"
    else:
        task_shape = "sequential"

    roles = tuple(sorted({s["role"] for s in stage_specs})) if stage_specs else ()

    # Collect tags from frontmatter.
    tags_raw = frontmatter.get("tags", "")
    tags: list[str] = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
    tags.append("deerflow_skill")

    fingerprint = TaskFingerprint(
        task_shape=task_shape,
        tool_count=0,
        subtask_count=step_count,
        required_roles=roles,
        tags=tuple(tags),
    )

    topology = "skill_organism" if step_count > 1 else "single_worker"

    return PatternTemplate(
        template_id=uuid4().hex[:8],
        name=name,
        topology=topology,
        stage_specs=tuple(stage_specs),
        intervention_policy={},
        fingerprint=fingerprint,
        tags=tuple(tags),
    )


# ---------------------------------------------------------------------------
# PatternTemplate -> Skill
# ---------------------------------------------------------------------------

_TOPOLOGY_CATEGORY_MAP: dict[str, str] = {
    "single_worker": "general",
    "skill_organism": "workflow",
    "specialist_swarm": "multi-agent",
    "reviewer_gate": "review",
}


def template_to_skill(template: PatternTemplate) -> str:
    """Export an Operon :class:`PatternTemplate` as DeerFlow Markdown.

    Parameters
    ----------
    template:
        The pattern template to serialise.

    Returns
    -------
    str
        A Markdown string with YAML frontmatter and numbered workflow steps
        compatible with DeerFlow's skill format.
    """
    category = _TOPOLOGY_CATEGORY_MAP.get(template.topology, "general")
    tags_str = ", ".join(template.tags) if template.tags else ""

    # Preserve roles as comma-separated list in frontmatter for roundtrip fidelity.
    roles = ", ".join(spec.get("role", "worker") for spec in template.stage_specs)

    lines: list[str] = [
        "---",
        f"name: {template.name}",
        "description: Operon pattern template",
        "version: 1.0",
        f"category: {category}",
        f"roles: {roles}",
        f"tags: {tags_str}",
        "---",
        "",
        f"# {template.name}",
        "",
    ]

    for idx, spec in enumerate(template.stage_specs, start=1):
        instructions = spec.get("instructions", spec.get("role", "worker"))
        lines.append(f"{idx}. {instructions}")

    # Trailing newline.
    lines.append("")
    return "\n".join(lines)
