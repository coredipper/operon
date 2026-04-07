"""Report generation for topology validation results.

Produces JSON (machine-readable) and Markdown (human-readable) reports
from a :class:`ValidationReport`.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .topology_validation import ValidationReport


def generate_report(
    report: ValidationReport,
    out_dir: str = "eval/results/topology_validation",
) -> None:
    """Write JSON and Markdown reports to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    _write_json(report, out_dir)
    _write_markdown(report, out_dir)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _write_json(report: ValidationReport, out_dir: str) -> None:
    from .topology_validation import _pair_to_dict

    data = {
        "metadata": report.metadata,
        "summary": report.summary,
        "theorems": [
            {
                "name": t.theorem_name,
                "description": t.description,
                "n_pairs": t.n_pairs,
                "spearman_rho": round(t.spearman_rho, 4),
                "spearman_p": round(t.spearman_p, 4),
                "direction_correct": t.direction_correct,
                "validation_pass": t.validation_pass,
                "informational": t.informational,
                "notes": t.notes,
            }
            for t in report.theorems
        ],
        "by_topology_class": {
            cls: {k: round(v, 4) if isinstance(v, float) else v for k, v in vals.items()}
            for cls, vals in report.by_topology_class.items()
        },
        "pairs": [_pair_to_dict(p) for p in report.pairs],
    }

    path = os.path.join(out_dir, "validation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _write_markdown(report: ValidationReport, out_dir: str) -> None:
    lines: list[str] = []
    _a = lines.append

    _a("# Topology Validation Report")
    _a("")
    m = report.metadata
    _a(f"**Date**: {m.get('timestamp', 'N/A')}  ")
    _a(f"**Model**: {m.get('model', 'N/A')} via {m.get('provider', 'N/A')}  ")
    _a(f"**Tasks**: {m.get('n_tasks', 0)} x {m.get('n_configs', 0)} configs x {m.get('n_repeats', 0)} repeats = {m.get('total_runs', 0)} runs  ")
    _a(f"**Duration**: {m.get('elapsed_seconds', 0):.0f}s  ")
    _a("")

    # Summary
    s = report.summary
    _a("## Summary")
    _a("")
    _a(f"- Theorems validated: **{s.get('theorems_validated', 0)}/{s.get('theorems_total', 0)}**")
    _a(f"- Overall success rate: **{s.get('success_rate', 0):.1%}**")
    _a(f"- Mean quality score: **{s.get('mean_quality', 0):.3f}**")
    _a("")

    # Theorem table
    _a("## Theorem Correlations")
    _a("")
    _a("| Theorem | Description | rho | p-value | Direction | Pass |")
    _a("|---------|-------------|-----|---------|-----------|------|")
    for t in report.theorems:
        direction = "correct" if t.direction_correct else "wrong"
        if t.informational:
            passed = "info"
        else:
            passed = "yes" if t.validation_pass else "no"
        _a(
            f"| {t.theorem_name} | {t.description} | "
            f"{t.spearman_rho:+.3f} | {t.spearman_p:.4f} | "
            f"{direction} | {passed} |"
        )
    _a("")

    # Notes per theorem
    _a("### Notes")
    _a("")
    for t in report.theorems:
        _a(f"- **{t.theorem_name}**: {t.notes}")
    _a("")

    # By topology class
    _a("## Performance by Topology Class")
    _a("")
    _a("| Class | Runs | Success Rate | Mean Quality | Mean Latency (ms) | Mean Tokens |")
    _a("|-------|------|-------------|-------------|-------------------|-------------|")
    for cls, vals in sorted(report.by_topology_class.items()):
        _a(
            f"| {cls} | {vals['n_runs']} | "
            f"{vals['success_rate']:.1%} | {vals['mean_quality']:.3f} | "
            f"{vals['mean_latency_ms']:.0f} | {vals['mean_tokens']:.0f} |"
        )
    _a("")

    # Per-task detail (top/bottom by quality delta)
    _a("## Per-Task Results")
    _a("")
    _a("| Task | Config | Rep | Quality | Success | Tokens | Latency (ms) | Error Bound | Risk |")
    _a("|------|--------|-----|---------|---------|--------|-------------|-------------|------|")
    for p in sorted(report.pairs, key=lambda x: x.measurement.quality_score):
        pr = p.prediction
        ms = p.measurement
        _a(
            f"| {pr.task_id} | {pr.config_name} | {pr.repeat} | "
            f"{ms.quality_score:.2f} | {'yes' if ms.success else 'no'} | "
            f"{ms.total_tokens} | {ms.total_latency_ms:.0f} | "
            f"{pr.error_centralized_bound:.2f} | {pr.risk_score:.4f} |"
        )
    _a("")

    path = os.path.join(out_dir, "validation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
