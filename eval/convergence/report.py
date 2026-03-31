"""Report generation for the C6 convergence evaluation harness.

Produces JSON and Markdown outputs from evaluation results, including
a ranked configuration comparison table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .metrics import AggregateMetrics


def ranking_table(aggregates: dict[str, AggregateMetrics]) -> str:
    """Generate a Markdown ranking table from aggregate metrics.

    Configurations are sorted by success rate (desc), then risk score (asc).
    """
    rows: list[tuple[str, AggregateMetrics]] = sorted(
        aggregates.items(),
        key=lambda kv: (-kv[1].success_rate, kv[1].mean_risk_score),
    )

    lines: list[str] = []
    lines.append("| Rank | Config | Success Rate | Mean Risk | Mean Tokens | Mean Latency (ms) | Interventions | N |")
    lines.append("|------|--------|-------------|-----------|-------------|-------------------|---------------|---|")

    for rank, (config_id, agg) in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {config_id} "
            f"| {agg.success_rate:.1%} "
            f"| {agg.mean_risk_score:.4f} "
            f"| {agg.mean_token_cost:.0f} "
            f"| {agg.mean_latency_ms:.0f} "
            f"| {agg.mean_interventions:.1f} "
            f"| {agg.n_tasks} |"
        )

    return "\n".join(lines)


def generate_convergence_report(
    results: dict,
    out_json: str | None = None,
    out_markdown: str | None = None,
) -> dict:
    """Generate a convergence evaluation report.

    Parameters
    ----------
    results:
        The output dict from ConvergenceHarness.run().
    out_json:
        Optional path to write JSON results.
    out_markdown:
        Optional path to write Markdown report.

    Returns
    -------
    dict
        The results dict (pass-through for chaining).
    """
    # Write JSON.
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True, default=str)

    # Write Markdown.
    if out_markdown:
        os.makedirs(os.path.dirname(out_markdown) or ".", exist_ok=True)
        md = _build_markdown(results)
        with open(out_markdown, "w", encoding="utf-8") as f:
            f.write(md)

    return results


def _build_markdown(results: dict) -> str:
    """Build a Markdown report from results."""
    lines: list[str] = []

    lines.append("# C6 Convergence Evaluation Report")
    lines.append("")
    lines.append(f"**Seed:** {results.get('seed', 'unknown')}")
    lines.append(f"**Tasks:** {results.get('n_tasks', 0)}")
    lines.append(f"**Configurations:** {results.get('n_configs', 0)}")
    lines.append("")

    # Comparison table.
    lines.append("## Configuration Ranking")
    lines.append("")
    comparison = results.get("comparison", [])
    if comparison:
        lines.append("| Rank | Config | Success Rate | Mean Risk | Mean Tokens | Mean Latency (ms) | Interventions | N |")
        lines.append("|------|--------|-------------|-----------|-------------|-------------------|---------------|---|")
        for rank, row in enumerate(comparison, 1):
            lines.append(
                f"| {rank} | {row['config_id']} "
                f"| {row['success_rate']:.1%} "
                f"| {row['mean_risk_score']:.4f} "
                f"| {row['mean_token_cost']:.0f} "
                f"| {row['mean_latency_ms']:.0f} "
                f"| {row['mean_interventions']:.1f} "
                f"| {row['n_tasks']} |"
            )
    lines.append("")

    # Structural variation.
    lines.append("## Structural Variation")
    lines.append("")
    variation = results.get("variation", {})
    if variation:
        for config_id, val in sorted(variation.items()):
            lines.append(f"- **{config_id}**: {val:.4f}")
    lines.append("")

    # Credit assignment.
    lines.append("## Credit Assignment (Top Roles)")
    lines.append("")
    credit = results.get("credit", {})
    if credit:
        sorted_credit = sorted(credit.items(), key=lambda kv: -abs(kv[1]))[:10]
        for role, val in sorted_credit:
            lines.append(f"- **{role}**: {val:+.4f}")
    lines.append("")

    return "\n".join(lines) + "\n"
