"""Aggregate benchmark results and produce comparison tables.

Usage:
    python -m eval.benchmarks.report_benchmarks eval/results/benchmarks/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from eval.benchmarks.types import (
    BenchmarkResult,
    ComparisonRow,
    TrialResult,
    Variant,
    build_comparison_table,
)


def load_results(directory: str) -> list[BenchmarkResult]:
    """Load all result JSON files from a directory."""
    results: list[BenchmarkResult] = []

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(directory, fname)) as f:
            data = json.load(f)

        for r in data.get("results", []):
            br = BenchmarkResult(
                name=r["name"],
                scenario_name=r["scenario"],
                seed=r.get("seed", 0),
                config=r.get("config", {}),
            )
            for variant_name in ["biological", "ablated", "naive"]:
                if variant_name not in r:
                    continue
                variant = Variant(variant_name)
                br.variants[variant] = {}
                for metric_name, metric_data in r[variant_name].items():
                    br.variants[variant][metric_name] = TrialResult(
                        variant=variant,
                        metric_name=metric_name,
                        success=metric_data["success"],
                        total=metric_data["total"],
                        latency_steps=metric_data.get("latency_steps"),
                    )
            results.append(br)

    return results


def print_comparison_table(rows: list[ComparisonRow], benchmark_name: str) -> None:
    """Print a markdown comparison table."""
    print(f"\n## {benchmark_name}\n")
    print("| Scenario | Metric | Biological | Ablated | Naive | Δ(Bio-Naive) | Sig? | N |")
    print("|----------|--------|------------|---------|-------|-------------|------|---|")

    for row in rows:
        sig = "**yes**" if row.significant else "no"
        delta_str = f"{row.delta_bio_vs_naive:+.3f}"
        print(
            f"| {row.scenario} | {row.metric} "
            f"| {row.biological_rate:.3f} [{row.biological_ci[0]:.2f},{row.biological_ci[1]:.2f}] "
            f"| {row.ablated_rate:.3f} [{row.ablated_ci[0]:.2f},{row.ablated_ci[1]:.2f}] "
            f"| {row.naive_rate:.3f} [{row.naive_ci[0]:.2f},{row.naive_ci[1]:.2f}] "
            f"| {delta_str} | {sig} | {row.n} |"
        )


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text."""
    for char in ("_", "&", "%", "#", "$", "{", "}"):
        text = text.replace(char, f"\\{char}")
    return text


def generate_latex_table(rows: list[ComparisonRow], benchmark_name: str) -> str:
    """Generate LaTeX table for paper inclusion."""
    lines = [
        r"\begin{table}[h]",
        rf"\caption{{{_escape_latex(benchmark_name)}: Biological vs Naive Comparison}}",
        r"\centering",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Scenario & Metric & Biological & Ablated & Naive & $\Delta$ & N \\",
        r"\midrule",
    ]

    for row in rows:
        sig = r"\textbf{" + f"{row.delta_bio_vs_naive:+.3f}" + "}" if row.significant else f"{row.delta_bio_vs_naive:+.3f}"
        lines.append(
            f"{_escape_latex(row.scenario)} & {_escape_latex(row.metric)} "
            f"& {row.biological_rate:.3f} & {row.ablated_rate:.3f} "
            f"& {row.naive_rate:.3f} & {sig} & {row.n} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("directory", help="Directory containing result JSON files")
    parser.add_argument("--latex", action="store_true", help="Also output LaTeX tables")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        return 1

    results = load_results(args.directory)
    if not results:
        print("No results found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(results)} results from {args.directory}")

    # Group by benchmark name
    by_name: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    for name, bench_results in sorted(by_name.items()):
        rows = build_comparison_table(bench_results)
        print_comparison_table(rows, name)

        if args.latex:
            latex = generate_latex_table(rows, name)
            out_path = os.path.join(args.directory, f"{name}_table.tex")
            with open(out_path, "w") as f:
                f.write(latex)
            print(f"\n  LaTeX table written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
