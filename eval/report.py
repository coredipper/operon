from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eval.utils import wilson_interval


@dataclass
class Aggregate:
    name: str
    success: int = 0
    total: int = 0
    seed_rates: list[float] = None

    def __post_init__(self):
        if self.seed_rates is None:
            self.seed_rates = []

    def add(self, success: int, total: int) -> None:
        self.success += success
        self.total += total
        if total > 0:
            self.seed_rates.append(success / total)

    def stats(self) -> dict:
        rate = self.success / self.total if self.total else 0.0
        low, high = wilson_interval(self.success, self.total)
        mean = sum(self.seed_rates) / len(self.seed_rates) if self.seed_rates else 0.0
        variance = 0.0
        if len(self.seed_rates) > 1:
            variance = sum((r - mean) ** 2 for r in self.seed_rates) / (len(self.seed_rates) - 1)
        return {
            "success": self.success,
            "total": self.total,
            "rate": rate,
            "wilson_95": [low, high],
            "seed_mean": mean,
            "seed_std": variance ** 0.5,
        }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}\\\\%"


def _percent_ci(low: float, high: float) -> str:
    return f"[{low * 100:.1f}, {high * 100:.1f}]\\\\%"


def _load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def aggregate(files: list[Path]) -> dict:
    folding_strict = Aggregate("folding_strict")
    folding_cascade = Aggregate("folding_cascade")
    healing_with_error = Aggregate("healing_with_error")
    healing_blind = Aggregate("healing_blind")
    immune_sensitivity = Aggregate("immune_sensitivity")
    immune_fpr = Aggregate("immune_false_positive")
    bfcl_strict = Aggregate("bfcl_strict")
    bfcl_cascade = Aggregate("bfcl_cascade")
    agentdojo_sensitivity = Aggregate("agentdojo_sensitivity")
    agentdojo_fpr = Aggregate("agentdojo_false_positive")

    for path in files:
        data = _load_results(path)
        suites = data.get("suites", {})

        folding = suites.get("folding", {}).get("results", {})
        if folding:
            strict = folding.get("strict", {})
            cascade = folding.get("cascade", {})
            folding_strict.add(strict.get("success", 0), strict.get("total", 0))
            folding_cascade.add(cascade.get("success", 0), cascade.get("total", 0))

        immune = suites.get("immune", {}).get("results", {})
        if immune:
            sens = immune.get("sensitivity", {})
            fpr = immune.get("false_positive_rate", {})
            immune_sensitivity.add(sens.get("success", 0), sens.get("total", 0))
            immune_fpr.add(fpr.get("success", 0), fpr.get("total", 0))

        healing = suites.get("healing", {}).get("results", {})
        if healing:
            with_error = healing.get("with_error_context", {}).get("success", {})
            blind = healing.get("blind_retry", {}).get("success", {})
            healing_with_error.add(with_error.get("success", 0), with_error.get("total", 0))
            healing_blind.add(blind.get("success", 0), blind.get("total", 0))

        bfcl = suites.get("bfcl_folding", {}).get("results", {})
        if bfcl:
            strict = bfcl.get("strict", {})
            cascade = bfcl.get("cascade", {})
            bfcl_strict.add(strict.get("success", 0), strict.get("total", 0))
            bfcl_cascade.add(cascade.get("success", 0), cascade.get("total", 0))

        adj = suites.get("agentdojo_immune", {}).get("results", {})
        if adj:
            sens = adj.get("sensitivity", {})
            fpr = adj.get("false_positive_rate", {})
            agentdojo_sensitivity.add(sens.get("success", 0), sens.get("total", 0))
            agentdojo_fpr.add(fpr.get("success", 0), fpr.get("total", 0))

    return {
        "folding_strict": folding_strict.stats(),
        "folding_cascade": folding_cascade.stats(),
        "healing_with_error": healing_with_error.stats(),
        "healing_blind": healing_blind.stats(),
        "immune_sensitivity": immune_sensitivity.stats(),
        "immune_false_positive": immune_fpr.stats(),
        "bfcl_strict": bfcl_strict.stats(),
        "bfcl_cascade": bfcl_cascade.stats(),
        "agentdojo_sensitivity": agentdojo_sensitivity.stats(),
        "agentdojo_false_positive": agentdojo_fpr.stats(),
        "seeds": len(files),
    }


def write_latex(summary: dict, out_path: Path) -> None:
    rows = [
        ("Chaperone Folding (Strict)", summary["folding_strict"]),
        ("Chaperone Folding (Cascade)", summary["folding_cascade"]),
        ("Healing Loop (Error Context)", summary["healing_with_error"]),
        ("Healing Loop (Blind Retry)", summary["healing_blind"]),
        ("Immune Detection (Sensitivity)", summary["immune_sensitivity"]),
        ("Immune Detection (False Positive)", summary["immune_false_positive"]),
    ]

    external_rows = [
        ("BFCL Folding (Strict)", summary.get("bfcl_strict")),
        ("BFCL Folding (Cascade)", summary.get("bfcl_cascade")),
        ("AgentDojo Immune (Sensitivity)", summary.get("agentdojo_sensitivity")),
        ("AgentDojo Immune (False Positive)", summary.get("agentdojo_false_positive")),
    ]
    external_rows = [(n, s) for n, s in external_rows if s and s.get("total", 0) > 0]

    lines = []
    lines.append("% Auto-generated by eval/report.py")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{p{0.46\\linewidth}ccc}")
    lines.append("\\toprule")
    lines.append("Metric & Rate & 95\\% CI & N \\\\")
    lines.append("\\midrule")

    for name, stats in rows:
        rate = _percent(stats["rate"])
        low, high = stats["wilson_95"]
        ci = _percent_ci(low, high)
        total = stats["total"]
        lines.append(f"{name} & {rate} & {ci} & {total} \\\\")

    if external_rows:
        lines.append("\\midrule")
        for name, stats in external_rows:
            rate = _percent(stats["rate"])
            low, high = stats["wilson_95"]
            ci = _percent_ci(low, high)
            total = stats["total"]
            lines.append(f"{name} & {rate} & {ci} & {total} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    seed_count = summary.get("seeds", 0)
    lines.append(
        "\\caption{Evaluation results aggregated across "
        f"{seed_count} deterministic seeds (Wilson 95\\% CI). "
        "Top: synthetic motif tests. Bottom: "
        "external benchmark--derived tests (BFCL, AgentDojo).}}"
    )
    lines.append("\\label{tab:eval-summary}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--glob", default="eval/results/seed-*.json", help="Glob for result files")
    parser.add_argument("--out-json", default="eval/results/summary.json")
    parser.add_argument("--out-tex", default="eval/results/summary.tex")

    args = parser.parse_args()

    files = [Path(p) for p in glob.glob(args.glob)]
    if not files:
        raise SystemExit("No result files found")

    summary = aggregate(files)

    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_latex(summary, Path(args.out_tex))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
