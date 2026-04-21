"""Paper 6 analysis — Mann-Whitney U on mutations-to-convergence.

Reads per-seed trajectory JSON files from
``eval/results/theorem_6/{arm}/seed_*.json`` and produces:

- a summary table with Wilson 95% CIs per arm,
- Mann-Whitney U p-values and rank-biserial effect sizes for
  cert-binary vs each baseline,
- a CDF plot (matplotlib PNG) of mutations-to-convergence,
- a machine-readable summary JSON at
  ``eval/results/theorem_6/summary.json``.

The summary JSON is what paper-6's results section consumes verbatim
(numbers copied into tex tables).  Keep the schema stable.

Usage::

    python -m eval.convergence.analysis_theorem_6
    python -m eval.convergence.analysis_theorem_6 --results-dir alt/path
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Optional deps — imported lazily inside functions that actually need
# them so unit tests can import this module without scipy/matplotlib
# installed.  CI does not ship scipy; the analysis run does.
def _require_scipy_stats():
    """Return ``scipy.stats`` or raise a clear RuntimeError."""
    try:
        import scipy.stats as stats  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scipy is required for paper-6 analysis (pip install scipy)."
        ) from exc
    return stats


BUDGET_SENTINEL = None
"""Sentinel: no convergence within budget is recorded as ``None`` in the JSON.

Analysis substitutes ``budget_iterations`` for non-converged runs so
Mann-Whitney U sees a comparable number; the substitution is documented
in the paper's methods section.
"""


# ---------------------------------------------------------------------------
# Wilson 95% CI
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Matches the CI convention used in papers 4 and 5 (memory references).
    Returns ``(lower, upper)``; both in [0, 1].  Defined as 0/0 → (0, 0).
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def load_arm_records(
    arm_dir: Path,
    *,
    include_mock: bool = False,
    include_errored: bool = False,
) -> list[dict[str, Any]]:
    """Load per-seed records from ``arm_dir``; filter mocked and errored runs by default.

    A record is considered **errored** if its ``gepa_error`` field is
    non-null (the experiment driver captures any GEPA exception there
    via ``raise_on_exception=False``).  The paper's methods section
    states that failed runs are excluded from statistics; this is the
    entry point that enforces that contract.  Pass ``include_errored=True``
    when debugging or when reporting the error rate separately.
    """
    if not arm_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(arm_dir.glob("seed_*.json")):
        # Fail loudly on unreadable / corrupt JSON.  Silently continuing
        # would quietly shrink n_runs and skew Wilson CIs / Mann-Whitney
        # without any signal in the summary (Roborev #864 M2).  A
        # truncated result file is a bug in the run, not a routine
        # condition to tolerate.
        try:
            text = path.read_text()
        except OSError as exc:
            raise RuntimeError(f"Unreadable result file: {path}") from exc
        try:
            rec = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Corrupt result file (JSON parse failed): {path}"
            ) from exc
        if rec.get("mock_reflection_lm") and not include_mock:
            continue
        if rec.get("gepa_error") is not None and not include_errored:
            continue
        records.append(rec)
    return records


def count_errored_records(arm_dir: Path, *, include_mock: bool = False) -> int:
    """Count errored runs in ``arm_dir`` (non-null ``gepa_error``).

    Raises ``RuntimeError`` on unreadable or corrupt result files, same
    as :func:`load_arm_records` — incomplete datasets must not
    masquerade as valid results (Roborev #864 M2).
    """
    if not arm_dir.exists():
        return 0
    n = 0
    for path in sorted(arm_dir.glob("seed_*.json")):
        try:
            text = path.read_text()
        except OSError as exc:
            raise RuntimeError(f"Unreadable result file: {path}") from exc
        try:
            rec = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Corrupt result file (JSON parse failed): {path}"
            ) from exc
        if rec.get("mock_reflection_lm") and not include_mock:
            continue
        if rec.get("gepa_error") is not None:
            n += 1
    return n


def convergence_iters(
    records: list[dict[str, Any]], *, replace_unconverged_with: int | None = None
) -> list[int]:
    """Extract convergence iteration per record.

    Records that did not converge within budget are either dropped (if
    ``replace_unconverged_with`` is None) or replaced with the given
    sentinel value (e.g. ``budget_iterations``).  The default for
    downstream statistics is to replace with the budget so every record
    contributes a comparable number.
    """
    out: list[int] = []
    for rec in records:
        conv = rec.get("convergence_iteration", BUDGET_SENTINEL)
        if conv is None:
            if replace_unconverged_with is not None:
                out.append(int(replace_unconverged_with))
        else:
            out.append(int(conv))
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mann_whitney(treatment: list[int], baseline: list[int]) -> dict[str, Any]:
    """Mann-Whitney U + rank-biserial effect size.

    ``treatment`` should be the cert-binary arm; lower values = faster
    convergence = desired treatment effect.  Uses ``alternative="less"``
    to test the directional conjecture.
    """
    if not treatment or not baseline:
        return {
            "u": None,
            "p_value": None,
            "effect_size_rank_biserial": None,
            "n_treatment": len(treatment),
            "n_baseline": len(baseline),
            "note": "insufficient data",
        }
    stats = _require_scipy_stats()
    u_stat, p = stats.mannwhitneyu(
        treatment, baseline, alternative="less"
    )
    n1, n2 = len(treatment), len(baseline)
    # Rank-biserial correlation, convention: **positive = treatment
    # dominates** (fewer mutations to converge).  scipy's U (with
    # alternative="less") counts (x > y) pairs; a treatment that's
    # uniformly smaller yields U=0 → rb=1.  A treatment uniformly
    # larger yields rb=-1.  rb=0 means stochastic equality.
    rb = 1 - (2 * u_stat) / (n1 * n2)
    return {
        "u": float(u_stat),
        "p_value": float(p),
        "effect_size_rank_biserial": float(rb),
        "n_treatment": n1,
        "n_baseline": n2,
    }


def arm_summary(
    records: list[dict[str, Any]], *, budget: int | None = None
) -> dict[str, Any]:
    """Per-arm aggregate statistics."""
    n = len(records)
    converged = sum(1 for r in records if r.get("convergence_iteration") is not None)
    replace_with = budget if budget is not None else None
    iters = convergence_iters(records, replace_unconverged_with=replace_with)
    mean_iters = (sum(iters) / len(iters)) if iters else None
    ci_low, ci_high = wilson_ci(converged, n)
    return {
        "n_runs": n,
        "n_converged": converged,
        "convergence_rate": converged / n if n else 0.0,
        "convergence_rate_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "mean_iterations_to_convergence": (
            round(mean_iters, 3) if mean_iters is not None else None
        ),
        "convergence_iterations": iters,
    }


# ---------------------------------------------------------------------------
# CDF plot
# ---------------------------------------------------------------------------


def plot_cdf(
    arm_to_series: dict[str, tuple[list[int], int]],
    *,
    out_path: Path,
    budget: int,
) -> bool:
    """Plot convergence CDF per arm without imputing non-converged runs.

    ``arm_to_series`` maps each arm to ``(converged_iterations, total_runs)``
    where ``converged_iterations`` contains only the runs that actually
    converged and ``total_runs`` is the full denominator (converged +
    non-converged, excluding errored runs).  The y-axis is therefore the
    proportion of the arm's total runs that had converged by iteration
    $k$ — curves level off at the arm's convergence rate, not at 1.0.
    This matches the caption (``P(converged by iteration)``) and avoids
    treating budget-exhausted runs as successful convergence events
    (Roborev #854 M2; right-censored data handled by not imputing).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(range(budget + 2))
    for arm, (iters, total) in arm_to_series.items():
        if total <= 0:
            continue
        sorted_iters = sorted(iters)
        ys = [sum(1 for v in sorted_iters if v <= x) / total for x in xs]
        ax.step(
            xs, ys, where="post",
            label=f"{arm} (converged {len(iters)}/{total})",
        )

    ax.set_xlabel("GEPA iteration")
    ax.set_ylabel("P(converged by iteration)")
    ax.set_title("Paper 6 — mutations-to-convergence CDF")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, budget)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze(
    results_dir: Path,
    *,
    treatment_arm: str = "cert-binary",
    baseline_arms: tuple[str, ...] = ("scalar", "scalar-evidence"),
) -> dict[str, Any]:
    """Load all arms, compute stats, return the full summary dict."""
    arms = (treatment_arm, *baseline_arms)
    arm_records: dict[str, list[dict[str, Any]]] = {
        arm: load_arm_records(results_dir / arm) for arm in arms
    }
    # Count errored runs separately so the paper can report them
    # alongside per-arm statistics rather than silently losing them.
    errored_counts: dict[str, int] = {
        arm: count_errored_records(results_dir / arm) for arm in arms
    }

    # Budget taken from any record; assume uniform across the sweep.
    budget: int | None = None
    for recs in arm_records.values():
        if recs:
            budget = int(recs[0].get("budget_iterations", 50))
            break

    per_arm = {
        arm: arm_summary(records, budget=budget)
        for arm, records in arm_records.items()
    }

    treatment_iters = convergence_iters(
        arm_records[treatment_arm], replace_unconverged_with=budget
    )

    comparisons: dict[str, Any] = {}
    for baseline in baseline_arms:
        baseline_iters = convergence_iters(
            arm_records[baseline], replace_unconverged_with=budget
        )
        comparisons[baseline] = mann_whitney(treatment_iters, baseline_iters)

    # For the CDF: raw converged-iter values only (no imputation at budget).
    # The denominator is total non-errored runs in the arm so curves
    # level off at the convergence rate, not at 1.0.
    arm_to_series = {
        arm: (
            convergence_iters(recs, replace_unconverged_with=None),
            len(recs),
        )
        for arm, recs in arm_records.items()
    }
    cdf_path = results_dir / "cdf.png"
    cdf_ok = False
    if budget:
        cdf_ok = plot_cdf(arm_to_series, out_path=cdf_path, budget=budget)

    summary = {
        "results_dir": str(results_dir),
        "treatment_arm": treatment_arm,
        "baseline_arms": list(baseline_arms),
        "budget_iterations": budget,
        "per_arm": per_arm,
        "errored_runs_excluded": errored_counts,
        "comparisons_treatment_vs_baseline": comparisons,
        "cdf_plot": str(cdf_path) if cdf_ok else None,
    }

    summary_path = results_dir / "summary.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="eval/results/theorem_6"
    )
    parser.add_argument("--treatment", default="cert-binary")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["scalar", "scalar-evidence"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    summary = analyze(
        Path(args.results_dir),
        treatment_arm=args.treatment,
        baseline_arms=tuple(args.baselines),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
