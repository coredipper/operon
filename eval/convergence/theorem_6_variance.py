"""Paper 6 — run-to-run variance characterization.

The single-run paper-6 artifacts report exact point estimates (e.g.
cert-binary vs scalar-evidence tie at 1.0 mean iterations, formatter
ablation tie at 1.0).  Those rest on a non-deterministic reflection LM
(``ollama/gemma4`` with no temperature/seed pin), so a re-run can flip
a seed's convergence iteration.  This driver re-runs the protocol N
times and reports the **distribution** (median + min/max + tie-rate)
of each headline statistic, so the paper can replace fragile point
estimates with honest ranges.

The only varying factor across reps is LM sampling: GEPA gets
``seed=seed`` and the data RNG is SHA-256-seeded, so per-(arm, seed)
spread across reps is exactly the reflection-LM variance.

Two blocks, in order:

1. **Ablation tie** (cheap, the must-have): for each rep, re-run
   cert-binary with the *default* formatter and with the *minimal*
   formatter on seeds 0--4, and record whether they tie.
2. **Main-sweep parity** (slower, the ideally): for each rep, re-run
   all three arms on seeds 0--9 and record the cert-binary vs
   scalar-evidence Mann-Whitney p-value and mean ratio.

Per-cell trajectories are written to a throwaway temp dir (500+ events
each — not worth committing across hundreds of runs); only the
aggregate ``eval/results/theorem_6/variance_summary.json`` lands in the
repo.  The file is checkpointed after every rep so a crash or interrupt
leaves a usable partial result.

Usage::

    python -m eval.convergence.theorem_6_variance               # N=10, both blocks
    python -m eval.convergence.theorem_6_variance --n-reps 5
    python -m eval.convergence.theorem_6_variance --skip-main-sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .analysis_theorem_6 import mann_whitney
from .theorem_6_ablation import run_minimal_seed
from .theorem_6_experiment import ARMS, run_single

BUDGET_ITERATIONS = 50
ABLATION_SEEDS = [0, 1, 2, 3, 4]
MAIN_SWEEP_SEEDS = list(range(10))
REFLECTION_LM = "ollama/gemma4"
SUMMARY_PATH = Path("eval/results/theorem_6/variance_summary.json")
#: Raw per-rep records, persisted alongside the aggregate so an
#: interrupted run (laptop sleep, killed shell) can resume instead of
#: redoing completed reps.  Not committed — it is the run's scratch state.
RAW_REPS_PATH = Path("eval/results/theorem_6/.variance_reps.json")


def _conv_or_raise(record: dict[str, Any], *, label: str, budget: int) -> int:
    """Convergence iteration for a run; non-converged imputed at budget.

    Raises ``RuntimeError`` if the run captured a ``gepa_error``.  A
    non-converged run (``convergence_iteration is None`` with no error)
    is legitimately imputed at ``budget`` per the paper's central-tendency
    policy, but an *errored* run must NOT be imputed — silently treating
    it as a budget-length run would spike the arm mean (one cell at 50
    among ten ~1s) and masquerade as variance.  ``load_arm_records``
    excludes errored runs from a finished sweep; for a controlled
    variance study an unexpected GEPA exception should instead halt
    loudly so the cause is investigated rather than averaged in.
    """
    if record.get("gepa_error") is not None:
        raise RuntimeError(f"GEPA error in {label}: {record['gepa_error']}")
    conv = record.get("convergence_iteration")
    return budget if conv is None else int(conv)


def _dist(values: list[float]) -> dict[str, Any]:
    """Summary stats for a list of per-rep scalars."""
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 4),
    }


def _aggregate(
    reps: list[dict[str, Any]], *, n_reps: int, include_main: bool
) -> dict[str, Any]:
    """Build the variance summary from the per-rep raw records."""
    summary: dict[str, Any] = {
        "n_reps": n_reps,
        "reps_completed": len(reps),
        "budget_iterations": BUDGET_ITERATIONS,
        "reflection_lm": REFLECTION_LM,
        "ablation_seeds": ABLATION_SEEDS,
        "main_sweep_seeds": MAIN_SWEEP_SEEDS,
    }

    # --- Ablation tie block ---
    default_means = [r["ablation"]["default_mean"] for r in reps]
    minimal_means = [r["ablation"]["minimal_mean"] for r in reps]
    tie_flags = [
        r["ablation"]["default_mean"] == r["ablation"]["minimal_mean"] for r in reps
    ]
    summary["ablation"] = {
        "default_mean_iters": _dist(default_means),
        "minimal_mean_iters": _dist(minimal_means),
        "per_rep_default_mean": default_means,
        "per_rep_minimal_mean": minimal_means,
        "exact_tie_rate": round(sum(tie_flags) / len(tie_flags), 4) if tie_flags else None,
        "per_seed_default": _per_seed(reps, "ablation", "default_per_seed", ABLATION_SEEDS),
        "per_seed_minimal": _per_seed(reps, "ablation", "minimal_per_seed", ABLATION_SEEDS),
    }

    # --- Main-sweep parity block ---
    if include_main and reps and "main_sweep" in reps[0]:
        per_arm: dict[str, Any] = {}
        for arm in ARMS:
            arm_means = [r["main_sweep"]["arm_means"][arm] for r in reps]
            per_arm[arm] = {
                "mean_iters": _dist(arm_means),
                "per_rep_mean": arm_means,
                "per_seed": _per_seed(
                    reps, "main_sweep", f"per_seed::{arm}", MAIN_SWEEP_SEEDS
                ),
            }
        pvals = [r["main_sweep"]["cert_vs_evidence_p"] for r in reps]
        ratios = [r["main_sweep"]["cert_vs_evidence_ratio"] for r in reps]
        summary["main_sweep"] = {
            "per_arm": per_arm,
            "cert_vs_evidence_pvalue": _dist(pvals),
            "cert_vs_evidence_ratio": _dist(ratios),
            "per_rep_pvalue": pvals,
            "per_rep_ratio": ratios,
            "reps_p_ge_0.05": sum(1 for p in pvals if p is not None and p >= 0.05),
            "reps_gate_cleared": sum(1 for r in ratios if r is not None and r <= 0.75),
        }

    return summary


def _per_seed(
    reps: list[dict[str, Any]], block: str, key: str, seeds: list[int]
) -> dict[str, list[int]]:
    """Collect per-seed convergence iteration across reps."""
    if "::" in key:
        sub, arm = key.split("::", 1)
        out: dict[str, list[int]] = {str(s): [] for s in seeds}
        for r in reps:
            per_seed = r[block][sub][arm]
            for s in seeds:
                out[str(s)].append(per_seed[str(s)])
        return out
    out = {str(s): [] for s in seeds}
    for r in reps:
        per_seed = r[block][key]
        for s in seeds:
            out[str(s)].append(per_seed[str(s)])
    return out


def _checkpoint(reps: list[dict[str, Any]], *, n_reps: int, include_main: bool) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(_aggregate(reps, n_reps=n_reps, include_main=include_main), indent=2)
    )
    # Persist raw reps so an interrupted run can resume (see RAW_REPS_PATH).
    # ``n_reps`` is stored so a resume can validate the requested run shape.
    RAW_REPS_PATH.write_text(
        json.dumps({"n_reps": n_reps, "include_main": include_main, "reps": reps})
    )


def _load_resume(*, include_main: bool, n_reps: int) -> list[dict[str, Any]]:
    """Load previously completed reps if a compatible checkpoint exists.

    Only resumes when the saved ``include_main`` matches the requested
    run shape — an ablation-only checkpoint must not be extended with
    full-sweep reps (or vice versa), since the per-rep records would
    have different keys.  A checkpoint with more reps than the requested
    ``n_reps`` is truncated to the first ``n_reps`` so the aggregate's
    ``n_reps`` and the number of summarised reps always agree (a 10-rep
    checkpoint must not silently back a ``--n-reps 5`` summary).
    """
    if not RAW_REPS_PATH.exists():
        return []
    try:
        saved = json.loads(RAW_REPS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if saved.get("include_main") != include_main:
        return []
    reps = saved.get("reps", [])
    if len(reps) > n_reps:
        print(
            f"[variance] checkpoint has {len(reps)} reps but --n-reps={n_reps}; "
            f"using first {n_reps}",
            flush=True,
        )
        reps = reps[:n_reps]
    return reps


def run_variance(
    *, n_reps: int = 10, include_main: bool = True, resume: bool = True
) -> dict[str, Any]:
    reps: list[dict[str, Any]] = (
        _load_resume(include_main=include_main, n_reps=n_reps) if resume else []
    )
    if reps:
        print(
            f"[variance] resuming from {len(reps)} completed rep(s); "
            f"targeting {n_reps} total",
            flush=True,
        )
    tmp_root = Path(tempfile.mkdtemp(prefix="paper6_variance_"))
    t0 = time.monotonic()

    for rep in range(len(reps), n_reps):
        rep_dir = tmp_root / f"rep{rep}"
        rep_record: dict[str, Any] = {}

        # --- Block 1: ablation tie (default vs minimal on seeds 0-4) ---
        default_per_seed: dict[str, int] = {}
        minimal_per_seed: dict[str, int] = {}
        for seed in ABLATION_SEEDS:
            d = run_single(
                "cert-binary",
                seed,
                budget_iterations=BUDGET_ITERATIONS,
                reflection_lm=REFLECTION_LM,
                output_dir=rep_dir / "ablation_default",
            )
            m = run_minimal_seed(
                seed,
                budget_iterations=BUDGET_ITERATIONS,
                reflection_lm=REFLECTION_LM,
            )
            default_per_seed[str(seed)] = _conv_or_raise(
                d, label=f"cert-binary(default)/seed={seed}", budget=BUDGET_ITERATIONS
            )
            minimal_per_seed[str(seed)] = _conv_or_raise(
                m, label=f"cert-binary(minimal)/seed={seed}", budget=BUDGET_ITERATIONS
            )
        rep_record["ablation"] = {
            "default_per_seed": default_per_seed,
            "minimal_per_seed": minimal_per_seed,
            "default_mean": round(statistics.mean(default_per_seed.values()), 4),
            "minimal_mean": round(statistics.mean(minimal_per_seed.values()), 4),
        }

        # --- Block 2: main-sweep parity (3 arms x 10 seeds) ---
        if include_main:
            arm_per_seed: dict[str, dict[str, int]] = {}
            arm_means: dict[str, float] = {}
            for arm in ARMS:
                per_seed: dict[str, int] = {}
                for seed in MAIN_SWEEP_SEEDS:
                    # cert-binary seeds 0-4 already ran above for the
                    # ablation default arm; re-run for the full 0-9 set
                    # to keep the sweep self-contained per rep.
                    rec = run_single(
                        arm,
                        seed,
                        budget_iterations=BUDGET_ITERATIONS,
                        reflection_lm=REFLECTION_LM,
                        output_dir=rep_dir / "main_sweep",
                    )
                    per_seed[str(seed)] = _conv_or_raise(
                        rec, label=f"{arm}/seed={seed}", budget=BUDGET_ITERATIONS
                    )
                arm_per_seed[arm] = per_seed
                arm_means[arm] = round(statistics.mean(per_seed.values()), 4)
            cert = [arm_per_seed["cert-binary"][str(s)] for s in MAIN_SWEEP_SEEDS]
            evid = [arm_per_seed["scalar-evidence"][str(s)] for s in MAIN_SWEEP_SEEDS]
            mw = mann_whitney(cert, evid)
            min_baseline = min(
                arm_means["scalar"], arm_means["scalar-evidence"]
            )
            ratio = (
                arm_means["cert-binary"] / min_baseline if min_baseline else None
            )
            rep_record["main_sweep"] = {
                "per_seed": arm_per_seed,
                "arm_means": arm_means,
                "cert_vs_evidence_p": mw["p_value"],
                "cert_vs_evidence_ratio": round(ratio, 4) if ratio is not None else None,
            }

        reps.append(rep_record)
        _checkpoint(reps, n_reps=n_reps, include_main=include_main)
        elapsed = time.monotonic() - t0
        abl = rep_record["ablation"]
        print(
            f"[variance] rep {rep + 1}/{n_reps} done "
            f"(ablation default={abl['default_mean']} minimal={abl['minimal_mean']}"
            + (
                f", sweep cert={rep_record['main_sweep']['arm_means']['cert-binary']} "
                f"evid={rep_record['main_sweep']['arm_means']['scalar-evidence']} "
                f"p={rep_record['main_sweep']['cert_vs_evidence_p']:.3f}"
                if include_main
                else ""
            )
            + f") [{elapsed:.0f}s elapsed]",
            flush=True,
        )

    # Always write the final summary, even when the loop added no new
    # reps (e.g. a complete checkpoint was resumed, or it was truncated
    # to a smaller --n-reps): the on-disk summary must match the
    # returned aggregate, and the "wrote" message must be truthful.
    _checkpoint(reps, n_reps=n_reps, include_main=include_main)
    final = _aggregate(reps, n_reps=n_reps, include_main=include_main)
    print(f"[variance] wrote {SUMMARY_PATH} ({len(reps)} reps)")
    return final


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument(
        "--skip-main-sweep",
        action="store_true",
        help="run only the ablation-tie block (faster)",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore any resume checkpoint and start from rep 0",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    run_variance(
        n_reps=args.n_reps,
        include_main=not args.skip_main_sweep,
        resume=not args.fresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
