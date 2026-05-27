"""Paper 6 — obligation-formatter ablation.

On seeds 0--4, runs ``cert-binary`` with a *minimal* obligation
formatter (one line per violating window: ``constraint violated:
window k``, no numbers, no theorem framing) and compares against
the default-formatter results from the main sweep
(``eval/results/theorem_6/cert-binary/seed_{0..4}.json``).

Writes per-seed JSONs to
``eval/results/theorem_6/ablation_minimal/seed_{N}.json`` and an
aggregate to ``eval/results/theorem_6/ablation_summary.json``.

The aggregate mirrors the main analysis policy
(``analysis_theorem_6.py``): mocked runs are filtered, errored runs
are excluded and reported separately under
``errored_runs_excluded``, non-converged runs are imputed at
``budget_iterations`` for the central-tendency statistic, and
budget + LM metadata are propagated so a single
``ablation_summary.json`` is sufficient to reconstruct what was run.

Usage::

    python -m eval.convergence.theorem_6_ablation
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

from .analysis_theorem_6 import load_arm_records
from .synthetic_signal_harness import (
    THROTTLE_COMPONENT_NAME,
    SyntheticDataset,
    TaskConfig,
    Trajectory,
    seed_candidate,
)
from .theorem_6_experiment import (
    MUTABLE_COMPONENTS,
    THEOREM,
    THROTTLE_REFLECTION_TEMPLATE,
    _build_harness,
    _first_convergence_iter,
    _IterationRecorder,
    _ThrottleOnlySelector,
)

DEFAULT_BUDGET_ITERATIONS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_VALSET_SIZE = 16
DEFAULT_REFLECTION_LM = "ollama/gemma4"


def minimal_obligation_formatter(verification: Any, trajectory: Any) -> str:
    """Per-window obligation text with NO numbers and NO theorem framing.

    Emits exactly one line per violating window:
    ``constraint violated: window <k>``.  Strips out the per-window
    mean values, the threshold, and the theorem name.  Used by the
    ablation in paper 6 §3.5 to test whether the signal in cert-binary
    comes from the obligation evidence's numeric content or from
    structural knowledge of which window failed.
    """
    if isinstance(trajectory, Trajectory):
        violating = trajectory.violating_windows
    else:
        violating = []
    if not violating:
        return "constraint satisfied"
    return "\n".join(f"constraint violated: window {k}" for k in violating)


def run_minimal_seed(
    seed: int,
    *,
    budget_iterations: int = DEFAULT_BUDGET_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    valset_size: int = DEFAULT_VALSET_SIZE,
    reflection_lm: str = DEFAULT_REFLECTION_LM,
) -> dict[str, Any]:
    cfg = TaskConfig()
    harness = _build_harness(cfg, seed)
    adapter = OperonCertificateAdapter(
        theorem=THEOREM,
        harness=harness,
        components=list(MUTABLE_COMPONENTS),
        obligation_formatter=minimal_obligation_formatter,
        retain_trajectories_for_reflection=True,
        source=f"paper6/ablation-minimal/seed={seed}",
    )
    trainset = list(SyntheticDataset(size=batch_size, offset=seed * 1000))
    valset = list(
        SyntheticDataset(size=valset_size, offset=seed * 1000 + batch_size + 1)
    )

    recorder = _IterationRecorder()
    import gepa

    max_metric_calls = budget_iterations * batch_size * 2
    t_start = time.monotonic()
    gepa_error: str | None = None
    try:
        gepa.optimize(
            seed_candidate=seed_candidate(),
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            seed=seed,
            raise_on_exception=False,
            display_progress_bar=False,
            callbacks=[recorder],
            module_selector=_ThrottleOnlySelector(),
            reflection_prompt_template={
                THROTTLE_COMPONENT_NAME: THROTTLE_REFLECTION_TEMPLATE,
            },
        )
    except Exception as exc:  # noqa: BLE001
        gepa_error = f"{type(exc).__name__}: {exc}"
    wallclock = time.monotonic() - t_start

    convergence_iter = _first_convergence_iter(recorder.iters)
    return {
        "arm": "cert-binary-minimal",
        "seed": seed,
        "budget_iterations": budget_iterations,
        "batch_size": batch_size,
        "max_metric_calls": max_metric_calls,
        "convergence_iteration": convergence_iter,
        "wallclock_s": round(wallclock, 3),
        "iteration_events": recorder.iters,
        "reflection_lm": str(reflection_lm),
        "mock_reflection_lm": False,
        "gepa_error": gepa_error,
        "formatter": "minimal (constraint violated: window k)",
        "theorem": THEOREM,
    }


def _arm_stats(
    records: list[dict[str, Any]], *, budget: int, errored_excluded: int
) -> dict[str, Any]:
    """Aggregate the arm-level fields that match the main analysis schema.

    Imputes non-converged runs at ``budget`` for the central-tendency
    statistic, matching the methods §3.5 contract.  Reports the raw
    convergence iterations (post-filter) so downstream consumers can
    re-derive their own central tendency without re-running the sweep.
    """
    convergence_iters = [
        r["convergence_iteration"] if r["convergence_iteration"] is not None else budget
        for r in records
    ]
    n_converged = sum(
        1
        for r in records
        if r["convergence_iteration"] is not None
    )
    mean_iters = statistics.mean(convergence_iters) if convergence_iters else None
    return {
        "n_runs": len(records),
        "n_converged": n_converged,
        "errored_runs_excluded": errored_excluded,
        "mean_iters": mean_iters,
        "convergence_iterations": convergence_iters,
    }


def main() -> int:
    out_dir = Path("eval/results/theorem_6/ablation_minimal")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2, 3, 4]
    budget = DEFAULT_BUDGET_ITERATIONS
    reflection_lm = DEFAULT_REFLECTION_LM

    # Run the minimal-formatter arm.  Mocked-LM runs are not produced
    # by this driver (the driver never sets the mock flag); errored
    # runs are written to disk with ``gepa_error`` populated and are
    # filtered out before computing summary statistics, matching the
    # main analysis policy.
    for seed in seeds:
        print(f"[paper6-ablation] seed={seed}")
        record = run_minimal_seed(
            seed,
            budget_iterations=budget,
            reflection_lm=reflection_lm,
        )
        (out_dir / f"seed_{seed}.json").write_text(
            json.dumps(record, indent=2, default=str)
        )
        print(
            f"[paper6-ablation] seed={seed} "
            f"convergence={record['convergence_iteration']} "
            f"wallclock={record['wallclock_s']}s "
            f"error={record['gepa_error']}"
        )

    # Reload through the main-analysis loader so the same filter policy
    # applies on both sides of the comparison.  ``load_arm_records``
    # drops mocked and errored records by default, the contract the
    # paper's results table relies on.  Both arms are then filtered to
    # the declared ``seeds`` so a stale ``seed_*.json`` left in either
    # directory cannot silently inflate ``n_runs`` / shift ``mean_iters``.
    default_dir = Path("eval/results/theorem_6/cert-binary")
    default_all = load_arm_records(default_dir, include_mock=False, include_errored=True)
    default_records = [r for r in default_all if r["seed"] in seeds]
    default_kept = [r for r in default_records if r.get("gepa_error") is None]
    default_excluded = len(default_records) - len(default_kept)

    minimal_all = load_arm_records(out_dir, include_mock=False, include_errored=True)
    minimal_records = [r for r in minimal_all if r["seed"] in seeds]
    minimal_kept = [r for r in minimal_records if r.get("gepa_error") is None]
    minimal_excluded = len(minimal_records) - len(minimal_kept)

    # Report the LM actually used per arm (the default arm reads it
    # from disk because the sweep may have run with a different LM
    # than this ablation invocation).
    default_lms = sorted({str(r.get("reflection_lm")) for r in default_kept})
    minimal_lms = sorted({str(r.get("reflection_lm")) for r in minimal_kept})

    summary = {
        "seeds": seeds,
        "budget_iterations": budget,
        "reflection_lm": reflection_lm,
        "default_formatter": {
            **_arm_stats(
                default_kept, budget=budget, errored_excluded=default_excluded
            ),
            "source_dir": str(default_dir),
            "reflection_lm_observed": default_lms,
        },
        "minimal_formatter": {
            **_arm_stats(
                minimal_kept, budget=budget, errored_excluded=minimal_excluded
            ),
            "source_dir": str(out_dir),
            "reflection_lm_observed": minimal_lms,
        },
    }
    summary_path = Path("eval/results/theorem_6/ablation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[paper6-ablation] wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
