"""Paper 6 — obligation-formatter ablation.

On seeds 0--4, runs ``cert-binary`` with a *minimal* obligation
formatter (one line per violating window: ``constraint violated:
window k``, no numbers, no theorem framing) and compares against
the default-formatter results from the main sweep
(``eval/results/theorem_6/cert-binary/seed_{0..4}.json``).

Writes per-seed JSONs to
``eval/results/theorem_6/ablation_minimal/seed_{N}.json`` and an
aggregate to ``eval/results/theorem_6/ablation_summary.json``.

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


def run_minimal_seed(seed: int, budget_iterations: int = 50) -> dict[str, Any]:
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
    batch_size = 16
    trainset = list(SyntheticDataset(size=batch_size, offset=seed * 1000))
    valset = list(
        SyntheticDataset(size=16, offset=seed * 1000 + batch_size + 1)
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
            reflection_lm="ollama/gemma4",
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
    record = {
        "arm": "cert-binary-minimal",
        "seed": seed,
        "convergence_iteration": convergence_iter,
        "wallclock_s": round(wallclock, 3),
        "iteration_events": recorder.iters,
        "gepa_error": gepa_error,
        "formatter": "minimal (constraint violated: window k)",
        "theorem": THEOREM,
    }
    return record


def main() -> int:
    out_dir = Path("eval/results/theorem_6/ablation_minimal")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2, 3, 4]
    minimal_records: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[paper6-ablation] seed={seed}")
        record = run_minimal_seed(seed)
        (out_dir / f"seed_{seed}.json").write_text(
            json.dumps(record, indent=2, default=str)
        )
        minimal_records.append(record)
        print(
            f"[paper6-ablation] seed={seed} "
            f"convergence={record['convergence_iteration']} "
            f"wallclock={record['wallclock_s']}s "
            f"error={record['gepa_error']}"
        )

    default_records: list[dict[str, Any]] = []
    default_dir = Path("eval/results/theorem_6/cert-binary")
    for seed in seeds:
        path = default_dir / f"seed_{seed}.json"
        default_records.append(json.loads(path.read_text()))

    def mean_iters(records: list[dict[str, Any]], budget: int = 50) -> float:
        vals = [
            r["convergence_iteration"] if r["convergence_iteration"] is not None else budget
            for r in records
        ]
        return statistics.mean(vals)

    default_iters = [r["convergence_iteration"] for r in default_records]
    minimal_iters = [r["convergence_iteration"] for r in minimal_records]
    summary = {
        "seeds": seeds,
        "default_formatter": {
            "convergence_iterations": default_iters,
            "mean_iters": mean_iters(default_records),
        },
        "minimal_formatter": {
            "convergence_iterations": minimal_iters,
            "mean_iters": mean_iters(minimal_records),
        },
    }
    summary_path = Path("eval/results/theorem_6/ablation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[paper6-ablation] wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
