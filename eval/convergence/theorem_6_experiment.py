"""Paper 6 experiment driver — Theorem 3 conjecture test.

Runs GEPA with one of three adapter arms on the synthetic
``behavioral_stability_windowed`` task, for a given seed.  Emits a
per-iteration trajectory JSON file at
``eval/results/theorem_6/{arm}/{seed}.json``.

Usage::

    # single run
    python -m eval.convergence.theorem_6_experiment \
        --arm cert-binary --seed 0 --budget 50

    # full sweep (3 arms × 10 seeds)
    python -m eval.convergence.theorem_6_experiment --sweep

    # smoke (tiny budget, mock reflection LM, for CI)
    python -m eval.convergence.theorem_6_experiment \
        --arm cert-binary --seed 0 --budget 3 --mock-reflection-lm

Arms
----
- ``cert-binary``       — ``OperonCertificateAdapter`` (treatment)
- ``scalar``            — ``ScalarRewardAdapter`` (baseline)
- ``scalar-evidence``   — ``ScalarWithEvidenceAdapter`` (active control)

Reflection LM
-------------
Defaults to ``ollama/gemma4`` via LiteLLM (matches Bogdan's local setup
per memory ``reference_repos.md``).  ``--mock-reflection-lm`` swaps in
a deterministic dummy that returns a fixed, parseable mutation — used
for smoke tests, *not* for measured runs.  Mocked runs are tagged in
the output JSON so analysis can filter them out.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from operon_ai.convergence.gepa_adapter import OperonCertificateAdapter

from .scalar_reward_adapter import ScalarRewardAdapter
from .scalar_with_evidence_adapter import ScalarWithEvidenceAdapter
from .synthetic_signal_harness import (
    SEED_COMPONENT_NAME,
    SyntheticDataset,
    TaskConfig,
    Trajectory,
    render_window_evidence,
    run_rollout,
    seed_candidate,
)

# ---------------------------------------------------------------------------
# Task-specific obligation formatter (Roborev #854 H1, #855)
# ---------------------------------------------------------------------------


def stability_windowed_obligation_formatter(
    verification: Any, trajectory: Any
) -> str:
    """Per-window obligation text for ``behavioral_stability_windowed``.

    Emits exactly one framing line (``Theorem: <name> [FAILED|HOLDS]``)
    followed by the shared evidence block
    (:func:`render_window_evidence`).  Scalar-evidence emits a
    ``Score: <value>`` line followed by the *same* shared block, so the
    two arms' renderings agree after the first line by construction.
    Any future drift fails ``TestCertBinaryContentMatched`` immediately
    because the test does a byte-for-byte comparison on the tail.
    """
    status = "HOLDS" if verification.holds else "FAILED"
    header = f"Theorem: {verification.certificate.theorem} [{status}]"
    if isinstance(trajectory, Trajectory):
        return header + "\n" + render_window_evidence(trajectory)
    # Fallback: trajectory not a Trajectory instance.  Use the verifier's
    # aggregate evidence so the formatter is total (never raises), but
    # this path is not exercised in the paper-6 experiment.
    lines = [header, "Evidence:"]
    for key, value in verification.evidence.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------

ARMS = ("cert-binary", "scalar", "scalar-evidence")
THEOREM = "behavioral_stability_windowed"


def _build_harness(
    config: TaskConfig, run_seed: int
) -> Callable[[dict[str, str], Any], tuple[Any, Any, dict[str, Any]]]:
    """Return a closure ``(candidate, data_inst) -> rollout`` bound to config+seed."""

    def harness(
        candidate: dict[str, str], data_inst: int
    ) -> tuple[Any, Any, dict[str, Any]]:
        text = candidate.get(SEED_COMPONENT_NAME, "")
        return run_rollout(text, data_inst, config=config, run_seed=run_seed)

    return harness


def build_adapter(arm: str, config: TaskConfig, run_seed: int) -> Any:
    """Instantiate the adapter for the named arm."""
    harness = _build_harness(config, run_seed)
    if arm == "cert-binary":
        return OperonCertificateAdapter(
            theorem=THEOREM,
            harness=harness,
            components=[SEED_COMPONENT_NAME],
            # Trajectory-aware formatter so cert-binary and scalar-evidence
            # carry the same per-window obligation *content*; the only
            # remaining difference is binarization + the theorem framing
            # line.  See Roborev #854 H1.
            obligation_formatter=stability_windowed_obligation_formatter,
            # The formatter needs the rollout trajectory (window means,
            # violating indices) at reflection time; opt in to trajectory
            # side-channeling regardless of ``capture_traces``.  See
            # Roborev #856 / #857.
            retain_trajectories_for_reflection=True,
            source=f"paper6/cert-binary/seed={run_seed}",
        )
    if arm == "scalar":
        return ScalarRewardAdapter(harness=harness)
    if arm == "scalar-evidence":
        return ScalarWithEvidenceAdapter(harness=harness)
    raise ValueError(f"unknown arm: {arm!r} (expected one of {ARMS})")


# ---------------------------------------------------------------------------
# Mock reflection LM — for smoke tests only
# ---------------------------------------------------------------------------


class _MockReflectionLM:
    """Deterministic reflection LM that always returns a lower-throttle prompt.

    Used in smoke tests to validate the wiring without hitting Ollama.
    Tracks internal state (monotonically decreasing throttle) so each
    call strictly improves the candidate, guaranteeing convergence in a
    bounded number of calls.  Mocked runs are diagnostic only; analysis
    scripts filter on the ``mock_reflection_lm`` flag in output JSON.
    """

    def __init__(self, decrement: float = 0.3, floor: float = 0.1) -> None:
        self.decrement = decrement
        self.floor = floor
        self.call_count = 0
        self._current = 1.0

    def __call__(self, prompt: str) -> str:  # GEPA LM protocol
        self.call_count += 1
        del prompt  # Ignore — mock mutates its own state, not the prompt's.
        self._current = max(self.floor, self._current - self.decrement)
        # Emit a ``CONFIG:``-prefixed line so the harness's
        # parse_throttle reads the assignment as real config instead
        # of defaulting to 1.0 (Roborev #870).  We intentionally do
        # NOT wrap in a markdown fenced code block: GEPA's default
        # reflective proposer strips fences during normalization, so
        # a fenced mock output would arrive at parse_throttle as
        # bare ``policy_throttle = X`` and every smoke run would
        # silently score as "never converged."  The plain CONFIG:
        # prefix survives GEPA's round-trip intact.
        return (
            f"<{SEED_COMPONENT_NAME}>\n"
            "You operate a synthetic throttled source.\n\n"
            f"CONFIG: policy_throttle = {self._current:.3f}\n"
            f"</{SEED_COMPONENT_NAME}>"
        )


# ---------------------------------------------------------------------------
# Convergence detection via GEPA result inspection
# ---------------------------------------------------------------------------


def _val_scores_from_recorder(
    iter_events: list[dict[str, Any]],
) -> list[list[float]]:
    """Convert iteration-end events into a per-iteration score trajectory.

    Each iteration's ``best_score`` becomes a single-entry list so the
    downstream convergence detector — which is structured to read a
    list-of-lists for future compatibility with per-instance scores —
    keeps its existing shape.
    """
    trajectory: list[list[float]] = []
    for event in iter_events:
        best = event.get("best_score")
        if best is None:
            trajectory.append([])
        else:
            trajectory.append([float(best)])
    return trajectory


def _first_convergence_iter(
    iter_events: list[dict[str, Any]],
    *,
    consecutive: int = 3,
    threshold: float = 1.0,
) -> int | None:
    """GEPA iteration number at which convergence was first observed.

    Returns the *iteration field of the first event in the streak*,
    not a list index.  GEPA emits iteration 0 for the initial evaluation
    and iterations 1, 2, ... for each mutation step, so this return
    value is the count of GEPA iterations run, which matches the paper's
    ``mutations-to-convergence`` metric (see Roborev #854 M1).

    A run converges when ``best_score >= threshold`` for ``consecutive``
    consecutive iterations; the reported iteration is the first event
    in that streak.
    """
    if not iter_events:
        return None
    streak = 0
    for i, event in enumerate(iter_events):
        best = event.get("best_score")
        if best is not None and float(best) >= threshold:
            streak += 1
            if streak >= consecutive:
                first = iter_events[i - consecutive + 1]
                # Prefer GEPA's own iteration counter; fall back to list
                # index if the event is malformed.
                reported = first.get("iteration")
                value = (
                    int(reported)
                    if reported is not None
                    else i - consecutive + 1
                )
                # Clamp to ≥ 1: GEPA's iteration 0 is the initial
                # evaluation, not a mutation step.  If the streak
                # starts at iter 0, the candidate converged
                # pre-mutation — report as 1 (the first mutation
                # step, which continues the passing streak) so the
                # mutations-to-convergence metric stays 1-based as
                # the paper defines it (Roborev #864 M1).
                return max(1, value)
        else:
            streak = 0
    return None


# ---------------------------------------------------------------------------
# Per-iteration callback hook
# ---------------------------------------------------------------------------


class _IterationRecorder:
    """GEPACallback implementation: records best-valset-score per iteration.

    Implements only the two lifecycle hooks we need
    (``on_iteration_end`` and ``on_optimization_end``).  GEPA's
    callback Protocol is ``@runtime_checkable`` and uses ``hasattr`` —
    any method we don't implement is simply not called.

    We read the best valset aggregate score out of the event's
    ``state`` each iteration; this is the trajectory used downstream
    for convergence detection.
    """

    def __init__(self) -> None:
        self.iters: list[dict[str, Any]] = []
        self._t0 = time.monotonic()
        self.total_iterations: int | None = None
        self.best_candidate_idx: int | None = None

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        iteration = event.get("iteration")
        state = event.get("state")
        proposal_accepted = event.get("proposal_accepted")
        best_score: float | None = None
        if state is not None and hasattr(state, "program_full_scores_val_set"):
            scores = list(state.program_full_scores_val_set)
            if scores:
                best_score = max(scores)
        self.iters.append({
            "iteration": iteration,
            "best_score": best_score,
            "proposal_accepted": proposal_accepted,
            "wallclock_s": round(time.monotonic() - self._t0, 3),
        })

    def on_optimization_end(self, event: dict[str, Any]) -> None:
        self.total_iterations = event.get("total_iterations")
        self.best_candidate_idx = event.get("best_candidate_idx")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_single(
    arm: str,
    seed: int,
    *,
    budget_iterations: int = 50,
    batch_size: int = 16,
    reflection_lm: Any = "ollama/gemma4",
    valset_size: int = 16,
    output_dir: Path = Path("eval/results/theorem_6"),
    config: TaskConfig | None = None,
    mock_reflection_lm: bool = False,
) -> dict[str, Any]:
    """Run one ``(arm, seed)`` cell and write the trajectory JSON.

    Returns the record that was written.
    """
    cfg = config or TaskConfig()
    adapter = build_adapter(arm, cfg, seed)
    trainset = SyntheticDataset(size=batch_size, offset=seed * 1000)
    valset = SyntheticDataset(
        size=valset_size, offset=seed * 1000 + batch_size + 1
    )

    lm_obj: Any
    mock = None
    if mock_reflection_lm:
        mock = _MockReflectionLM()
        lm_obj = mock
    else:
        lm_obj = reflection_lm

    recorder = _IterationRecorder()

    # GEPA's `optimize` imports heavy dependencies; defer the import.
    import gepa

    # One "GEPA iteration" ≈ (reflection_minibatch_size * 2) metric calls
    # for pareto selection.  Translate budget_iterations into max_metric_calls.
    max_metric_calls = budget_iterations * batch_size * 2

    t_start = time.monotonic()
    try:
        _ = gepa.optimize(
            seed_candidate=seed_candidate(),
            trainset=list(trainset),
            valset=list(valset),
            adapter=adapter,
            reflection_lm=lm_obj,
            max_metric_calls=max_metric_calls,
            seed=seed,
            raise_on_exception=False,
            display_progress_bar=False,
            callbacks=[recorder],
        )
        gepa_error = None
    except Exception as exc:  # noqa: BLE001 - experiment driver should capture failures
        gepa_error = f"{type(exc).__name__}: {exc}"
    wallclock = time.monotonic() - t_start

    val_score_traj = _val_scores_from_recorder(recorder.iters)
    convergence_iter = _first_convergence_iter(recorder.iters)

    record = {
        "arm": arm,
        "seed": seed,
        "budget_iterations": budget_iterations,
        "batch_size": batch_size,
        "max_metric_calls": max_metric_calls,
        "config": dataclasses.asdict(cfg),
        "val_scores_per_iteration": val_score_traj,
        "convergence_iteration": convergence_iter,
        "wallclock_s": round(wallclock, 3),
        "iteration_events": recorder.iters,
        "reflection_lm": str(reflection_lm) if not mock_reflection_lm else "mock",
        "mock_reflection_lm": mock_reflection_lm,
        "mock_call_count": mock.call_count if mock else None,
        "gepa_error": gepa_error,
        "theorem": THEOREM,
    }

    out_dir = output_dir / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))
    return record


def run_sweep(
    *,
    arms: tuple[str, ...] = ARMS,
    seeds: range = range(10),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run every ``(arm, seed)`` cell sequentially, return all records."""
    records: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            print(f"[paper6] arm={arm} seed={seed}", flush=True)
            records.append(run_single(arm, seed, **kwargs))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=50, help="max GEPA iterations")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--valset-size", type=int, default=16)
    parser.add_argument("--reflection-lm", default="ollama/gemma4")
    parser.add_argument("--output-dir", default="eval/results/theorem_6")
    parser.add_argument("--sweep", action="store_true", help="run all arms × 10 seeds")
    parser.add_argument(
        "--mock-reflection-lm",
        action="store_true",
        help="use deterministic mock LM (smoke tests only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    shared = dict(
        budget_iterations=args.budget,
        batch_size=args.batch_size,
        valset_size=args.valset_size,
        reflection_lm=args.reflection_lm,
        output_dir=Path(args.output_dir),
        mock_reflection_lm=args.mock_reflection_lm,
    )
    if args.sweep:
        records = run_sweep(**shared)
        print(f"[paper6] wrote {len(records)} records")
        return 0
    if args.arm is None:
        print("Must pass --arm or --sweep", file=sys.stderr)
        return 2
    record = run_single(args.arm, args.seed, **shared)
    print(
        f"[paper6] arm={record['arm']} seed={record['seed']} "
        f"convergence={record['convergence_iteration']} "
        f"wallclock={record['wallclock_s']}s "
        f"error={record['gepa_error']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
