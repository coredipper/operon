"""SWE-bench-lite Phase 2: Docker-based pass/fail evaluation.

Replaces Phase 1's LLM judge with the ground truth: apply the patch
in a Docker container, run FAIL_TO_PASS + PASS_TO_PASS tests, report
pass/fail.

Pipeline:
  1. Load SWE-bench-lite tasks (N instances)
  2. For each condition (baseline, organism, langgraph):
       a. Run the model, capture output
       b. Extract unified diff
       c. Write to <condition>_predictions.jsonl
  3. Invoke swebench.harness.run_evaluation for each condition
  4. Parse per-instance reports, aggregate resolved/total
  5. Write eval/results/swebench_phase2.json

Usage:
  pip install swebench datasets
  docker info  # confirm daemon is running
  python eval/swebench_phase2.py [--model gemma4:latest] [--n 10]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operon_ai import ATP_Store, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.langgraph_compiler import run_organism_langgraph
from operon_ai.providers.base import ProviderConfig
from operon_ai.providers.openai_compatible_provider import OpenAICompatibleProvider

from eval._patch_extraction import extract_patch


# ---------------------------------------------------------------------------
# Provider + organism (reuse Phase 1 shape)
# ---------------------------------------------------------------------------

def _make_provider(model: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="not-needed",
        base_url="http://localhost:11434/v1",
        model=model,
    )


def _format_task(instance: dict) -> str:
    repo = instance["repo"]
    problem = instance["problem_statement"]
    hints = instance.get("hints_text", "") or ""
    prompt = f"Repository: {repo}\n\nIssue:\n{problem}\n"
    if hints.strip():
        prompt += f"\nHints:\n{hints[:500]}\n"
    return prompt


def _llm_call(provider, prompt: str) -> str:
    config = ProviderConfig(max_tokens=4096)
    return provider.complete(prompt, config).content


def _build_organism(provider):
    nucleus = Nucleus(provider=provider)
    return skill_organism(
        stages=[
            SkillStage(
                name="localize",
                role="Bug Locator",
                instructions=(
                    "You are analyzing a bug report for a Python project. "
                    "Identify the exact file(s) and function(s) where the bug "
                    "occurs. Explain the root cause concisely."
                ),
                mode="fixed",
            ),
            SkillStage(
                name="edit",
                role="Patch Author",
                instructions=(
                    "Based on the bug localization, write a minimal fix. "
                    "Output a unified diff (--- a/file\\n+++ b/file). "
                    "Change only what's necessary."
                ),
                mode="fixed",
            ),
            SkillStage(
                name="verify",
                role="Patch Reviewer",
                instructions=(
                    "Review the proposed patch. Check: (1) does it fix the "
                    "reported issue? (2) any regressions? (3) diff format valid? "
                    "If any problems, describe them."
                ),
                mode="fixed",
            ),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
        budget=ATP_Store(budget=2000, silent=True),
    )


# ---------------------------------------------------------------------------
# Run conditions
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    instance_id: str
    repo: str
    condition: str
    raw_output: str
    model_patch: str
    latency_ms: float
    extract_ok: bool


def run_baseline(provider, instance: dict) -> Prediction:
    task = _format_task(instance)
    prompt = (
        f"{task}\nFix this bug. Output a unified diff "
        "(--- a/file, +++ b/file) with your fix. Be minimal."
    )
    t0 = time.monotonic()
    raw = _llm_call(provider, prompt)
    elapsed = (time.monotonic() - t0) * 1000
    patch = extract_patch(raw)
    return Prediction(
        instance["instance_id"], instance["repo"], "baseline",
        raw, patch, elapsed, bool(patch),
    )


def run_organism(provider, instance: dict) -> Prediction:
    task = _format_task(instance)
    org = _build_organism(provider)
    t0 = time.monotonic()
    result = org.run(task)
    elapsed = (time.monotonic() - t0) * 1000
    full = "\n\n".join(
        f"[{sr.stage_name}]\n{sr.output}" for sr in result.stage_results
    )
    patch = extract_patch(full)
    return Prediction(
        instance["instance_id"], instance["repo"], "organism",
        full, patch, elapsed, bool(patch),
    )


def run_langgraph(provider, instance: dict) -> Prediction:
    task = _format_task(instance)
    org = _build_organism(provider)
    t0 = time.monotonic()
    result = run_organism_langgraph(org, task=task, verify_certificates=True)
    elapsed = (time.monotonic() - t0) * 1000
    parts = [f"[{name}]\n{out}" for name, out in result.stage_outputs.items()]
    full = "\n\n".join(parts) if parts else result.output
    patch = extract_patch(full)
    return Prediction(
        instance["instance_id"], instance["repo"], "langgraph",
        full, patch, elapsed, bool(patch),
    )


# ---------------------------------------------------------------------------
# Docker evaluation
# ---------------------------------------------------------------------------

def _write_predictions(preds: list[Prediction], path: Path, model_name: str) -> None:
    """Write predictions in swebench harness format (JSONL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in preds:
            f.write(json.dumps({
                "instance_id": p.instance_id,
                "model_patch": p.model_patch,
                "model_name_or_path": model_name,
            }) + "\n")


class HarnessFailed(Exception):
    """Raised when the swebench harness itself failed to run."""


def _run_harness(
    predictions_path: Path, run_id: str, report_dir: Path, timeout: int = 600
) -> None:
    """Invoke swebench.harness.run_evaluation. Report lands in *report_dir*.

    Raises :class:`HarnessFailed` if the harness exits non-zero.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "SWE-bench/SWE-bench_Lite",
        "--predictions_path", str(predictions_path.resolve()),
        "--run_id", run_id,
        "--max_workers", "2",
        "--timeout", str(timeout),
        "--cache_level", "env",
    ]
    print(f"  Harness: python -m swebench.harness.run_evaluation "
          f"--run_id {run_id} ...")
    try:
        # Harness writes {model_name}.{run_id}.json to CWD — run in report_dir
        subprocess.run(cmd, check=True, cwd=str(report_dir))
    except subprocess.CalledProcessError as e:
        raise HarnessFailed(
            f"swebench.harness.run_evaluation exited {e.returncode}"
        ) from e


# Per-instance evaluation state
EVAL_RESOLVED = "resolved"          # patch applied, tests passed
EVAL_UNRESOLVED = "unresolved"      # patch applied but tests failed
EVAL_ERROR = "error"                # patch failed to apply, timeout, etc.
EVAL_EMPTY = "empty_patch"          # no patch produced
EVAL_NOT_EVALUATED = "not_evaluated"  # harness did not report on this instance


def _parse_reports(
    report_dir: Path, run_id: str, model_name: str
) -> tuple[dict[str, str], bool]:
    """Parse harness output.

    Returns (status_by_instance, report_found). *status_by_instance* maps
    instance_id → one of the EVAL_* constants. Instances not mentioned in
    the report get EVAL_NOT_EVALUATED by the caller.
    """
    status: dict[str, str] = {}
    summary = report_dir / f"{model_name}.{run_id}.json"
    if not summary.exists():
        return status, False
    try:
        data = json.loads(summary.read_text())
        for inst_id in data.get("resolved_ids", []):
            status[inst_id] = EVAL_RESOLVED
        for inst_id in data.get("unresolved_ids", []):
            status[inst_id] = EVAL_UNRESOLVED
        for inst_id in data.get("error_ids", []):
            status[inst_id] = EVAL_ERROR
        for inst_id in data.get("empty_patch_ids", []):
            status[inst_id] = EVAL_EMPTY
        return status, True
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  WARN: failed to parse {summary}: {e}")
        return status, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SWE-bench-lite Phase 2 (Docker)")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--conditions", default="baseline,organism,langgraph")
    parser.add_argument("--skip-harness", action="store_true",
                        help="Only generate predictions, skip Docker evaluation")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]

    if "langgraph" in conditions:
        from operon_ai.convergence.langgraph_compiler import HAS_LANGGRAPH
        if not HAS_LANGGRAPH:
            print("WARNING: langgraph not installed, skipping")
            conditions = [c for c in conditions if c != "langgraph"]

    provider = _make_provider(args.model)

    # Probe model
    try:
        probe = _llm_call(provider, "Say ok.")
        print(f"Model probe: {probe.strip()[:30]}")
    except Exception as e:
        print(f"ERROR: cannot reach model: {e}")
        sys.exit(1)

    # Probe Docker (unless skipping)
    if not args.skip_harness:
        try:
            subprocess.run(["docker", "info"], check=True,
                           capture_output=True, timeout=10)
            print("Docker:      OK")
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired) as e:
            print(f"ERROR: Docker unavailable: {e}")
            print("Hint: start Docker Desktop, or pass --skip-harness")
            sys.exit(1)

    # Load dataset
    from datasets import load_dataset
    print(f"\nLoading SWE-bench-lite...")
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    instances = list(ds.select(range(args.offset, min(args.offset + args.n, len(ds)))))
    print(f"Selected {len(instances)} instances")

    condition_fns = {
        "baseline": run_baseline,
        "organism": run_organism,
        "langgraph": run_langgraph,
    }

    # -- Phase A: generate predictions -----------------------------------
    run_id = f"phase2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    preds_by_cond: dict[str, list[Prediction]] = {c: [] for c in conditions}

    for i, instance in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(instances)}] {instance['instance_id']}")
        print(f"  repo:  {instance['repo']}")
        print(f"  issue: {instance['problem_statement'][:70]}...")

        for cond in conditions:
            try:
                p = condition_fns[cond](provider, instance)
            except Exception as e:
                print(f"  {cond:10s} ERROR: {e}")
                p = Prediction(
                    instance["instance_id"], instance["repo"], cond,
                    f"ERROR: {e}", "", 0.0, False,
                )
            preds_by_cond[cond].append(p)
            status = "ok" if p.extract_ok else "no-patch"
            print(f"  {cond:10s} extract={status:8s} latency={p.latency_ms:.0f}ms "
                  f"(patch {len(p.model_patch)}B)")

    # -- Phase B: write predictions & run harness ------------------------
    results_dir = Path("eval/results/swebench_phase2_predictions") / run_id
    status_by_cond: dict[str, dict[str, str]] = {c: {} for c in conditions}
    harness_ok_by_cond: dict[str, bool] = {c: False for c in conditions}

    for cond in conditions:
        model_name = f"operon-{cond}"
        pred_path = results_dir / f"{cond}.jsonl"
        _write_predictions(preds_by_cond[cond], pred_path, model_name)
        print(f"\n  Wrote {len(preds_by_cond[cond])} predictions to {pred_path}")

        if args.skip_harness:
            continue

        print(f"\n{'='*60}")
        print(f"Running harness for {cond} ({len(preds_by_cond[cond])} instances)")
        print(f"{'='*60}")
        report_dir = Path("eval/results/swebench_phase2_reports") / f"{run_id}_{cond}"
        try:
            _run_harness(pred_path, f"{run_id}_{cond}", report_dir, args.timeout)
        except HarnessFailed as e:
            print(f"  HARNESS FAILED for {cond}: {e}")
            print(f"  Instances for {cond} will be recorded as not_evaluated")
            continue
        statuses, report_found = _parse_reports(
            report_dir, f"{run_id}_{cond}", model_name
        )
        status_by_cond[cond] = statuses
        harness_ok_by_cond[cond] = report_found

    # -- Phase C: aggregate & save ---------------------------------------
    summary: dict[str, dict] = {}
    for cond in conditions:
        preds = preds_by_cond[cond]
        statuses = status_by_cond[cond]
        harness_ok = harness_ok_by_cond[cond]
        n = len(preds)
        extracted = sum(1 for p in preds if p.extract_ok)

        # Only include instances with a real harness verdict in the denominator.
        evaluated = [
            p for p in preds
            if statuses.get(p.instance_id, EVAL_NOT_EVALUATED) != EVAL_NOT_EVALUATED
        ]
        passed = sum(1 for p in evaluated if statuses.get(p.instance_id) == EVAL_RESOLVED)

        summary[cond] = {
            "n": n,
            "patch_extracted": extracted,
            "evaluated": len(evaluated),
            "resolved": passed,
            "resolved_rate": (
                round(passed / len(evaluated), 3) if evaluated else None
            ),
            "harness_ran": harness_ok,
            "mean_latency_ms": round(
                sum(p.latency_ms for p in preds) / n, 0
            ) if n else 0,
        }

    out_path = Path("eval/results/swebench_phase2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "model": args.model,
        "dataset": "SWE-bench/SWE-bench_Lite",
        "run_id": run_id,
        "n_instances": len(instances),
        "offset": args.offset,
        "conditions": conditions,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "skip_harness": args.skip_harness,
        "results": [
            {
                "instance_id": p.instance_id,
                "repo": p.repo,
                "condition": p.condition,
                "patch_extracted": p.extract_ok,
                "patch_size_bytes": len(p.model_patch),
                "latency_ms": p.latency_ms,
                "eval_status": status_by_cond[p.condition].get(
                    p.instance_id, EVAL_NOT_EVALUATED
                ),
            }
            for cond in conditions for p in preds_by_cond[cond]
        ],
        "summary": summary,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond, s in summary.items():
        rate_str = (
            f"{s['resolved_rate']:.1%}"
            if s["resolved_rate"] is not None else "N/A"
        )
        harness_str = "ok" if s["harness_ran"] else "FAILED"
        print(
            f"  {cond:10s} resolved={s['resolved']}/{s['evaluated']} "
            f"({rate_str})  n={s['n']}  "
            f"extracted={s['patch_extracted']}/{s['n']}  "
            f"harness={harness_str}  "
            f"latency={s['mean_latency_ms']:.0f}ms"
        )
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
