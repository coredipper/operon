"""SWE-bench-lite evaluation via Operon organisms and LangGraph.

Phase 1: prompt-only evaluation. Loads SWE-bench-lite tasks, runs
organisms, judges patch quality via LLM. No Docker, no patch
application, no test execution.

Compares 3 conditions:
  - BASELINE: raw LLM call
  - ORGANISM: 3-stage skill_organism (localize → edit → verify)
  - LANGGRAPH: same organism via run_organism_langgraph()

Usage:
  pip install datasets
  python eval/swebench_lite.py [--model gemma4:latest] [--n 10]
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operon_ai import ATP_Store, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.langgraph_compiler import run_organism_langgraph
from operon_ai.providers.openai_compatible_provider import OpenAICompatibleProvider
from operon_ai.providers.base import ProviderConfig


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

def _make_provider(model: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="not-needed",
        base_url="http://localhost:11434/v1",
        model=model,
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_JUDGE_RUBRIC = (
    "You are evaluating a proposed bug fix for an open-source Python project. "
    "Score the OUTPUT on a 0.0-1.0 scale:\n"
    "- Bug identification (~30%): correctly identifies the root cause\n"
    "- Patch quality (~40%): fix is correct, minimal, and plausible\n"
    "- Format (~15%): output contains a valid unified diff or clear code change\n"
    "- Completeness (~15%): addresses the full issue, not just part of it\n\n"
    "Score anchors:\n"
    "  0.0-0.2: Wrong file, wrong bug, or refuses\n"
    "  0.3-0.4: Identifies area but fix is incorrect\n"
    "  0.5-0.6: Partially correct fix, missing edge cases\n"
    "  0.7-0.8: Good fix, minor issues\n"
    "  0.9-1.0: Correct fix with proper diff format\n\n"
    "Return ONLY JSON: {\"score\": <float>}\n\n"
)

_CONDITION_CONTEXT = {
    "baseline": "BASELINE: single LLM call asked to fix the bug and output a diff.",
    "organism": "ORGANISM: 3-stage pipeline (localize → edit → verify). All stage outputs shown.",
    "langgraph": "LANGGRAPH: same 3-stage pipeline compiled to LangGraph. All stage outputs shown.",
}


def _extract_score(content: str) -> float:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return float(json.loads(content)["score"])
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    m = re.search(r'"score"\s*:\s*([\d.]+)', content)
    if m:
        return float(m.group(1))
    m = re.search(r'\b(0\.\d+|1\.0)\b', content)
    if m:
        return float(m.group(1))
    return 0.5


def _judge(provider, task_prompt: str, output: str, condition: str) -> float:
    context = _CONDITION_CONTEXT.get(condition, "")
    judge_prompt = (
        f"{_JUDGE_RUBRIC}"
        f"CONDITION: {context}\n\n"
        f"ISSUE:\n{task_prompt[:1500]}\n\n"
        f"OUTPUT:\n{output[:6000]}\n"
    )
    config = ProviderConfig(max_tokens=2048, temperature=0.0)
    resp = provider.complete(judge_prompt, config)
    return _extract_score(resp.content)


# ---------------------------------------------------------------------------
# Task formatting
# ---------------------------------------------------------------------------

def _format_task(instance: dict) -> str:
    """Build a task prompt from a SWE-bench instance."""
    repo = instance["repo"]
    problem = instance["problem_statement"]
    hints = instance.get("hints_text", "")

    prompt = (
        f"Repository: {repo}\n\n"
        f"Issue:\n{problem}\n"
    )
    if hints and hints.strip():
        prompt += f"\nHints:\n{hints[:500]}\n"

    return prompt


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    instance_id: str
    repo: str
    condition: str
    quality: float
    output: str
    latency_ms: float
    stages: list[str]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def _llm_call(provider, prompt: str) -> str:
    config = ProviderConfig(max_tokens=4096)
    resp = provider.complete(prompt, config)
    return resp.content


def run_baseline(provider, instance: dict) -> RunResult:
    task = _format_task(instance)
    prompt = (
        f"{task}\n"
        "Fix this bug. Output a unified diff (--- a/file, +++ b/file format) "
        "with your fix. Be minimal — change only what's necessary."
    )
    t0 = time.monotonic()
    output = _llm_call(provider, prompt)
    elapsed = (time.monotonic() - t0) * 1000
    quality = _judge(provider, task, output, "baseline")
    return RunResult(
        instance["instance_id"], instance["repo"],
        "baseline", quality, output, elapsed, ["raw"],
    )


def _build_organism(provider):
    nucleus = Nucleus(provider=provider)
    return skill_organism(
        stages=[
            SkillStage(
                name="localize",
                role="Bug Locator",
                instructions=(
                    "You are analyzing a bug report for a Python project. "
                    "Identify the exact file(s) and function(s) where the bug occurs. "
                    "Explain the root cause concisely."
                ),
                mode="fixed",
            ),
            SkillStage(
                name="edit",
                role="Patch Author",
                instructions=(
                    "Based on the bug localization from the previous stage, "
                    "write a minimal fix. Output a unified diff "
                    "(--- a/file\\n+++ b/file format). "
                    "Change only what's necessary to fix the reported issue."
                ),
                mode="fixed",
            ),
            SkillStage(
                name="verify",
                role="Patch Reviewer",
                instructions=(
                    "Review the proposed patch from the previous stage. "
                    "Check: (1) does it fix the reported issue? "
                    "(2) does it introduce regressions? "
                    "(3) is the diff format valid? "
                    "If the patch has problems, describe them."
                ),
                mode="fixed",
            ),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
        budget=ATP_Store(budget=2000, silent=True),
    )


def run_organism(provider, instance: dict) -> RunResult:
    task = _format_task(instance)
    org = _build_organism(provider)

    t0 = time.monotonic()
    result = org.run(task)
    elapsed = (time.monotonic() - t0) * 1000

    full_output = "\n\n".join(
        f"[{sr.stage_name}]\n{sr.output}"
        for sr in result.stage_results
    )
    quality = _judge(provider, task, full_output, "organism")
    stages = [sr.stage_name for sr in result.stage_results]
    return RunResult(
        instance["instance_id"], instance["repo"],
        "organism", quality, full_output, elapsed, stages,
    )


def run_langgraph(provider, instance: dict) -> RunResult:
    task = _format_task(instance)
    org = _build_organism(provider)

    t0 = time.monotonic()
    result = run_organism_langgraph(org, task=task, verify_certificates=True)
    elapsed = (time.monotonic() - t0) * 1000

    # Concatenate all stage outputs
    parts = []
    for stage_name, output in result.stage_outputs.items():
        parts.append(f"[{stage_name}]\n{output}")
    full_output = "\n\n".join(parts) if parts else result.output

    quality = _judge(provider, task, full_output, "langgraph")
    stages = result.metadata.get("stages_completed", [])
    return RunResult(
        instance["instance_id"], instance["repo"],
        "langgraph", quality, full_output, elapsed, stages,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SWE-bench-lite prompt-only evaluation")
    parser.add_argument("--model", default="gemma4:latest", help="Ollama model")
    parser.add_argument("--n", type=int, default=10, help="Number of instances")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N instances")
    parser.add_argument("--conditions", default="baseline,organism,langgraph",
                        help="Comma-separated conditions to run")
    args = parser.parse_args()

    provider = _make_provider(args.model)
    conditions = [c.strip() for c in args.conditions.split(",")]

    print(f"Model:      {args.model}")
    print(f"Instances:  {args.n} (offset {args.offset})")
    print(f"Conditions: {conditions}")

    # Verify model
    try:
        resp = _llm_call(provider, "Say ok.")
        print(f"Probe:      {resp.strip()[:30]}")
    except Exception as e:
        print(f"ERROR: Cannot reach model: {e}")
        sys.exit(1)

    # Load dataset
    print("\nLoading SWE-bench-lite...")
    from datasets import load_dataset
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    instances = list(ds.select(range(args.offset, min(args.offset + args.n, len(ds)))))
    print(f"Selected {len(instances)} instances")

    # Run conditions
    condition_fns = {
        "baseline": run_baseline,
        "organism": run_organism,
        "langgraph": run_langgraph,
    }

    results: list[RunResult] = []

    for i, instance in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(instances)}] {instance['instance_id']}")
        print(f"  repo: {instance['repo']}")
        print(f"  issue: {instance['problem_statement'][:80]}...")
        print(f"{'='*60}")

        for cond in conditions:
            fn = condition_fns[cond]
            r = fn(provider, instance)
            results.append(r)
            print(f"  {cond:12s} quality={r.quality:.2f}  latency={r.latency_ms:.0f}ms")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        if cond_results:
            mean_q = sum(r.quality for r in cond_results) / len(cond_results)
            mean_lat = sum(r.latency_ms for r in cond_results) / len(cond_results)
            print(f"  {cond:12s} mean_quality={mean_q:.3f}  mean_latency={mean_lat:.0f}ms  n={len(cond_results)}")

    # Comparison
    if "baseline" in conditions and "organism" in conditions:
        b_mean = sum(r.quality for r in results if r.condition == "baseline") / max(1, sum(1 for r in results if r.condition == "baseline"))
        o_mean = sum(r.quality for r in results if r.condition == "organism") / max(1, sum(1 for r in results if r.condition == "organism"))
        delta = o_mean - b_mean
        print(f"\n  organism vs baseline: {delta:+.3f}")
        if delta > 0.05:
            print("  → Organism pipeline IMPROVES over raw prompting")
        elif delta < -0.05:
            print("  → Organism pipeline DEGRADES vs raw prompting")
        else:
            print("  → No significant difference")

    if "organism" in conditions and "langgraph" in conditions:
        o_mean = sum(r.quality for r in results if r.condition == "organism") / max(1, sum(1 for r in results if r.condition == "organism"))
        l_mean = sum(r.quality for r in results if r.condition == "langgraph") / max(1, sum(1 for r in results if r.condition == "langgraph"))
        delta = l_mean - o_mean
        print(f"  langgraph vs organism: {delta:+.3f}")
        if abs(delta) <= 0.05:
            print("  → LangGraph produces same results (expected: same code path)")

    # Save
    out_path = Path("eval/results/swebench_lite.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "model": args.model,
        "dataset": "SWE-bench/SWE-bench_Lite",
        "n_instances": len(instances),
        "offset": args.offset,
        "conditions": conditions,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "judge": {"model": args.model, "temperature": 0.0, "rubric": "swebench_patch_quality"},
        "results": [
            {
                "instance_id": r.instance_id,
                "repo": r.repo,
                "condition": r.condition,
                "quality": r.quality,
                "latency_ms": r.latency_ms,
                "stages": r.stages,
            }
            for r in results
        ],
        "summary": {
            cond: {
                "mean_quality": round(sum(r.quality for r in results if r.condition == cond) / max(1, sum(1 for r in results if r.condition == cond)), 3),
                "n": sum(1 for r in results if r.condition == cond),
            }
            for cond in conditions
        },
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
