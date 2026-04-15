"""Composition non-interference experiment.

Tests Ma et al.'s claim (arXiv:2604.05013) that atomic coding skills
compose without negative interference. Runs three conditions:

  1. INDIVIDUAL: each skill (localize, edit, test) runs solo on a task
  2. COMPOSED: localize → edit → test runs as a serial pipeline
  3. BASELINE: raw LLM call with no organism wrapper

If composition degrades quality (composed < mean(individual)), there
is negative interference. If composed >= mean(individual), Ma et al.'s
claim holds for this model and task.

Usage:
  python eval/composition_interference.py [--model gemma4:latest] [--reps 3]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operon_ai import Nucleus, SkillStage, skill_organism
from operon_ai.providers.openai_compatible_provider import OpenAICompatibleProvider
from operon_ai.providers.base import ProviderConfig


# ---------------------------------------------------------------------------
# Task definitions — coding tasks where localize/edit/test make sense
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "sql_injection",
        "prompt": (
            "The following Python function has a security vulnerability. "
            "Find the bug, fix it, and write a test.\n\n"
            "```python\n"
            "def get_user(db, username):\n"
            '    query = f"SELECT * FROM users WHERE name = \'{username}\'"\n'
            "    return db.execute(query).fetchone()\n"
            "```"
        ),
        "expected_keywords": ["parameterized", "placeholder", "?", "%s"],
    },
    {
        "id": "off_by_one",
        "prompt": (
            "The following function has an off-by-one error. "
            "Find it, fix it, and write a test.\n\n"
            "```python\n"
            "def paginate(items, page, per_page=10):\n"
            "    start = page * per_page\n"
            "    end = start + per_page\n"
            "    return items[start:end]\n"
            "```\n\n"
            "Note: pages should be 1-indexed (page 1 = first page)."
        ),
        "expected_keywords": ["page - 1", "(page - 1)", "1-indexed"],
    },
    {
        "id": "race_condition",
        "prompt": (
            "The following code has a TOCTOU race condition. "
            "Find the bug, fix it, and write a test.\n\n"
            "```python\n"
            "import os\n\n"
            "def safe_write(path, content):\n"
            "    if not os.path.exists(path):\n"
            "        with open(path, 'w') as f:\n"
            "            f.write(content)\n"
            "    else:\n"
            "        raise FileExistsError(path)\n"
            "```"
        ),
        "expected_keywords": ["O_EXCL", "O_CREAT", "atomic", "os.open", "lock"],
    },
]


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------

def _make_provider(model: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="not-needed",
        base_url="http://localhost:11434/v1",
        model=model,
    )


def _llm_call(provider, prompt: str, max_tokens: int = 2048) -> str:
    """Direct LLM call, no organism wrapper."""
    config = ProviderConfig(max_tokens=max_tokens)
    resp = provider.complete(prompt, config)
    return resp.content


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_JUDGE_RUBRIC = (
    "You are an evaluation judge. Score the OUTPUT for the given TASK "
    "on a 0.0-1.0 scale using these criteria:\n"
    "- Correctness (~50%): identifies the bug correctly, fix is correct\n"
    "- Completeness (~30%): covers find + fix + test as applicable\n"
    "- Clarity (~20%): well-organized, easy to understand\n\n"
    "Score anchors:\n"
    "  0.0-0.2: Wrong, off-topic, or refuses the task\n"
    "  0.3-0.5: Partially correct but major gaps or errors\n"
    "  0.6-0.7: Mostly correct, minor issues\n"
    "  0.8-0.9: Strong, nearly complete and accurate\n"
    "  1.0: Perfect — correct, complete, and clear\n\n"
    "Return ONLY JSON: {\"score\": <float>}\n\n"
)


def _extract_score(content: str) -> float:
    """Extract a 0.0-1.0 score from judge response text."""
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


def _judge(provider, task_prompt: str, output: str) -> float:
    """Score output 0.0-1.0 using the SAME rubric for all conditions.

    Uses temperature=0.0 for reproducibility and a 6000-char output
    limit to avoid truncating composed pipeline outputs.
    """
    judge_prompt = (
        f"{_JUDGE_RUBRIC}"
        f"TASK: {task_prompt[:800]}\n"
        f"OUTPUT: {output[:6000]}\n"
    )
    config = ProviderConfig(max_tokens=2048, temperature=0.0)
    resp = provider.complete(judge_prompt, config)
    return _extract_score(resp.content)


# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    condition: str
    task_id: str
    rep: int
    quality: float
    output: str
    stages: list[str]
    latency_ms: float


def run_baseline(provider, task: dict, rep: int) -> RunResult:
    """Raw LLM call — no organism."""
    prompt = task["prompt"] + "\n\nFind the bug, fix it, and write a test."
    t0 = time.monotonic()
    output = _llm_call(provider, prompt)
    elapsed = (time.monotonic() - t0) * 1000
    quality = _judge(provider, task["prompt"], output)
    return RunResult("baseline", task["id"], rep, quality, output, ["raw"], elapsed)


def run_individual(provider, task: dict, rep: int, skill: str) -> RunResult:
    """Single skill in an organism, judged with skill-specific rubric."""
    skill_instructions = {
        "localize": "Find the exact location and nature of the bug in the code.",
        "edit": "Fix the bug with a correct, minimal code change. Show the fixed code.",
        "test": "Write a unit test that verifies the bug is fixed.",
    }
    nucleus = Nucleus(provider=provider)
    org = skill_organism(
        stages=[
            SkillStage(
                name=skill,
                role=skill.title(),
                instructions=skill_instructions[skill],
                mode="fixed",
            ),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
    )
    t0 = time.monotonic()
    result = org.run(task["prompt"])
    elapsed = (time.monotonic() - t0) * 1000
    # Use the SAME end-to-end rubric as baseline and composed
    # so the interference comparison is apples-to-apples.
    quality = _judge(provider, task["prompt"], result.final_output)
    return RunResult(f"individual_{skill}", task["id"], rep, quality,
                     result.final_output, [skill], elapsed)


def run_composed(provider, task: dict, rep: int) -> RunResult:
    """localize → edit → test as a serial pipeline.

    Judges the full pipeline output (all stage outputs concatenated),
    not just the last stage, so the judge sees the complete work.
    """
    nucleus = Nucleus(provider=provider)
    org = skill_organism(
        stages=[
            SkillStage(
                name="localize",
                role="Locator",
                instructions="Find the exact location and nature of the bug in the code.",
                mode="fixed",
            ),
            SkillStage(
                name="edit",
                role="Editor",
                instructions="Fix the bug with a correct, minimal code change. Show the fixed code.",
                mode="fixed",
            ),
            SkillStage(
                name="test",
                role="Tester",
                instructions="Write a unit test that verifies the bug is fixed.",
                mode="fixed",
            ),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
    )
    t0 = time.monotonic()
    result = org.run(task["prompt"])
    elapsed = (time.monotonic() - t0) * 1000

    # Concatenate all stage outputs so the judge sees the full pipeline work
    full_output = "\n\n".join(
        f"[{sr.stage_name}]\n{sr.output}"
        for sr in result.stage_results
    )
    quality = _judge(provider, task["prompt"], full_output)
    stages = [sr.stage_name for sr in result.stage_results]
    return RunResult("composed", task["id"], rep, quality,
                     full_output, stages, elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Composition non-interference experiment")
    parser.add_argument("--model", default="gemma4:latest", help="Ollama model name")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per condition")
    parser.add_argument("--tasks", default="all", help="Task IDs (comma-sep) or 'all'")
    args = parser.parse_args()

    provider = _make_provider(args.model)
    print(f"Model: {args.model}")
    print(f"Reps:  {args.reps}")

    # Verify model is reachable
    try:
        test = _llm_call(provider, "Say 'ok'.", max_tokens=10)
        print(f"Probe: {test.strip()[:30]}")
    except Exception as e:
        print(f"ERROR: Cannot reach model: {e}")
        sys.exit(1)

    tasks = TASKS
    if args.tasks != "all":
        ids = set(args.tasks.split(","))
        tasks = [t for t in TASKS if t["id"] in ids]

    results: list[RunResult] = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task['id']}")
        print(f"{'='*60}")

        for rep in range(1, args.reps + 1):
            print(f"\n--- Rep {rep}/{args.reps} ---")

            # Baseline
            r = run_baseline(provider, task, rep)
            results.append(r)
            print(f"  baseline:    {r.quality:.2f}  ({r.latency_ms:.0f}ms)")

            # Individual skills
            for skill in ["localize", "edit", "test"]:
                r = run_individual(provider, task, rep, skill)
                results.append(r)
                print(f"  {skill:12s} {r.quality:.2f}  ({r.latency_ms:.0f}ms)")

            # Composed
            r = run_composed(provider, task, rep)
            results.append(r)
            print(f"  composed:    {r.quality:.2f}  ({r.latency_ms:.0f}ms)")

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    for task in tasks:
        task_results = [r for r in results if r.task_id == task["id"]]

        baseline_q = [r.quality for r in task_results if r.condition == "baseline"]
        individual_q = [r.quality for r in task_results
                        if r.condition.startswith("individual_")]
        composed_q = [r.quality for r in task_results if r.condition == "composed"]

        mean_b = sum(baseline_q) / len(baseline_q) if baseline_q else 0
        mean_i = sum(individual_q) / len(individual_q) if individual_q else 0
        mean_c = sum(composed_q) / len(composed_q) if composed_q else 0

        # Per-skill breakdown
        skill_means = {}
        for skill in ["localize", "edit", "test"]:
            sq = [r.quality for r in task_results if r.condition == f"individual_{skill}"]
            skill_means[skill] = sum(sq) / len(sq) if sq else 0

        delta = mean_c - mean_i
        interference = "NEGATIVE" if delta < -0.1 else "NONE" if abs(delta) <= 0.1 else "POSITIVE"

        print(f"\n  Task: {task['id']}")
        print(f"    baseline:     {mean_b:.3f}")
        print(f"    localize:     {skill_means['localize']:.3f}")
        print(f"    edit:         {skill_means['edit']:.3f}")
        print(f"    test:         {skill_means['test']:.3f}")
        print(f"    mean(indiv):  {mean_i:.3f}")
        print(f"    composed:     {mean_c:.3f}")
        print(f"    delta:        {delta:+.3f}  → {interference} interference")

    # Overall
    all_individual = [r.quality for r in results if r.condition.startswith("individual_")]
    all_composed = [r.quality for r in results if r.condition == "composed"]
    all_baseline = [r.quality for r in results if r.condition == "baseline"]

    mean_all_i = sum(all_individual) / len(all_individual) if all_individual else 0
    mean_all_c = sum(all_composed) / len(all_composed) if all_composed else 0
    mean_all_b = sum(all_baseline) / len(all_baseline) if all_baseline else 0
    delta_all = mean_all_c - mean_all_i

    print(f"\n{'='*60}")
    print("OVERALL")
    print(f"{'='*60}")
    print(f"  baseline:      {mean_all_b:.3f}")
    print(f"  mean(indiv):   {mean_all_i:.3f}")
    print(f"  composed:      {mean_all_c:.3f}")
    print(f"  delta:         {delta_all:+.3f}")
    interference_all = "NEGATIVE" if delta_all < -0.1 else "NONE" if abs(delta_all) <= 0.1 else "POSITIVE"
    print(f"  interference:  {interference_all}")
    print()

    if interference_all == "NEGATIVE":
        print("  Ma et al. claim DOES NOT HOLD for this model/task set.")
        print("  Composition degrades quality.")
    else:
        print("  Ma et al. claim HOLDS for this model/task set.")
        print("  Composition does not degrade quality.")

    # Save results
    out_path = Path("eval/results/composition_interference.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "model": args.model,
        "reps": args.reps,
        "results": [
            {
                "condition": r.condition,
                "task_id": r.task_id,
                "rep": r.rep,
                "quality": r.quality,
                "latency_ms": r.latency_ms,
                "stages": r.stages,
            }
            for r in results
        ],
        "summary": {
            "baseline": mean_all_b,
            "mean_individual": mean_all_i,
            "composed": mean_all_c,
            "delta": delta_all,
            "interference": interference_all,
        },
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
