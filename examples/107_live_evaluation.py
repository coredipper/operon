#!/usr/bin/env python3
"""Example 107: Live evaluation harness with real LLM calls.

Runs 6 benchmark tasks (easy/medium/hard) through guided vs unguided
configurations with real OpenAI and Gemini providers. Measures actual
quality scores, token costs, and latencies.

Requires: OPENAI_API_KEY and GEMINI_API_KEY in environment (or .env file).

Usage:
    # Source env and run
    set -a && source .env && set +a && python examples/107_live_evaluation.py
"""

import os
import sys

# Load .env if present
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.convergence.live_evaluator import LiveEvaluator, _TASK_PROMPTS
from eval.convergence.tasks import get_benchmark_tasks as get_tasks


def main():
    tasks = get_tasks()
    # Filter to tasks with concrete prompts
    live_tasks = [t for t in tasks if t.task_id in _TASK_PROMPTS]
    print(f"Live tasks: {len(live_tasks)}")

    evaluator = LiveEvaluator()

    # Check providers
    providers = []
    if evaluator.openai.is_available():
        providers.append("openai")
        print("OpenAI: available (gpt-4o-mini)")
    if evaluator.gemini.is_available():
        providers.append("gemini")
        print("Gemini: available (gemini-2.0-flash)")

    if not providers:
        print("ERROR: No API keys found. Set OPENAI_API_KEY or GEMINI_API_KEY.")
        sys.exit(1)

    print()

    # Run each task: guided vs unguided, across available providers
    all_results = []
    for provider_name in providers:
        print(f"=== Provider: {provider_name} ===\n")

        for task in live_tasks:
            for guided in [False, True]:
                label = "guided" if guided else "unguided"
                print(f"  {task.task_id} ({task.difficulty}) [{label}]...", end=" ", flush=True)

                result = evaluator.evaluate_task(
                    task,
                    guided=guided,
                    provider_name=provider_name,
                )
                all_results.append(result)

                status = "PASS" if result.success else "FAIL"
                print(
                    f"{status}  quality={result.quality_score:.2f}  "
                    f"tokens={result.total_tokens:>5}  "
                    f"latency={result.total_latency_ms:>7.0f}ms  "
                    f"risk={result.risk_score:.3f}"
                )
                if result.quality_reason:
                    print(f"         reason: {result.quality_reason}")

            print()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for provider_name in providers:
        provider_results = [r for r in all_results if r.provider_name == provider_name]
        guided = [r for r in provider_results if r.config_name == "guided"]
        unguided = [r for r in provider_results if r.config_name == "unguided"]

        print(f"\n--- {provider_name} ---")
        for label, group in [("Unguided", unguided), ("Guided", guided)]:
            if not group:
                continue
            n = len(group)
            success_rate = sum(1 for r in group if r.success) / n * 100
            avg_quality = sum(r.quality_score for r in group) / n
            avg_tokens = sum(r.total_tokens for r in group) / n
            avg_latency = sum(r.total_latency_ms for r in group) / n
            avg_risk = sum(r.risk_score for r in group) / n
            print(
                f"  {label:>10}: "
                f"success={success_rate:5.1f}%  "
                f"quality={avg_quality:.3f}  "
                f"tokens={avg_tokens:7.0f}  "
                f"latency={avg_latency:7.0f}ms  "
                f"risk={avg_risk:.3f}"
            )

        # Delta
        if guided and unguided:
            g_q = sum(r.quality_score for r in guided) / len(guided)
            u_q = sum(r.quality_score for r in unguided) / len(unguided)
            g_r = sum(r.risk_score for r in guided) / len(guided)
            u_r = sum(r.risk_score for r in unguided) / len(unguided)
            print(f"     Delta: quality={g_q - u_q:+.3f}  risk={g_r - u_r:+.3f}")

    print("\n--- all done ---")


if __name__ == "__main__":
    main()
