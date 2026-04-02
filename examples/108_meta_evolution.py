#!/usr/bin/env python3
"""Example 108: Meta-evolution of organism configurations (C8).

Evolves SkillOrganism configurations (modes, models, thresholds) against
benchmark tasks using the FilesystemOptimizer.  Uses EpiplexityMonitor
for stall detection and hybrid proposers (tournament mutation + optional
LLM exploration).

Requires: GEMINI_API_KEY in environment (or .env file).
Optional: Local LLM server (vllama, Ollama) for the LLM proposer.

Usage:
    # Basic run (tournament mutator only)
    set -a && source .env && set +a && python examples/108_meta_evolution.py

    # With LLM proposer via vllama
    set -a && source .env && set +a && python examples/108_meta_evolution.py --llm http://localhost:8080/v1
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

from operon_ai.convergence import (
    CandidateConfig,
    EvolutionConfig,
    EvolutionLoop,
    MetaStageConfig as StageConfig,
)
from eval.convergence.live_evaluator import LiveEvaluator
from eval.convergence.tasks import get_benchmark_tasks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="C8 Meta-Evolution Example")
    parser.add_argument("--llm", type=str, default=None,
                        help="LLM proposer URL (e.g. http://localhost:8080/v1)")
    args = parser.parse_args()

    # Pick 3 easy tasks for a quick demo
    tasks = [t for t in get_benchmark_tasks() if t.task_id.startswith("easy_seq")][:3]
    print(f"Tasks: {[t.task_id for t in tasks]}")

    # Create 3 seed candidates with different strategies
    roles = tasks[0].required_roles
    seeds = [
        CandidateConfig(
            candidate_id="seed_all_fuzzy",
            parent_id=None, iteration=0,
            stage_configs=tuple(StageConfig(role=r, mode="fuzzy") for r in roles),
            intervention_policy={}, proposer="seed",
            reason="all-fuzzy baseline",
        ),
        CandidateConfig(
            candidate_id="seed_all_fixed",
            parent_id=None, iteration=0,
            stage_configs=tuple(StageConfig(role=r, mode="fixed") for r in roles),
            intervention_policy={}, proposer="seed",
            reason="all-fixed baseline",
        ),
        CandidateConfig(
            candidate_id="seed_mixed",
            parent_id=None, iteration=0,
            stage_configs=tuple(
                StageConfig(role=r, mode="fixed" if i == 0 else "fuzzy")
                for i, r in enumerate(roles)
            ),
            intervention_policy={}, proposer="seed",
            reason="fixed-reader fuzzy-writer",
        ),
    ]

    # Optional LLM proposer
    llm_provider = None
    if args.llm:
        from operon_ai.providers.openai_compatible_provider import OpenAICompatibleProvider
        from eval.convergence.live_evaluator import _list_models, _first_chat_model
        candidates = _list_models(args.llm)
        model = _first_chat_model(args.llm, candidates)
        if model:
            llm_provider = OpenAICompatibleProvider(
                api_key="not-needed", base_url=args.llm, model=model,
            )
            print(f"LLM proposer: {model}")
        else:
            print(f"WARNING: No chat-capable model at {args.llm}, running tournament-only")

    # Configure and run
    config = EvolutionConfig(
        max_iterations=3,
        population_size=6,
        stall_threshold=2,
        seed=42,
    )
    evaluator = LiveEvaluator()
    loop = EvolutionLoop(
        config=config,
        evaluator=evaluator,
        tasks=tasks,
        provider_name="gemini",
        llm_proposer_provider=llm_provider,
    )
    loop.seed(seeds)

    print(f"\nRunning evolution: {config.max_iterations} iterations x {len(tasks)} tasks")
    result = loop.run()

    # Results
    print(f"\n{'=' * 50}")
    print(f"Best score: {result['best_score']:.2f}")
    print(f"Assessments: {result['total_assessments']}")
    print(f"Store: {result['store_path']}")

    best = result["best"]
    if best:
        print(f"\nBest config ({best.candidate_id}, proposer={best.proposer}):")
        for i, sc in enumerate(best.stage_configs):
            print(f"  stage {i}: role={sc.role} mode={sc.mode} model={sc.model}")

    stats = result.get("monitor_stats", {})
    print(f"\nEpiplexity: mean={stats.get('mean_epiplexity', 0):.3f}, "
          f"stagnant_episodes={stats.get('stagnant_episodes', 0)}")


if __name__ == "__main__":
    main()
