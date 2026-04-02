#!/usr/bin/env python3
"""C8 Meta-Evolution CLI runner.

Evolves organism configurations against benchmark tasks using the
FilesystemOptimizer. Analogous to run_live_eval.py but iterative.

Usage:
    python run_meta_evolution.py --max-iterations 5 --provider gemini
    python run_meta_evolution.py --tasks easy_seq_01,easy_seq_02 --seed 42
"""

from __future__ import annotations

import argparse
import sys

from operon_ai.convergence.evolution_loop import EvolutionConfig, EvolutionLoop
from operon_ai.convergence.meta_types import CandidateConfig, StageConfig
from eval.convergence.live_evaluator import LiveEvaluator
from eval.convergence.tasks import get_benchmark_tasks


def _make_seeds(task_roles: tuple[str, ...]) -> list[CandidateConfig]:
    """Create 3 seed candidates: conservative, balanced, aggressive."""
    seeds = []

    # Conservative: all fixed mode
    seeds.append(CandidateConfig(
        candidate_id="seed_conservative",
        parent_id=None,
        iteration=0,
        stage_configs=tuple(
            StageConfig(role=r, mode="fixed") for r in task_roles
        ),
        intervention_policy={"max_rate": 0.3},
        proposer="seed",
        reason="all-fixed conservative baseline",
    ))

    # Balanced: first stages fixed, last stage fuzzy
    stages = []
    for i, r in enumerate(task_roles):
        mode = "fuzzy" if i == len(task_roles) - 1 else "fixed"
        stages.append(StageConfig(role=r, mode=mode))
    seeds.append(CandidateConfig(
        candidate_id="seed_balanced",
        parent_id=None,
        iteration=0,
        stage_configs=tuple(stages),
        intervention_policy={"max_rate": 0.5},
        proposer="seed",
        reason="fixed-early fuzzy-late balanced",
    ))

    # Aggressive: all fuzzy
    seeds.append(CandidateConfig(
        candidate_id="seed_aggressive",
        parent_id=None,
        iteration=0,
        stage_configs=tuple(
            StageConfig(role=r, mode="fuzzy") for r in task_roles
        ),
        intervention_policy={"max_rate": 0.7},
        proposer="seed",
        reason="all-fuzzy aggressive exploration",
    ))

    return seeds


def main():
    parser = argparse.ArgumentParser(description="C8 Meta-Evolution Runner")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--provider", default="gemini",
                        choices=["gemini", "openai", "anthropic", "ollama"])
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task IDs (default: all)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--llm-proposer", type=str, default=None,
                        help="Local LLM proposer URL (e.g. http://localhost:8080/v1)")
    parser.add_argument("--llm-proposer-model", type=str, default=None,
                        help="Model name for LLM proposer")
    args = parser.parse_args()

    # Load tasks
    all_tasks = get_benchmark_tasks()
    if args.tasks:
        task_ids = set(args.tasks.split(","))
        tasks = [t for t in all_tasks if t.task_id in task_ids]
        if not tasks:
            print(f"No tasks matched: {args.tasks}")
            sys.exit(1)
    else:
        tasks = all_tasks

    print(f"C8 Meta-Evolution: {len(tasks)} tasks, {args.max_iterations} iterations")
    print(f"Provider: {args.provider}")

    # Create evaluator
    evaluator = LiveEvaluator()

    # Determine roles from first task
    roles = tasks[0].required_roles
    seeds = _make_seeds(roles)
    print(f"Seeds: {len(seeds)} ({', '.join(s.candidate_id for s in seeds)})")

    # Configure and run
    config = EvolutionConfig(
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        seed=args.seed,
    )

    # Optional LLM proposer for exploration when stalled
    llm_provider = None
    if args.llm_proposer:
        if args.llm_proposer in ("gemini", "openai", "anthropic"):
            # Use an API provider as LLM proposer
            custom = args.llm_proposer_model
            if args.llm_proposer == "gemini":
                from operon_ai.providers.gemini_provider import GeminiProvider
                llm_provider = GeminiProvider(model=custom or "gemini-2.5-flash")
            elif args.llm_proposer == "openai":
                from operon_ai.providers.openai_provider import OpenAIProvider
                llm_provider = OpenAIProvider(model=custom or "gpt-4o-mini")
            elif args.llm_proposer == "anthropic":
                from operon_ai.providers.anthropic_provider import AnthropicProvider
                llm_provider = AnthropicProvider(model=custom or "claude-haiku-4-5-20251001")
            print(f"LLM proposer: {args.llm_proposer} model={getattr(llm_provider, 'model', '?')}")
        else:
            # Local OpenAI-compatible server URL
            from operon_ai.providers.openai_compatible_provider import OpenAICompatibleProvider
            from eval.convergence.live_evaluator import _list_models, _first_chat_model
            model = args.llm_proposer_model
            if not model:
                candidates = _list_models(args.llm_proposer)
                model = _first_chat_model(args.llm_proposer, candidates)
                if not model:
                    print(f"WARNING: No chat-capable model at {args.llm_proposer}, disabling LLM proposer")
            if model:
                llm_provider = OpenAICompatibleProvider(
                    api_key="not-needed",
                    base_url=args.llm_proposer,
                    model=model,
                )
                print(f"LLM proposer: {args.llm_proposer} model={model}")

    loop = EvolutionLoop(
        config=config,
        evaluator=evaluator,
        tasks=tasks,
        provider_name=args.provider,
        llm_proposer_provider=llm_provider,
    )
    loop.seed(seeds)

    print("\nStarting evolution loop...")
    result = loop.run()

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Evolution complete: {result['total_assessments']} assessments")
    print(f"Best score: {result['best_score']:.3f}")
    print(f"Store: {result['store_path']}")

    best = result["best"]
    if best:
        print(f"\nBest config ({best.candidate_id}):")
        print(f"  Proposer: {best.proposer}")
        print(f"  Reason: {best.reason}")
        for i, sc in enumerate(best.stage_configs):
            print(f"  Stage {i}: role={sc.role} mode={sc.mode} model={sc.model}")

    stats = result.get("monitor_stats", {})
    if stats:
        print(f"\nEpiplexity stats:")
        print(f"  Stagnant episodes: {stats.get('stagnant_episodes', 0)}")
        print(f"  Mean epiplexity: {stats.get('mean_epiplexity', 0):.3f}")


if __name__ == "__main__":
    main()
