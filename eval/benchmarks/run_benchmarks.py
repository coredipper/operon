"""CLI runner for the biological benchmark suite.

Usage:
    python -m eval.benchmarks.run_benchmarks --benchmark all --seed 42
    python -m eval.benchmarks.run_benchmarks --benchmark epiplexity --seed 42 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Operon biological benchmark suite")
    parser.add_argument(
        "--benchmark",
        choices=["all", "epiplexity", "quorum", "metabolism"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", help="Path to write results JSON")
    parser.add_argument("--trials", type=int, help="Override number of trials per scenario")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_results: list[dict] = []

    benchmarks_to_run = (
        ["epiplexity", "quorum", "metabolism"] if args.benchmark == "all"
        else [args.benchmark]
    )

    for bench in benchmarks_to_run:
        print(f"\n{'='*60}")
        print(f"  Running: {bench}")
        print(f"{'='*60}")
        t0 = time.time()

        if bench == "epiplexity":
            from eval.benchmarks.epiplexity_bench import EpiplexityBenchConfig, run_epiplexity_bench
            config = EpiplexityBenchConfig()
            if args.trials:
                config.n_trials = args.trials
            results = run_epiplexity_bench(config, rng)

        elif bench == "quorum":
            from eval.benchmarks.quorum_bench import QuorumBenchConfig, run_quorum_bench
            config = QuorumBenchConfig()
            if args.trials:
                config.n_trials = args.trials
            results = run_quorum_bench(config, rng)

        elif bench == "metabolism":
            from eval.benchmarks.metabolism_bench import MetabolismBenchConfig, run_metabolism_bench
            config = MetabolismBenchConfig()
            if args.trials:
                config.n_trials = args.trials
            results = run_metabolism_bench(config, rng)

        else:
            continue

        elapsed = time.time() - t0
        print(f"  {bench}: {len(results)} scenarios, {elapsed:.1f}s")

        for r in results:
            all_results.append(r.as_dict())

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Summary: {len(all_results)} total results")
    print(f"{'='*60}\n")

    for r in all_results:
        print(f"  [{r['name']}] {r['scenario']}")
        for variant in ["biological", "ablated", "naive"]:
            if variant in r:
                metrics = r[variant]
                parts = []
                for metric, data in metrics.items():
                    ci = data.get("wilson_95", [0, 0])
                    parts.append(f"{metric}={data['rate']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
                print(f"    {variant:12s}: {', '.join(parts)}")

    # Write results
    if args.out:
        out_path = args.out
    else:
        os.makedirs("eval/results/benchmarks", exist_ok=True)
        out_path = f"eval/results/benchmarks/seed_{args.seed}.json"

    with open(out_path, "w") as f:
        json.dump({"seed": args.seed, "results": all_results}, f, indent=2)
    print(f"\n  Results written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
