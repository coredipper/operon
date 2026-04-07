"""Run empirical topology validation: compare epistemic predictions vs measured behavior."""
import argparse
import sys
import time

sys.path.insert(0, ".")

from eval.convergence.live_evaluator import LiveEvaluator
from eval.convergence.tasks import get_benchmark_tasks
from eval.convergence.topology_validation import run_validation
from eval.convergence.validation_report import generate_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Operon topology validation — epistemic predictions vs real LLM execution"
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds per task-config pair (default: 3)",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Specific task IDs to run (default: all)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only easy tasks (5 tasks, for smoke testing)",
    )
    parser.add_argument(
        "--out", default="eval/results/topology_validation",
        help="Output directory (default: eval/results/topology_validation)",
    )
    args = parser.parse_args()

    # Check Ollama — use the same chat-capable probe as LiveEvaluator
    evaluator = LiveEvaluator()
    if not evaluator.ollama:
        print("ERROR: No chat-capable Ollama model found at localhost:11434.")
        print("Start Ollama with: ollama serve")
        print("Pull a model with: ollama pull gemma4")
        return 1
    print(f"Model: {evaluator.ollama.model} via Ollama")

    # Select tasks
    all_tasks = get_benchmark_tasks()
    if args.tasks:
        task_ids = set(args.tasks)
        tasks = [t for t in all_tasks if t.task_id in task_ids]
        if not tasks:
            print(f"ERROR: No matching tasks for {args.tasks}")
            return 1
    elif args.quick:
        tasks = [t for t in all_tasks if t.task_id.startswith("easy_")]
    else:
        tasks = all_tasks

    n_runs = len(tasks) * 2 * args.seeds
    print(f"Tasks: {len(tasks)} x 2 configs x {args.seeds} seeds = {n_runs} runs")
    if args.resume:
        print("Resuming from checkpoint")
    print()

    start = time.time()

    def progress(done: int, total: int, task_id: str, config: str) -> None:
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"  [{done}/{total}] {task_id} ({config}) "
            f"— {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
        )

    report = run_validation(
        n_seeds=args.seeds,
        tasks=tasks,
        resume=args.resume,
        out_dir=args.out,
        progress_callback=progress,
    )

    elapsed = time.time() - start
    print()
    print(f"Completed {report.metadata['total_runs']} runs in {elapsed:.0f}s")
    print()

    # Generate reports
    generate_report(report, args.out)
    print(f"Results written to {args.out}/")
    print()

    # Print summary
    s = report.summary
    print("=== Summary ===")
    print(f"Success rate: {s['success_rate']:.1%}")
    print(f"Mean quality: {s['mean_quality']:.3f}")
    print()

    print("=== Theorem Correlations ===")
    print(f"{'Theorem':<22s} {'rho':>6s} {'p':>8s} {'Dir':>5s} {'Pass':>5s}")
    print("-" * 50)
    for t in report.theorems:
        d = "ok" if t.direction_correct else "WRONG"
        if t.informational:
            p = "info"
        else:
            p = "YES" if t.validation_pass else "no"
        print(f"{t.theorem_name:<22s} {t.spearman_rho:>+6.3f} {t.spearman_p:>8.4f} {d:>5s} {p:>5s}")
    print()

    print(f"Validated: {s['theorems_validated']}/{s['theorems_total']} theorems")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
