"""Run live evaluation: Gemini API + Claude CLI + Codex CLI."""
import argparse, os, sys, time

# Load .env
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

sys.path.insert(0, ".")

from eval.convergence.live_evaluator import LiveEvaluator, _TASK_PROMPTS
from eval.convergence.tasks import get_benchmark_tasks

_parser = argparse.ArgumentParser(description="Operon live evaluation")
_parser.add_argument(
    "--judge",
    choices=["lmstudio", "ollama", "gemini", "anthropic", "openai"],
    default=None,
    help="Force cross-judging with this provider (default: self-judge)",
)
_parser.add_argument(
    "--provider",
    action="append",
    help="Only run these providers (can repeat, e.g. --provider gemini --provider ollama)",
)
_args = _parser.parse_args()

tasks = get_benchmark_tasks()
live_tasks = [t for t in tasks if t.task_id in _TASK_PROMPTS]
print(f"Tasks: {len(live_tasks)}")
if _args.judge:
    print(f"Judge: {_args.judge} (cross-judging)")

evaluator = LiveEvaluator()

# Validate --judge early — fail before expensive execution
if _args.judge:
    try:
        evaluator._pick_judge("gemini", judge_provider=_args.judge)
    except ValueError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)

providers = []
# API providers
if evaluator.gemini.is_available():
    providers.append("gemini")
    print(f"  Gemini: available ({evaluator.gemini.model})")
if evaluator.anthropic.is_available():
    providers.append("anthropic")
    print(f"  Anthropic: available ({evaluator.anthropic.model})")
if evaluator.openai.is_available():
    providers.append("openai")
    print(f"  OpenAI: available ({evaluator.openai.model})")
# Local providers
if evaluator.ollama:
    providers.append("ollama")
    print(f"  Ollama: available ({evaluator.ollama.model})")
if evaluator.lmstudio:
    providers.append("lmstudio")
    print(f"  LM Studio: available ({evaluator.lmstudio.model})")
# CLI providers
import shutil
if shutil.which("claude"):
    providers.append("claude")
    print("  Claude CLI: available")
if shutil.which("codex"):
    providers.append("codex")
    print("  Codex CLI: available")

if _args.provider:
    unknown = set(_args.provider) - set(providers)
    if unknown:
        print(f"ERROR: requested providers not available: {unknown}")
        print(f"Available: {providers}")
        raise SystemExit(1)
    providers = [p for p in providers if p in _args.provider]

if not providers:
    print("ERROR: no providers available")
    raise SystemExit(1)

print(f"\nProviders: {providers} ({len(providers)} total)\n")

all_results = []

for provider_name in providers:
    print(f"{'='*60}")
    print(f"Provider: {provider_name}")
    print(f"{'='*60}\n")

    for task in live_tasks:
        for guided in [False, True]:
            label = "guided" if guided else "unguided"
            sys.stdout.write(f"  {task.task_id} ({task.difficulty}) [{label}]... ")
            sys.stdout.flush()

            r = evaluator.evaluate_task(task, guided=guided, provider_name=provider_name, judge_provider=_args.judge)
            all_results.append(r)

            s = "PASS" if r.success else "FAIL"
            print(f"{s}  q={r.quality_score:.2f}  tok={r.total_tokens:>6}  lat={r.total_latency_ms:>8.0f}ms  risk={r.risk_score:.3f}")

            # Rate limit buffer per provider
            delays = {"gemini": 5, "anthropic": 2, "openai": 2, "ollama": 0, "lmstudio": 0}
            time.sleep(delays.get(provider_name, 1))
        print()

# ==========================================================================
# SUMMARY
# ==========================================================================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Provider':<10} {'Config':<10} {'N':>3} {'Success':>8} {'Quality':>8} {'Tokens':>8} {'Latency':>10} {'Risk':>6}")
print("-" * 68)

def _is_guided(config_name: str) -> bool:
    return config_name.endswith("guided") and not config_name.endswith("unguided")

for p in providers:
    for cfg in ["unguided", "guided"]:
        if cfg == "guided":
            group = [r for r in all_results if r.provider_name == p and _is_guided(r.config_name)]
        else:
            group = [r for r in all_results if r.provider_name == p and r.config_name.endswith("unguided")]
        if not group:
            continue
        n = len(group)
        sr = sum(1 for r in group if r.success) / n * 100
        q = sum(r.quality_score for r in group) / n
        t = sum(r.total_tokens for r in group) / n
        l = sum(r.total_latency_ms for r in group) / n
        ri = sum(r.risk_score for r in group) / n
        print(f"{p:<10} {cfg:<10} {n:>3} {sr:>7.0f}% {q:>8.3f} {t:>8.0f} {l:>9.0f}ms {ri:>6.3f}")

print(f"\n{'='*60}")
print("GUIDANCE EFFECT (guided - unguided)")
print(f"{'='*60}\n")
for p in providers:
    g = [r for r in all_results if r.provider_name == p and r.config_name.endswith("guided") and not r.config_name.endswith("unguided")]
    u = [r for r in all_results if r.provider_name == p and r.config_name.endswith("unguided")]
    if g and u:
        gq = sum(r.quality_score for r in g) / len(g)
        uq = sum(r.quality_score for r in u) / len(u)
        gt = sum(r.total_tokens for r in g) / len(g)
        ut = sum(r.total_tokens for r in u) / len(u)
        print(f"  {p:<10}: quality {gq-uq:+.3f}  tokens {gt-ut:+.0f}")

print(f"\n{'='*60}")
print("CROSS-PROVIDER (unguided)")
print(f"{'='*60}")
print(f"\n{'Task':<20}", end="")
for p in providers:
    print(f"{p:>12}", end="")
print()
print("-" * (20 + 12 * len(providers)))
for task in live_tasks:
    print(f"{task.task_id:<20}", end="")
    for p in providers:
        r = next((r for r in all_results if r.task_id == task.task_id and r.provider_name == p and r.config_name.endswith("unguided")), None)
        if r:
            print(f"{r.quality_score:>12.2f}", end="")
        else:
            print(f"{'—':>12}", end="")
    print()

print("\n--- done ---")
