"""
Example 104 -- C6 Convergence Evaluation Harness
==================================================

Runs the convergence evaluation harness with 3 tasks x 7 configurations,
prints the ranked comparison table, and asserts that guided configurations
produce lower risk scores than their unguided counterparts.

Usage:
    python examples/104_evaluation_harness.py
"""

from eval.convergence.configurations import get_configurations
from eval.convergence.harness import ConvergenceHarness, HarnessConfig
from eval.convergence.metrics import collect_metrics, compare_configs
from eval.convergence.report import ranking_table
from eval.convergence.metrics import AggregateMetrics

# ---------------------------------------------------------------------------
# 1. Run harness: 3 tasks x 7 configs
# ---------------------------------------------------------------------------

config = HarnessConfig(
    seed=1337,
    tasks=["easy_seq_01", "med_mix_03", "hard_par_01"],
)

harness = ConvergenceHarness(config)

print(f"Tasks: {len(harness.tasks)}")
print(f"Configs: {len(harness.configurations)}")
print()

results = harness.run()

# ---------------------------------------------------------------------------
# 2. Print comparison table
# ---------------------------------------------------------------------------

print("=== Configuration Ranking ===")
print()
comparison = results["comparison"]
print(f"{'Rank':<5} {'Config':<25} {'Success':>8} {'Risk':>8} {'Tokens':>8} {'Latency':>10} {'Warns':>6}")
print("-" * 75)
for rank, row in enumerate(comparison, 1):
    print(
        f"{rank:<5} {row['config_id']:<25} "
        f"{row['success_rate']:>7.0%} "
        f"{row['mean_risk_score']:>8.4f} "
        f"{row['mean_token_cost']:>8.0f} "
        f"{row['mean_latency_ms']:>10.0f} "
        f"{row['mean_interventions']:>6.1f}"
    )
print()

# ---------------------------------------------------------------------------
# 3. Print structural variation
# ---------------------------------------------------------------------------

print("=== Structural Variation ===")
for config_id, val in sorted(results["variation"].items()):
    print(f"  {config_id:<25} {val:.4f}")
print()

# ---------------------------------------------------------------------------
# 4. Print top credit roles
# ---------------------------------------------------------------------------

print("=== Top Credit Roles ===")
credit = results["credit"]
sorted_credit = sorted(credit.items(), key=lambda kv: -abs(kv[1]))[:8]
for role, val in sorted_credit:
    print(f"  {role:<25} {val:+.4f}")
print()

# ---------------------------------------------------------------------------
# --test: Assert guided configs have lower risk than unguided
# ---------------------------------------------------------------------------

aggregates = results["aggregates"]

# Swarms guided vs baseline.
swarms_baseline_risk = aggregates["swarms_baseline"]["mean_risk_score"]
swarms_guided_risk = aggregates["swarms_operon"]["mean_risk_score"]
assert swarms_guided_risk <= swarms_baseline_risk, (
    f"Swarms guided risk ({swarms_guided_risk:.4f}) should be <= "
    f"baseline ({swarms_baseline_risk:.4f})"
)

# Scion guided vs baseline.
scion_baseline_risk = aggregates["scion_baseline"]["mean_risk_score"]
scion_guided_risk = aggregates["scion_operon"]["mean_risk_score"]
assert scion_guided_risk <= scion_baseline_risk, (
    f"Scion guided risk ({scion_guided_risk:.4f}) should be <= "
    f"baseline ({scion_baseline_risk:.4f})"
)

# Operon adaptive should also have reasonable risk.
operon_risk = aggregates["operon_adaptive"]["mean_risk_score"]
assert 0.0 <= operon_risk <= 1.0, f"Operon risk out of bounds: {operon_risk}"

print(f"Swarms: baseline={swarms_baseline_risk:.4f}, guided={swarms_guided_risk:.4f}")
print(f"Scion:  baseline={scion_baseline_risk:.4f}, guided={scion_guided_risk:.4f}")
print(f"Operon adaptive: {operon_risk:.4f}")
print()
print("--- all assertions passed ---")
