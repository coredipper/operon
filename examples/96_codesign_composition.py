"""
Example 96 — Co-Design Adapter Composition
=============================================

Demonstrates Zardini-style co-design formalization: convergence adapters
as design problems composed via series/parallel/feedback operations.

Usage:
    python examples/96_codesign_composition.py
"""

from operon_ai.convergence import (
    DesignProblem,
    compose_series,
    compose_parallel,
    feedback_fixed_point,
    feasibility_check,
)

# ---------------------------------------------------------------------------
# 1. Define adapter design problems
# ---------------------------------------------------------------------------

# Swarms adapter: pattern spec → topology advice + risk score
swarms_dp = DesignProblem(
    name="SwarmTopologyAdapter",
    evaluate_fn=lambda r: {
        "topology_advice": f"analyzed_{r.get('pattern', 'unknown')}",
        "risk_score": 0.3 if r.get("agents", 0) < 5 else 0.7,
    },
    feasibility_fn=lambda r: "pattern" in r and r.get("agents", 0) > 0,
)

# Memory bridge: memory entries → bi-temporal facts
memory_dp = DesignProblem(
    name="MemoryBridgeAdapter",
    evaluate_fn=lambda r: {
        "facts_created": len(r.get("entries", [])),
        "audit_trail": True,
    },
    feasibility_fn=lambda r: len(r.get("entries", [])) > 0,
)

print("=== Individual Design Problems ===")
result = feasibility_check(swarms_dp, {"pattern": "HierarchicalSwarm", "agents": 4})
print(f"  Swarms: feasible={result['feasible']}, risk={result.get('functionalities', {}).get('risk_score')}")

result = feasibility_check(memory_dp, {"entries": [{"id": 1}, {"id": 2}]})
print(f"  Memory: feasible={result['feasible']}, facts={result.get('functionalities', {}).get('facts_created')}")
print()

# ---------------------------------------------------------------------------
# 2. Series composition: Swarms analysis → Memory bridge
# ---------------------------------------------------------------------------

pipeline = compose_series(swarms_dp, memory_dp, name="SwarmsThenMemory")
print(f"=== Series: {pipeline.name} ===")
# This won't be feasible because swarms output doesn't have 'entries'
check = feasibility_check(pipeline, {"pattern": "SequentialWorkflow", "agents": 2})
print(f"  Feasible: {check['feasible']} (expected: False — output mismatch)")
print()

# ---------------------------------------------------------------------------
# 3. Parallel composition: both run on same input
# ---------------------------------------------------------------------------

parallel = compose_parallel(swarms_dp, memory_dp, name="SwarmsAndMemory")
print(f"=== Parallel: {parallel.name} ===")
combined_input = {"pattern": "HierarchicalSwarm", "agents": 3, "entries": [{"id": 1}]}
check = feasibility_check(parallel, combined_input)
print(f"  Feasible: {check['feasible']}")
if check['feasible']:
    print(f"  Output: {check['functionalities']}")
print()

# ---------------------------------------------------------------------------
# 4. Feedback fixed-point: adaptive scoring loop
# ---------------------------------------------------------------------------

# Simulate scoring loop: each iteration improves score toward 0.95
scoring_dp = DesignProblem(
    name="ScoringLoop",
    evaluate_fn=lambda s: {
        "score": s.get("score", 0.0) * 0.8 + 0.95 * 0.2,  # EMA toward 0.95
        "iterations": s.get("iterations", 0) + 1,
    },
)

final_state, iterations, converged = feedback_fixed_point(
    scoring_dp,
    initial={"score": 0.5, "iterations": 0},
    convergence_key="score",
    epsilon=0.001,
    max_iterations=50,
)

print(f"=== Feedback Fixed Point ===")
print(f"  Converged: {converged}")
print(f"  Iterations: {iterations}")
print(f"  Final score: {final_state['score']:.4f}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert swarms_dp.is_feasible({"pattern": "test", "agents": 1})
assert not swarms_dp.is_feasible({})
assert converged is True
assert iterations < 50
assert abs(final_state["score"] - 0.95) < 0.01
print("\n--- all assertions passed ---")
