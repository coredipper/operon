"""
Example 68: Pattern-First API
=============================

Demonstrates the thin coordination wrappers introduced after the
epistemic topology work:

1. `advise_topology(...)` for quick architecture guidance
2. `reviewer_gate(...)` for single-worker + reviewer execution
3. `specialist_swarm(...)` for centralized specialist coordination
"""

from __future__ import annotations

import sys

from operon_ai import advise_topology, reviewer_gate, specialist_swarm


def main() -> None:
    print("=" * 68)
    print("Pattern-First API")
    print("=" * 68)

    print("\n--- 1. Topology Advice ---")
    seq = advise_topology(
        task_shape="sequential",
        tool_count=2,
        subtask_count=3,
        error_tolerance=0.02,
    )
    par = advise_topology(
        task_shape="parallel",
        tool_count=4,
        subtask_count=3,
        error_tolerance=0.1,
    )
    print(f"Sequential task -> {seq.recommended_pattern}: {seq.rationale}")
    print(f"Parallel task   -> {par.recommended_pattern}: {par.rationale}")

    print("\n--- 2. Reviewer Gate ---")
    gate = reviewer_gate(
        executor=lambda prompt: f"EXECUTE: {prompt}",
        reviewer=lambda prompt, candidate: "safe" in prompt.lower() and "EXECUTE" in candidate,
    )
    decision = gate.run("Deploy safe schema migration")
    print(f"Allowed: {decision.allowed}")
    print(f"Status:  {decision.status}")
    print(f"Output:  {decision.output}")
    print(f"Class:   {gate.analysis.classification.topology_class.value}")

    print("\n--- 3. Specialist Swarm ---")
    swarm = specialist_swarm(
        roles=["legal", "security", "finance"],
        workers={
            "legal": lambda task, role: f"{role}: contract risk is low",
            "security": lambda task, role: f"{role}: controls look adequate",
            "finance": lambda task, role: f"{role}: annual cost is acceptable",
        },
        aggregator=lambda task, outputs: " | ".join(outputs.values()),
    )
    result = swarm.run("Assess this vendor")
    print(f"Outputs:    {result.outputs}")
    print(f"Aggregate:  {result.aggregate}")
    print(f"Class:      {result.analysis.classification.topology_class.value}")
    print(f"Amplifies:  {result.analysis.error_bound.amplification_ratio:.2f}x")


if __name__ == "__main__":
    try:
        main()
        if "--test" in sys.argv:
            print("\n[OK] Pattern-first example completed.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
