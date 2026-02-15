"""
Example 37: Metabolic Swarm Budgeting (Coalgebraic Resource Constraints)
========================================================================

Demonstrates the Metabolic Coalgebra formalism where a swarm of agents
shares a finite token budget. The system models resource-constrained
computation using the theory of Quantitative Polynomial Functors.

Key concepts:
- Metabolic Coalgebra: State S = L x R (Logical State x Resource State)
- Ischemia Detection: Apoptosis when r < c (budget insufficient)
- Shared Mitochondria: Global ATP pool for multi-agent coordination
- Halting Guarantee: Strictly decreasing resource state ensures termination

Prerequisites:
- Example 04 for basic ATP budgeting concepts
- Example 25 for resource allocation trade-offs

See Also:
- Article Section 3.5: Metabolic Coalgebras - Formalizing Resource Constraints
- Lynch (2015): The bioenergetic costs of a gene
- Nakov (2021): Quantitative Polynomial Functors

Usage:
    python examples/37_metabolic_swarm_budgeting.py           # Run demo
    python examples/37_metabolic_swarm_budgeting.py --test    # Smoke test
"""

import random
import sys
import time
from dataclasses import dataclass

from operon_ai import (
    ATP_Store,
)


@dataclass
class AgentState:
    """
    Represents S = L x R (Logical State x Resource State).

    In the Metabolic Coalgebra formalism:
    - L (logical state): The agent's memory/context
    - R (resource state): Managed externally by SharedMitochondria
    """
    memory: list[str]
    name: str


class MetabolicAgent:
    """
    An agent whose transitions are guarded by metabolic cost.

    The structure map alpha: S -> P(S) + bot is a partial map:
    - alpha(l, r) = (l', r - c) if r >= c
    - alpha(l, r) = bot if r < c (Apoptosis)
    """

    def __init__(self, name: str, cost_per_step: int):
        self.name = name
        self.cost = cost_per_step  # The metabolic rate (c)
        self.state = AgentState(memory=[], name=name)
        self.alive = True

    def step(self, mitochondria: ATP_Store, task: str) -> str:
        """
        The Transition Function: (l, r) -> (l', r - c) | bot

        Returns:
            "SOLVED: <task>" on success
            "THINKING" on partial progress
            "DEAD" on apoptosis (insufficient resources)
        """
        if not self.alive:
            return "DEAD"

        # 1. Check Resource Constraint (Ischemia Check)
        if not mitochondria.consume(self.cost, operation=f"{self.name}:step"):
            print(f"  [{self.name}] APOPTOSIS - Starved of tokens")
            self.alive = False
            return "DEAD"

        # 2. Perform Logical Work (Mock LLM Call)
        print(f"  [{self.name}] Consumed {self.cost} ATP. Processing: {task[:30]}...")
        time.sleep(0.05)  # Simulate computation

        # 3. Update logical state
        self.state.memory.append(f"Attempted: {task}")

        # 4. Probabilistic success (mock LLM reasoning)
        if random.random() > 0.7:
            return f"SOLVED: {task}"
        return "THINKING"


class MetabolicSwarm:
    """
    A swarm of agents sharing a global ATP budget (SharedMitochondria).

    This implements the tensor product of agents in the Metabolic Coalgebra
    framework, where all agents draw from the same resource pool.
    """

    def __init__(
        self,
        workers: list[MetabolicAgent],
        verifier: MetabolicAgent,
        budget: ATP_Store,
    ):
        self.workers = workers
        self.verifier = verifier
        self.budget = budget
        self.max_cycles = 10  # Entropy-based apoptosis limit

    def run(self, task: str) -> dict:
        """
        Execute the swarm until solution, starvation, or cycle limit.

        Returns:
            dict with 'success', 'cycles', 'cause', 'remaining_budget'
        """
        cycle = 0

        while cycle < self.max_cycles:
            cycle += 1
            print(f"\n--- Cycle {cycle} | Remaining ATP: {self.budget.atp} ---")

            # Check for Total System Failure (Global Ischemia)
            if self.budget.atp <= 0:
                print("  SYSTEM FAILURE: Global Ischemia (Token exhaustion)")
                return {
                    "success": False,
                    "cycles": cycle,
                    "cause": "ischemia",
                    "remaining_budget": 0,
                }

            # Parallel Execution (Tensor Product of agents)
            candidates = []
            for worker in self.workers:
                if worker.alive:
                    result = worker.step(self.budget, task)
                    if result.startswith("SOLVED"):
                        candidates.append(result)

            # Check if all workers died
            alive_workers = [w for w in self.workers if w.alive]
            if not alive_workers and not candidates:
                print("  All workers terminated. Swarm collapse.")
                return {
                    "success": False,
                    "cycles": cycle,
                    "cause": "swarm_collapse",
                    "remaining_budget": self.budget.atp,
                }

            # Quorum / Verification Logic
            if candidates:
                print(f"  Candidates found: {len(candidates)}. Verifying...")
                verdict = self.verifier.step(self.budget, "Verify Solution")

                if verdict == "DEAD":
                    print("  Verifier died before confirming. Verification failed.")
                    return {
                        "success": False,
                        "cycles": cycle,
                        "cause": "verifier_death",
                        "remaining_budget": self.budget.atp,
                    }

                print("  SOLUTION VERIFIED & ACCEPTED")
                return {
                    "success": True,
                    "cycles": cycle,
                    "cause": "solved",
                    "remaining_budget": self.budget.atp,
                }

        # Cycle limit reached (Entropy-based apoptosis)
        print("  APOPTOSIS: Cycle limit reached (Entropy check)")
        return {
            "success": False,
            "cycles": cycle,
            "cause": "entropy_limit",
            "remaining_budget": self.budget.atp,
        }


def run_scenario(name: str, budget_amount: int, seed: int = 42) -> dict:
    """Run a single budgeted swarm scenario."""
    random.seed(seed)

    print("=" * 60)
    print(f"Scenario: {name}")
    print(f"Budget: {budget_amount} ATP")
    print("=" * 60)

    # Initialize shared resource pool
    budget = ATP_Store(budget=budget_amount)

    # Define the colony (3 Workers, 1 Verifier)
    workers = [
        MetabolicAgent("Worker_1", cost_per_step=10),
        MetabolicAgent("Worker_2", cost_per_step=10),
        MetabolicAgent("Worker_3", cost_per_step=10),
    ]
    verifier = MetabolicAgent("Verifier", cost_per_step=5)

    # Create swarm
    swarm = MetabolicSwarm(workers, verifier, budget)

    # Execute
    result = swarm.run("Prove P != NP")

    # Summary
    print()
    print("-" * 40)
    if result["success"]:
        print(f"  Result: SUCCESS in {result['cycles']} cycles")
    else:
        print(f"  Result: FAILED ({result['cause']}) after {result['cycles']} cycles")
    print(f"  Remaining ATP: {result['remaining_budget']}")
    print("-" * 40)

    return result


def main():
    print("=" * 60)
    print("Metabolic Swarm Budgeting - Coalgebraic Resource Constraints")
    print("=" * 60)
    print()
    print("This example demonstrates the Metabolic Coalgebra formalism")
    print("where agents undergo apoptosis when resources are exhausted.")
    print()

    # Scenario A: Starvation (Budget too low)
    result_a = run_scenario("Starvation (Insufficient Budget)", budget_amount=30, seed=42)
    print()

    # Scenario B: Success (Budget sufficient)
    result_b = run_scenario("Success (Sufficient Budget)", budget_amount=200, seed=42)

    print()
    print("=" * 60)
    print("Key Insight: The Metabolic Coalgebra guarantees halting.")
    print("Every transition consumes resources, so the system cannot")
    print("run forever - it either succeeds or undergoes apoptosis.")
    print("=" * 60)


def run_smoke_test():
    """Automated smoke test for CI."""
    random.seed(42)

    # Test 1: Starvation scenario
    budget = ATP_Store(budget=15)  # Only enough for ~1 worker step
    workers = [MetabolicAgent("W1", cost_per_step=10)]
    verifier = MetabolicAgent("V1", cost_per_step=5)
    swarm = MetabolicSwarm(workers, verifier, budget)

    result = swarm.run("Test task")
    assert not result["success"], "Should fail with insufficient budget"
    assert result["cause"] in ("ischemia", "swarm_collapse"), f"Unexpected cause: {result['cause']}"
    print("  Test 1: Starvation scenario - PASSED")

    # Test 2: Success scenario (with high probability)
    random.seed(123)  # Seed that leads to quick success
    budget = ATP_Store(budget=500)
    workers = [
        MetabolicAgent("W1", cost_per_step=10),
        MetabolicAgent("W2", cost_per_step=10),
    ]
    verifier = MetabolicAgent("V1", cost_per_step=5)
    swarm = MetabolicSwarm(workers, verifier, budget)

    result = swarm.run("Test task")
    # With 500 budget and seed 123, should eventually succeed or hit cycle limit
    assert result["remaining_budget"] < 500, "Should have consumed some budget"
    print("  Test 2: Budget consumption - PASSED")

    # Test 3: Individual agent apoptosis
    budget = ATP_Store(budget=5)
    agent = MetabolicAgent("Test", cost_per_step=10)
    result = agent.step(budget, "Task")
    assert result == "DEAD", "Should die with insufficient budget"
    assert not agent.alive, "Agent should be marked as dead"
    print("  Test 3: Agent apoptosis - PASSED")

    print()
    print("Smoke test passed!")


if __name__ == "__main__":
    try:
        if "--test" in sys.argv:
            run_smoke_test()
        else:
            main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        raise
