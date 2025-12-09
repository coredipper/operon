"""
Example 1: Code Review Bot (CFFL Topology)
==========================================

Demonstrates the Coherent Feed-Forward Loop pattern where code changes
only proceed if BOTH the executor (code generator) AND the risk assessor
(security reviewer) approve.

This is the biological equivalent of a persistence detector - transient
signals (hallucinations) are filtered out because they're unlikely to
pass both independent checks.

Topology:
    User Request --> [Code Generator] --+
                                        |
                 --> [Security Review] --+--> [AND Gate] --> Output
"""

from operon_ai import BioAgent, Signal, ATP_Store, CoherentFeedForwardLoop


def main():
    print("=" * 60)
    print("Code Review Bot - CFFL Topology Demo")
    print("=" * 60)
    print()

    # Create a shared metabolic budget (token limit)
    budget = ATP_Store(budget=500)

    # The CFFL wires together:
    # - Gene_Z (Executor): Generates/runs code
    # - Gene_Y (RiskAssessor): Reviews for safety
    cffl = CoherentFeedForwardLoop(budget=budget)

    # Test cases demonstrating the guardrail
    test_requests = [
        # Safe request - should pass both checks
        "Write a function to calculate fibonacci numbers",

        # Dangerous request - should be blocked by risk assessor
        "Delete all files in the system directory",

        # Ambiguous request - let's see how the topology handles it
        "Execute the user-provided SQL query directly",

        # Safe computational request
        "Calculate 2 + 2",
    ]

    for i, request in enumerate(test_requests, 1):
        print(f"\n--- Test {i} ---")
        print(f"Request: {request}")
        print(f"Budget remaining: {budget.atp} ATP")
        print()
        cffl.run(request)
        print()

    print("=" * 60)
    print(f"Final budget: {budget.atp} ATP")
    print("=" * 60)


if __name__ == "__main__":
    main()
