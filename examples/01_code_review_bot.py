"""
Example 1: Code Review Bot (CFFL Topology)
==========================================

Demonstrates the Coherent Feed-Forward Loop pattern where code changes
only proceed if BOTH the executor (code generator) AND the risk assessor
(security reviewer) approve.

This is a two-key execution guardrail: topology enforces an interlock
(you cannot execute without approval), but it does not guarantee that
the executor/verifier failures are statistically independent. In practice,
use diversity (models/prompts/tools) and tool-grounded verification to
reduce correlated errors.

Topology:
    User Request --> [Code Generator] --+
                                        |
                 --> [Security Review] --+--> [AND Gate] --> Output
"""

from operon_ai import (
    ATP_Store,
    CoherentFeedForwardLoop,
)


def main():
    try:
        print("=" * 60)
        print("Code Review Bot - CFFL Topology Demo")
        print("=" * 60)
        print()

        # Create a shared metabolic budget (token limit)
        budget = ATP_Store(budget=500)

        # The CFFL wires together:
        # - Gene_Z (Executor): Generates/runs code
        # - Gene_Y (RiskAssessor): Reviews for safety
        cffl = CoherentFeedForwardLoop(budget=budget, silent=True)

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
            result = cffl.run(request)
            if result.blocked:
                print(f"🛑 BLOCKED: {result.block_reason}")
            else:
                if result.approval_token:
                    print(f"✅ PERMITTED (approval: {result.approval_token.issuer})")
                else:
                    print("✅ PERMITTED")
            print()

        print("=" * 60)
        print(f"Final budget: {budget.atp} ATP")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
