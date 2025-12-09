"""
Example 4: Budget-Aware Agent (ATP/Metabolic Topology)
======================================================

Demonstrates metabolic resource management where agents track their
"ATP" (token budget) and gracefully degrade when resources run low.

This mirrors how biological cells manage energy - when ATP is depleted,
non-essential processes shut down (ischemia) before critical failure.

Topology:
    Request --> [Budget Check] --> Sufficient ATP?
                                        |
                    YES: Execute normally (consume ATP)
                                        |
                    NO: Apoptosis (graceful shutdown)
"""

from operon_ai import BioAgent, Signal, ATP_Store


def run_agent_until_exhaustion(agent: BioAgent, tasks: list[str]):
    """Run an agent through tasks until ATP is exhausted."""
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}: {task[:50]}... ---")
        print(f"ATP before: {agent.atp.atp}")

        signal = Signal(content=task)
        result = agent.express(signal)

        print(f"Action: {result.action_type}")
        print(f"Result: {result.payload[:80]}..." if len(result.payload) > 80 else f"Result: {result.payload}")
        print(f"ATP after: {agent.atp.atp}")

        if result.action_type == "FAILURE" and "ATP" in result.payload:
            print("\n*** APOPTOSIS TRIGGERED - Agent shutting down ***")
            break


def main():
    print("=" * 60)
    print("Budget-Aware Agent - Metabolic Management Demo")
    print("=" * 60)

    # Scenario 1: Agent with generous budget
    print("\n" + "=" * 60)
    print("Scenario 1: Well-funded agent (100 ATP)")
    print("=" * 60)

    generous_budget = ATP_Store(budget=100)
    wealthy_agent = BioAgent(
        name="WealthyWorker",
        role="Executor",
        atp_store=generous_budget
    )

    tasks = [
        "Calculate the sum of 1 to 100",
        "Analyze the sentiment of this text",
        "Generate a summary of the document",
        "Translate this to French",
        "Create a bullet-point list",
    ]

    run_agent_until_exhaustion(wealthy_agent, tasks)

    # Scenario 2: Agent with limited budget
    print("\n" + "=" * 60)
    print("Scenario 2: Resource-constrained agent (25 ATP)")
    print("=" * 60)

    limited_budget = ATP_Store(budget=25)
    frugal_agent = BioAgent(
        name="FrugalWorker",
        role="Executor",
        atp_store=limited_budget
    )

    run_agent_until_exhaustion(frugal_agent, tasks)

    # Scenario 3: Multiple agents sharing a budget
    print("\n" + "=" * 60)
    print("Scenario 3: Shared budget between agents (50 ATP total)")
    print("=" * 60)

    shared_budget = ATP_Store(budget=50)

    agent_a = BioAgent(name="AgentA", role="Executor", atp_store=shared_budget)
    agent_b = BioAgent(name="AgentB", role="Executor", atp_store=shared_budget)
    agent_c = BioAgent(name="AgentC", role="Executor", atp_store=shared_budget)

    agents = [agent_a, agent_b, agent_c]

    print(f"Initial shared budget: {shared_budget.atp} ATP")

    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} ---")
        for agent in agents:
            if shared_budget.atp <= 0:
                print(f"{agent.name}: Cannot act - shared budget exhausted")
                continue

            signal = Signal(content=f"Task for round {round_num}")
            result = agent.express(signal)
            print(f"{agent.name}: {result.action_type} (Budget: {shared_budget.atp})")

    print("\n" + "=" * 60)
    print("Key Insight: Metabolic budgeting prevents runaway costs.")
    print("Agents that exhaust their ATP undergo apoptosis rather")
    print("than failing unpredictably mid-task.")
    print("=" * 60)


if __name__ == "__main__":
    main()
