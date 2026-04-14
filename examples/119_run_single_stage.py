"""
Example 119: Direct Per-Stage Execution
=========================================

Demonstrates using ``run_single_stage()`` directly for fine-grained
control over the execution loop.

1. Build a 3-stage organism
2. Execute stages one at a time via run_single_stage()
3. Inspect per-stage decisions ("continue" / "halt" / "blocked")
4. Inject custom state between stages
5. Show how to build a custom execution loop with early exit

This is the power-user API — useful when you need to intercept,
modify, or branch between stages. Both ``organism.run()`` and the
LangGraph compiler call ``run_single_stage()`` internally, so
behavior is identical.

Usage: python examples/119_run_single_stage.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.types import RunContext, SkillStageResult


def main():
    print("=" * 60)
    print("Direct Per-Stage Execution")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build organism
    # ------------------------------------------------------------------
    fast = Nucleus(provider=MockProvider(responses={
        "classify": "ROUTE: code_fix",
        "implement": "Fixed: added null check in auth handler",
    }))
    deep = Nucleus(provider=MockProvider(responses={}))

    org = skill_organism(
        stages=[
            SkillStage(
                name="intake",
                role="Normalizer",
                handler=lambda task, state, outputs, stage: {
                    "request": task,
                    "priority": "high" if "crash" in task.lower() else "normal",
                },
            ),
            SkillStage(
                name="router",
                role="Classifier",
                instructions="Classify the request type.",
                mode="fixed",
            ),
            SkillStage(
                name="executor",
                role="Engineer",
                instructions="Implement the fix.",
                mode="fuzzy",
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=1000, silent=True),
    )

    # ------------------------------------------------------------------
    # 2. Manual stage-by-stage execution
    # ------------------------------------------------------------------
    print("\n--- Manual Per-Stage Execution ---")

    task = "Fix crash in auth handler"
    state: dict = {}
    stage_outputs: dict = {}
    stage_results: list[SkillStageResult] = []

    # Call on_run_start for components
    for component in org.components:
        component.on_run_start(task, state)

    for stage in org.stages:
        print(f"\n  Stage: {stage.name}")

        decision = org.run_single_stage(
            stage, task, state, stage_outputs, stage_results,
        )

        print(f"    decision:  {decision}")
        print(f"    output:    {str(stage_outputs.get(stage.name, ''))[:60]}")

        if decision != "continue":
            print(f"    → pipeline stopped: {decision}")
            break
    else:
        print("\n  → all stages completed")

    # ------------------------------------------------------------------
    # 3. Inspect accumulated state
    # ------------------------------------------------------------------
    print("\n--- Accumulated State ---")
    print(f"  stages run:     {len(stage_results)}")
    print(f"  stage outputs:  {list(stage_outputs.keys())}")
    print(f"  last_stage:     {state.get('last_stage')}")
    print(f"  priority:       {state.get('intake', {}).get('priority', 'N/A')}")

    # ------------------------------------------------------------------
    # 4. Custom execution: inject state between stages
    # ------------------------------------------------------------------
    print("\n--- Custom Loop: State Injection ---")

    task2 = "Add unit test for login"
    state2: dict = {}
    outputs2: dict = {}
    results2: list[SkillStageResult] = []

    for component in org.components:
        component.on_run_start(task2, state2)

    # Run intake
    d1 = org.run_single_stage(
        org.stages[0], task2, state2, outputs2, results2,
    )
    print(f"  intake: {d1}")

    # Inject custom context before router
    state2["override_route"] = "testing_pipeline"
    state2["injected_by"] = "custom_loop"
    print(f"  → injected: override_route=testing_pipeline")

    # Run router — it sees the injected state
    d2 = org.run_single_stage(
        org.stages[1], task2, state2, outputs2, results2,
    )
    print(f"  router: {d2} (state has override_route={state2.get('override_route')})")

    # Run executor
    d3 = org.run_single_stage(
        org.stages[2], task2, state2, outputs2, results2,
    )
    print(f"  executor: {d3}")

    # ------------------------------------------------------------------
    # 5. RunContext typed accessors
    # ------------------------------------------------------------------
    print("\n--- RunContext Typed Accessors ---")
    ctx = RunContext(state2)
    print(f"  watcher_intervention: {ctx.watcher_intervention}")
    print(f"  verifier_signals:     {ctx.verifier_signals}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    # All 3 stages completed in manual loop
    assert len(stage_results) == 3
    assert state.get("last_stage") == "executor"

    # State injection is visible
    assert state2.get("override_route") == "testing_pipeline"
    assert state2.get("injected_by") == "custom_loop"

    # RunContext is a dict subclass
    assert isinstance(ctx, dict)

    # Custom loop completed all 3 stages
    assert len(results2) == 3

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
