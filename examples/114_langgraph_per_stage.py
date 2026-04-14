"""
Example 114: Per-Stage LangGraph Compiler
==========================================

Demonstrates the per-stage LangGraph compiler that creates one
LangGraph node per SkillStage. Each node calls
``organism.run_single_stage()``, so all structural guarantees
(certificates, watcher, halt_on_block) are handled by the organism.

1. Build a 3-stage organism
2. Compile to a per-stage LangGraph graph
3. Inspect the graph topology (one node per stage)
4. Execute via ``run_organism_langgraph()`` — full lifecycle
5. Verify certificates survived compilation

Key insight: LangGraph provides the execution host, graph topology,
and observability. The organism provides the structural guarantees.
The compiler is a functor — it maps structure without reimplementing
behavior.

References:
  Liu (arXiv:2604.11767) — Typed lambda calculus for agent composition

Usage: python examples/114_langgraph_per_stage.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.langgraph_compiler import (
    organism_to_langgraph,
    run_organism_langgraph,
)


def build_organism():
    """Build a 3-stage coding pipeline with mock providers."""
    fast = Nucleus(provider=MockProvider(responses={
        "classify": "ROUTE: code_fix",
    }))
    deep = Nucleus(provider=MockProvider(responses={
        "implement": "Applied fix: session validation added to login handler.",
    }))

    return skill_organism(
        stages=[
            SkillStage(
                name="intake",
                role="Normalizer",
                handler=lambda task, state, outputs, stage: {
                    "request": task,
                    "channel": "api",
                    "normalized": True,
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
                instructions="Implement the requested change.",
                mode="fuzzy",
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=1000, silent=True),
    )


def main():
    print("=" * 60)
    print("Per-Stage LangGraph Compiler")
    print("=" * 60)

    org = build_organism()

    # ------------------------------------------------------------------
    # 1. Compile to LangGraph — one node per stage
    # ------------------------------------------------------------------
    print("\n--- Compile ---")
    graph = organism_to_langgraph(org)

    # Inspect the graph topology
    all_nodes = list(graph.nodes.keys())
    # LangGraph adds __start__ and __end__ pseudo-nodes
    stage_nodes = [n for n in all_nodes if not n.startswith("__")]

    print(f"  total nodes:    {len(all_nodes)}")
    print(f"  stage nodes:    {stage_nodes}")
    print(f"  organism stages: {[s.name for s in org.stages]}")
    print(f"  1:1 mapping:    {set(stage_nodes) == {s.name for s in org.stages}}")

    # ------------------------------------------------------------------
    # 2. Execute via run_organism_langgraph (full lifecycle)
    # ------------------------------------------------------------------
    print("\n--- Execute ---")
    result = run_organism_langgraph(org, task="Fix the login bug")

    print(f"  output:         {result.output[:60]}...")
    print(f"  stages run:     {result.metadata.get('stages_completed', [])}")
    print(f"  halted:         {result.metadata.get('halted', False)}")
    print(f"  timing:         {result.timing_ms:.1f} ms")

    # Per-stage outputs
    print("\n--- Per-Stage Outputs ---")
    for stage_name, output in result.stage_outputs.items():
        display = output[:50] + "..." if len(str(output)) > 50 else output
        print(f"  {stage_name}: {display}")

    # ------------------------------------------------------------------
    # 3. Certificate verification
    # ------------------------------------------------------------------
    print("\n--- Certificates ---")
    for cv in result.certificates_verified:
        status = "HOLDS" if cv["holds"] else "FAILS"
        print(f"  {cv['theorem']}: {status}")

    # ------------------------------------------------------------------
    # 4. Compare: same organism, native execution
    # ------------------------------------------------------------------
    print("\n--- Native Execution (for comparison) ---")
    native_result = org.run("Fix the login bug")
    print(f"  output:         {native_result.final_output[:60]}...")
    print(f"  stages run:     {len(native_result.stage_results)}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    # Per-stage graph topology (set compare — node order not guaranteed)
    assert set(stage_nodes) == {s.name for s in org.stages}, (
        f"Expected 1:1 mapping, got {stage_nodes}"
    )

    # All stages executed
    assert len(result.stage_outputs) == 3, (
        f"Expected 3 stage outputs, got {len(result.stage_outputs)}"
    )

    # Certificates held
    for cv in result.certificates_verified:
        assert cv["holds"], f"{cv['theorem']} should hold"

    # LangGraph and native produce same stages
    native_stages = [sr.stage_name for sr in native_result.stage_results]
    lg_stages = result.metadata.get("stages_completed", [])
    assert native_stages == lg_stages, (
        f"Stage order mismatch: native={native_stages} vs lg={lg_stages}"
    )

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
