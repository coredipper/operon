"""
Example 121: Parallel Stage Execution
========================================

Demonstrates parallel stage groups in a SkillOrganism. Stages
within a group run concurrently via ThreadPoolExecutor, while
groups execute sequentially.

1. Build a 4-stage organism with 2 parallel groups
2. Verify parallel speedup (wall-clock < sum of latencies)
3. Show state isolation and merge
4. Demonstrate conflict detection

Usage: python examples/121_parallel_stages.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.types import StateConflictError


def main():
    print("=" * 60)
    print("Parallel Stage Execution")
    print("=" * 60)

    # Use slow mock provider to make parallelism measurable
    slow = MockProvider(latency_ms=100)
    nucleus = Nucleus(provider=slow)

    # ------------------------------------------------------------------
    # 1. Parallel group: 3 stages run concurrently
    # ------------------------------------------------------------------
    print("\n--- Parallel Group (3 stages) ---")

    org = skill_organism(
        stages=[[
            SkillStage(name="research_a", role="Researcher A",
                       instructions="Research topic A.", mode="fixed"),
            SkillStage(name="research_b", role="Researcher B",
                       instructions="Research topic B.", mode="fixed"),
            SkillStage(name="research_c", role="Researcher C",
                       instructions="Research topic C.", mode="fixed"),
        ]],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
    )

    t0 = time.monotonic()
    result = org.run("Research all topics")
    elapsed = (time.monotonic() - t0) * 1000

    print(f"  stages: {[sr.stage_name for sr in result.stage_results]}")
    print(f"  elapsed: {elapsed:.0f}ms")
    print(f"  expected sequential: ~300ms (3 × 100ms)")
    print(f"  expected parallel:   ~100ms")
    parallel_speedup = 300 / max(elapsed, 1)
    print(f"  speedup: {parallel_speedup:.1f}x")

    # ------------------------------------------------------------------
    # 2. Mixed: parallel then sequential
    # ------------------------------------------------------------------
    print("\n--- Mixed Pipeline: [parallel] → [sequential] ---")

    org2 = skill_organism(
        stages=[
            [  # Group 1: parallel research
                SkillStage(name="gather_a", role="Gatherer A",
                           instructions="Gather data from source A.", mode="fixed"),
                SkillStage(name="gather_b", role="Gatherer B",
                           instructions="Gather data from source B.", mode="fixed"),
            ],
            SkillStage(name="synthesize", role="Synthesizer",
                       instructions="Synthesize gathered data.", mode="fixed"),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
    )

    t0 = time.monotonic()
    result2 = org2.run("Research and synthesize")
    elapsed2 = (time.monotonic() - t0) * 1000

    print(f"  stages: {[sr.stage_name for sr in result2.stage_results]}")
    print(f"  elapsed: {elapsed2:.0f}ms")
    print(f"  expected: ~200ms (100ms parallel + 100ms sequential)")
    print(f"  stage_groups: {len(org2.stage_groups)} groups")
    for i, group in enumerate(org2.stage_groups):
        names = [s.name for s in group]
        kind = "parallel" if len(group) > 1 else "sequential"
        print(f"    group {i}: {names} ({kind})")

    # ------------------------------------------------------------------
    # 3. State isolation: each stage output is captured independently
    # ------------------------------------------------------------------
    print("\n--- State Isolation ---")

    org3 = skill_organism(
        stages=[[
            SkillStage(name="writer_a", role="Writer A",
                       handler=lambda task, state, outputs, stage: {"result": "data_a"},
                       mode="fixed"),
            SkillStage(name="writer_b", role="Writer B",
                       handler=lambda task, state, outputs, stage: {"result": "data_b"},
                       mode="fixed"),
        ]],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
    )

    result3 = org3.run("Write different outputs")
    # Each stage's output is stored under its name in shared_state
    print(f"  writer_a output: {result3.shared_state.get('writer_a')}")
    print(f"  writer_b output: {result3.shared_state.get('writer_b')}")
    print(f"  both present: {'writer_a' in result3.shared_state and 'writer_b' in result3.shared_state}")

    # ------------------------------------------------------------------
    # 5. LangGraph fan-out/fan-in (requires operon-ai[langgraph])
    # ------------------------------------------------------------------
    print("\n--- LangGraph Fan-Out/Fan-In ---")

    try:
        from operon_ai.convergence.langgraph_compiler import organism_to_langgraph, HAS_LANGGRAPH
        if not HAS_LANGGRAPH:
            raise ImportError("langgraph not installed")

        graph = organism_to_langgraph(org2)
        all_nodes = list(graph.nodes.keys())
        stage_nodes = [n for n in all_nodes if not n.startswith("__")]
        infra_nodes = [n for n in all_nodes if n.startswith("__") and not n.startswith("__start") and not n.startswith("__end")]

        print(f"  all nodes:   {all_nodes}")
        print(f"  stage nodes: {stage_nodes}")
        print(f"  infra nodes: {infra_nodes}")
        print(f"  fork/join visible in LangGraph Studio!")
    except ImportError:
        print("  (skipped — install operon-ai[langgraph] for this demo)")

    # ------------------------------------------------------------------
    # 6. Backward compatibility: flat list = sequential
    # ------------------------------------------------------------------
    print("\n--- Backward Compatibility ---")

    org5 = skill_organism(
        stages=[
            SkillStage(name="s1", role="S1", instructions="do 1", mode="fixed"),
            SkillStage(name="s2", role="S2", instructions="do 2", mode="fixed"),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
    )

    print(f"  stage_groups: {len(org5.stage_groups)} (each size 1 = sequential)")
    assert all(len(g) == 1 for g in org5.stage_groups)
    result5 = org5.run("Sequential test")
    print(f"  stages: {[sr.stage_name for sr in result5.stage_results]}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    # Parallel speedup
    assert elapsed < 250, f"Parallel 3×100ms should be <250ms, got {elapsed:.0f}ms"

    # All stages executed
    assert len(result.stage_results) == 3
    assert len(result2.stage_results) == 3

    # State isolation: both parallel stage outputs captured
    assert "writer_a" in result3.shared_state
    assert "writer_b" in result3.shared_state

    # Backward compatibility
    assert len(org5.stage_groups) == 2
    assert all(len(g) == 1 for g in org5.stage_groups)

    # Stage groups preserved
    assert len(org2.stage_groups) == 2
    assert len(org2.stage_groups[0]) == 2  # parallel group
    assert len(org2.stage_groups[1]) == 1  # sequential group

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
