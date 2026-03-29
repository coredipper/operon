"""
Example 95 — AsyncThink Fork/Join
====================================

Demonstrates Fork/Join execution within a single stage using
AsyncOrganizer, inspired by Chi et al.'s AsyncThink paper.

Usage:
    python examples/95_async_thinking.py
"""

from operon_ai.convergence import AsyncOrganizer, async_stage_handler

# 1. Basic Fork/Join
organizer = AsyncOrganizer(capacity=4)

result = organizer.fork(
    task="Analyze competitor landscape",
    sub_queries=["pricing data", "feature comparison", "market share"],
    handler=lambda q: f"Analysis of: {q}",
)

print("=== Fork/Join Results ===")
print(f"  Outputs: {len(result.outputs)}")
print(f"  Concurrency ratio (η): {result.concurrency_ratio:.2f}")
print(f"  Fork count: {result.fork_count}")
for out in result.outputs:
    print(f"    {out}")
print()

# 2. Critical path analysis
dag = {
    "research": [],           # independent
    "analyze": ["research"],  # depends on research
    "report": ["analyze"],    # depends on analyze
    "review": ["report"],     # depends on report
}
cpl = organizer.critical_path_latency(dag)
print(f"=== Critical Path ===")
print(f"  DAG: research → analyze → report → review")
print(f"  Critical path length: {cpl}")
print()

# 3. Stage handler factory
handler = async_stage_handler(
    organizer=organizer,
    decompose=lambda task: task.split("; "),
    handler=lambda q: q.upper(),
    join=lambda outputs: " | ".join(str(o) for o in outputs),
)

output = handler("find papers; summarize findings; write report")
print(f"=== Stage Handler ===")
print(f"  Output: {output['output']}")
print(f"  Fork count: {output['async_think']['fork_count']}")
print(f"  η: {output['async_think']['concurrency_ratio']:.2f}")

# --test
assert len(result.outputs) == 3
assert result.concurrency_ratio == 0.25  # 1/4 in sequential mode
assert cpl == 4.0  # linear chain of 4
assert output["async_think"]["fork_count"] == 3
print("\n--- all assertions passed ---")
