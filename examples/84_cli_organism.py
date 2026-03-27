"""
Example 84 — CLI Organism
===========================

Demonstrates cli_organism(): build a full managed organism from a dict
of CLI commands. Each stage receives the original task string via stdin
(stages are independent, not piped). The watcher monitors all stages,
and a BiTemporalMemory substrate is attached for optional fact recording.

Note: to record outputs as bi-temporal facts, configure stages with
``emit_output_fact=True`` or a ``fact_extractor``. By default the
substrate is attached but empty.

Usage:
    python examples/84_cli_organism.py
"""

from operon_ai import (
    BiTemporalMemory,
    MockProvider,
    Nucleus,
    cli_organism,
)

# ---------------------------------------------------------------------------
# 1. Build a CLI organism from a command dict
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

m = cli_organism(
    commands={
        "generate": "echo",
        "transform": ["tr", "a-z", "A-Z"],
        "count": "wc -c",
    },
    input_mode="stdin",
    fast_nucleus=fast,
    deep_nucleus=deep,
    watcher=True,
    substrate=BiTemporalMemory(),
)

# ---------------------------------------------------------------------------
# 2. Run the organism (each stage receives the task independently)
# ---------------------------------------------------------------------------

result = m.run("hello from the cli organism")

print("=== CLI Organism Pipeline ===")
for sr in result.run_result.stage_results:
    output = sr.output
    if isinstance(output, dict):
        output = output.get("output", output)
    print(f"  {sr.stage_name}: {output}")
print()

# ---------------------------------------------------------------------------
# 3. Watcher summary
# ---------------------------------------------------------------------------

print("=== Watcher ===")
print(f"  {result.watcher_summary}")
print()

# ---------------------------------------------------------------------------
# 4. Substrate facts
# ---------------------------------------------------------------------------

print("=== Substrate ===")
status = m.status()
print(f"  Facts recorded: {status.get('substrate_facts', 0)}")
print()

# ---------------------------------------------------------------------------
# 5. Status
# ---------------------------------------------------------------------------

print("=== Full Status ===")
for key, value in m.status().items():
    print(f"  {key}: {value}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert len(result.run_result.stage_results) == 3
assert result.watcher_summary is not None
assert result.watcher_summary["total_stages_observed"] == 3
print("\n--- all assertions passed ---")
