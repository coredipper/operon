"""
Example 83 — CLI Stage Handler
================================

Demonstrates cli_handler(): shell out to external CLI tools as organism
stages. The watcher, convergence detection, and developmental gating all
work unchanged on CLI-backed stages.

Usage:
    python examples/83_cli_stage_handler.py
"""

from operon_ai import (
    MockProvider,
    Nucleus,
    SkillStage,
    WatcherComponent,
    cli_handler,
    skill_organism,
)

# ---------------------------------------------------------------------------
# 1. Basic: echo handler
# ---------------------------------------------------------------------------

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

organism = skill_organism(
    stages=[
        SkillStage(name="echo", role="Echo", handler=cli_handler("echo")),
        SkillStage(name="count", role="Counter",
                   handler=cli_handler("wc -w", input_mode="stdin")),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
)

result = organism.run("hello world from CLI")

print("=== Basic CLI Pipeline ===")
print(f"  Echo output: {result.stage_results[0].output}")
print(f"  Word count:  {result.stage_results[1].output}")
print()

# ---------------------------------------------------------------------------
# 2. Failure detection with watcher
# ---------------------------------------------------------------------------

watcher = WatcherComponent()

organism2 = skill_organism(
    stages=[
        SkillStage(name="succeed", role="OK", handler=cli_handler("echo")),
        SkillStage(name="fail", role="Fail", handler=cli_handler("false")),
        SkillStage(name="unreachable", role="Never",
                   handler=lambda t: "should not run"),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[watcher],
    halt_on_block=True,
)

result2 = organism2.run("test failure handling")

print("=== Failure Detection ===")
print(f"  Stages run: {len(result2.stage_results)}")
print(f"  Last action: {result2.stage_results[-1].action_type}")
print(f"  Watcher: {watcher.summary()['total_interventions']} interventions")
print()

# ---------------------------------------------------------------------------
# 3. Custom output parsing
# ---------------------------------------------------------------------------

import json

organism3 = skill_organism(
    stages=[
        SkillStage(
            name="json_gen",
            role="Generator",
            handler=cli_handler(
                ["python3", "-c", "import json; print(json.dumps({'status': 'ok', 'count': 42}))"],
                input_mode="none",
                parse_output=json.loads,
            ),
        ),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
)

result3 = organism3.run("generate data")
print("=== JSON Parsing ===")
print(f"  Parsed output: {result3.stage_results[0].output}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert "hello world from CLI" in str(result.stage_results[0].output)
assert len(result2.stage_results) == 2  # halt after fail, unreachable skipped
assert result2.stage_results[-1].action_type == "FAILURE"
print("\n--- all assertions passed ---")
