"""
Example 85 — Claude Code Pipeline
====================================

A real multi-stage pipeline using Claude Code CLI. Each stage calls
`claude --print` with context from previous stages. The watcher monitors
for failures and convergence.

Requires: Claude Code CLI installed (`claude` in PATH).
This example makes live LLM calls and costs tokens.

Usage:
    python examples/85_claude_code_pipeline.py
"""

import shutil
import sys

if not shutil.which("claude"):
    print("Claude Code CLI not found. Install from https://claude.ai/claude-code")
    print("Skipping example.")
    sys.exit(0)

from operon_ai import (
    BiTemporalMemory,
    MockProvider,
    Nucleus,
    SkillStage,
    WatcherComponent,
    WatcherConfig,
    cli_handler,
    managed_organism,
)

# ---------------------------------------------------------------------------
# 1. Chained handler: passes previous stage output as context
# ---------------------------------------------------------------------------


def chained_claude(stage_name: str, system_prompt: str = ""):
    """Build a handler that chains previous stage output into Claude's input."""

    def handler(task, state=None, outputs=None):
        state = state or {}
        outputs = outputs or {}

        # Build context from all previous stage outputs
        context_parts = []
        for prev_name, prev_output in outputs.items():
            if prev_name == stage_name:
                continue
            text = prev_output
            if isinstance(text, dict):
                text = text.get("output", str(text))
            context_parts.append(f"[{prev_name}]:\n{text}")

        context = "\n\n".join(context_parts)
        full_prompt = task
        if context:
            full_prompt = f"{task}\n\n--- Context from previous stages ---\n{context}"
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n{full_prompt}"

        return cli_handler(
            "claude --print",
            input_mode="stdin",
            timeout=120.0,
        )(full_prompt)

    return handler


# ---------------------------------------------------------------------------
# 2. Build the pipeline
# ---------------------------------------------------------------------------

# Nuclei needed for managed_organism but not used by CLI stages
fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

m = managed_organism(
    stages=[
        SkillStage(
            name="plan",
            role="Planner",
            handler=chained_claude(
                "plan",
                system_prompt="You are a senior engineer. Break this task into 2-3 concrete steps. Be concise — 3 sentences max.",
            ),
        ),
        SkillStage(
            name="implement",
            role="Implementer",
            handler=chained_claude(
                "implement",
                system_prompt="You are a Python developer. Given the plan, write the implementation. Return only code, no explanation. Keep it under 30 lines.",
            ),
        ),
        SkillStage(
            name="review",
            role="Reviewer",
            handler=chained_claude(
                "review",
                system_prompt="You are a code reviewer. Review the implementation for bugs, security issues, and style. Be concise — 3 bullet points max.",
            ),
        ),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    watcher=True,
    watcher_config=WatcherConfig(max_retries_per_stage=1),
    substrate=BiTemporalMemory(),
)

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

task = "Write a Python function that validates an email address using regex"

print(f"Task: {task}")
print("=" * 60)
print("Running 3-stage pipeline: plan → implement → review")
print("Each stage calls `claude --print` with context chaining...")
print()

result = m.run(task)

# ---------------------------------------------------------------------------
# 4. Show results
# ---------------------------------------------------------------------------

for sr in result.run_result.stage_results:
    output = sr.output
    if isinstance(output, dict):
        output = output.get("output", str(output))
    print(f"--- {sr.stage_name} ({sr.role}) ---")
    print(output[:500] if isinstance(output, str) else str(output)[:500])
    print()

# ---------------------------------------------------------------------------
# 5. Watcher + status
# ---------------------------------------------------------------------------

print("=== Watcher ===")
ws = result.watcher_summary
if ws:
    print(f"  Stages observed: {ws['total_stages_observed']}")
    print(f"  Interventions: {ws['total_interventions']}")
    print(f"  Convergent: {ws['convergent']}")
print()

print("=== Status ===")
for key, value in m.status().items():
    print(f"  {key}: {value}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert len(result.run_result.stage_results) == 3
assert result.watcher_summary is not None
# At least one stage should have produced non-empty output
outputs = [sr.output for sr in result.run_result.stage_results]
non_empty = [o for o in outputs if o and (isinstance(o, str) and len(o) > 5 or isinstance(o, dict))]
assert len(non_empty) >= 1, "Expected at least one stage to produce output"
print("\n--- all assertions passed ---")
