"""
Example 105 -- Prompt Optimization Interface (Phase C7)
========================================================

Demonstrates the PromptOptimizer protocol: create a NoOpOptimizer, attach
it to a multi-stage organism, and verify scoring.

Usage:
    python examples/105_prompt_optimization_interface.py
"""

from operon_ai.convergence.prompt_optimization import (
    NoOpOptimizer,
    PromptOptimizer,
    attach_optimizer,
)
from operon_ai.patterns.types import SkillStage

# ---------------------------------------------------------------------------
# 1. Create NoOpOptimizer and verify Protocol conformance
# ---------------------------------------------------------------------------

opt = NoOpOptimizer()
print("=== Protocol conformance ===")
print(f"  isinstance(opt, PromptOptimizer): {isinstance(opt, PromptOptimizer)}")
assert isinstance(opt, PromptOptimizer)

# ---------------------------------------------------------------------------
# 2. Build a 3-stage organism (placeholder handlers)
# ---------------------------------------------------------------------------


def _placeholder(task, state, outputs, stage, *, label=""):
    return f"[{label}] {task}"


stages = [
    SkillStage(name="planner", role="planner", mode="fuzzy",
               handler=lambda t, s, o, st: _placeholder(t, s, o, st, label="planner")),
    SkillStage(name="executor", role="executor", mode="fuzzy",
               handler=lambda t, s, o, st: _placeholder(t, s, o, st, label="executor")),
    SkillStage(name="reviewer", role="reviewer", mode="fixed",
               handler=lambda t, s, o, st: _placeholder(t, s, o, st, label="reviewer")),
]

print(f"\n=== Built {len(stages)} stages ===")
for s in stages:
    print(f"  {s.name}: prompt_optimizer={s.prompt_optimizer}")

# ---------------------------------------------------------------------------
# 3. Attach optimizer via attach_optimizer()
# ---------------------------------------------------------------------------

optimized_stages = attach_optimizer(stages, opt)

print("\n=== After attach_optimizer ===")
for s in optimized_stages:
    print(f"  {s.name}: prompt_optimizer={s.prompt_optimizer}")
    assert s.prompt_optimizer is not None, f"Stage {s.name} should have optimizer"

# ---------------------------------------------------------------------------
# 4. Call optimizer.optimize() -- should return prompt unchanged
# ---------------------------------------------------------------------------

prompt = "Analyze the user request and produce a plan."
result = opt.optimize(prompt, task="build a chatbot", stage_name="planner")
print(f"\n=== optimize() ===")
print(f"  input:  {prompt!r}")
print(f"  output: {result!r}")
assert result == prompt, "NoOpOptimizer must return prompt unchanged"

# ---------------------------------------------------------------------------
# 5. Call optimizer.score()
# ---------------------------------------------------------------------------

score_ok = opt.score(prompt, task="build a chatbot", result="good plan", success=True)
score_fail = opt.score(prompt, task="build a chatbot", result="bad plan", success=False)
print(f"\n=== score() ===")
print(f"  success=True  -> {score_ok}")
print(f"  success=False -> {score_fail}")
assert score_ok == 1.0
assert score_fail == 0.0

# ---------------------------------------------------------------------------
# 6. Verify attached hook produces same output
# ---------------------------------------------------------------------------

hook_result = optimized_stages[0].prompt_optimizer(prompt)
print(f"\n=== Attached hook call ===")
print(f"  hook({prompt!r}) -> {hook_result!r}")
assert hook_result == prompt

print("\n--- all assertions passed ---")
