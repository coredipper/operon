"""
Example 90 — Hybrid Assembly
===============================

Demonstrates hybrid_skill_organism(): uses library templates when available,
falls back to generating a template when the library is empty.

Usage:
    python examples/90_hybrid_assembly.py
"""

from operon_ai import MockProvider, Nucleus, PatternLibrary
from operon_ai.convergence import (
    default_template_generator,
    get_builtin_swarms_patterns,
    hybrid_skill_organism,
    seed_library_from_swarms,
)
from operon_ai.patterns.repository import TaskFingerprint

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

# ---------------------------------------------------------------------------
# 1. Empty library → generator fallback
# ---------------------------------------------------------------------------

empty_lib = PatternLibrary()
fp = TaskFingerprint(
    task_shape="sequential",
    tool_count=0,
    subtask_count=3,
    required_roles=("planner", "executor", "reviewer"),
)

print("=== Empty library → generator fallback ===")
organism = hybrid_skill_organism(
    "Write a Python function that validates email addresses",
    library=empty_lib,
    fingerprint=fp,
    fast_nucleus=fast,
    deep_nucleus=deep,
    template_generator=default_template_generator,
)
print(f"  Organism type: {type(organism).__name__}")
print(f"  Library now has: {len(empty_lib.retrieve_templates())} templates")
print()

# ---------------------------------------------------------------------------
# 2. Seeded library → adaptive path
# ---------------------------------------------------------------------------

seeded_lib = PatternLibrary()
seed_library_from_swarms(seeded_lib, get_builtin_swarms_patterns())

print("=== Seeded library → adaptive path ===")
organism2 = hybrid_skill_organism(
    "Analyze vendor security posture",
    library=seeded_lib,
    fingerprint=fp,
    fast_nucleus=fast,
    deep_nucleus=deep,
    score_threshold=0.1,  # low threshold to ensure library match
)
print(f"  Organism type: {type(organism2).__name__}")
print()

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert len(empty_lib.retrieve_templates()) >= 1  # generator registered its template
assert type(organism).__name__ == "ManagedOrganism"
assert type(organism2).__name__ == "AdaptiveSkillOrganism"
print("--- all assertions passed ---")
