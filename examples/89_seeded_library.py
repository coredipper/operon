"""
Example 89 — Seeded Pattern Library
======================================

Seeds an Operon PatternLibrary with templates from Swarms, DeerFlow, and
the ACG survey, then demonstrates fingerprint-based ranking.

Usage:
    python examples/89_seeded_library.py
"""

from operon_ai import PatternLibrary
from operon_ai.convergence import (
    get_builtin_swarms_patterns,
    seed_library_from_acg_survey,
    seed_library_from_deerflow,
    seed_library_from_swarms,
)
from operon_ai.patterns.repository import TaskFingerprint

# ---------------------------------------------------------------------------
# 1. Create an empty library and seed it
# ---------------------------------------------------------------------------

library = PatternLibrary()

swarms_count = seed_library_from_swarms(library, get_builtin_swarms_patterns())
print(f"Seeded {swarms_count} Swarms templates")

deerflow_sessions = [
    {
        "assistant_id": "research_agent",
        "skills": ["web_search", "summarize"],
        "sub_agents": [
            {"name": "searcher", "role": "researcher", "skills": ["web_search"]},
            {"name": "writer", "role": "writer", "skills": ["summarize"]},
        ],
        "recursion_limit": 50,
        "sandbox": "docker",
    },
]
deerflow_count = seed_library_from_deerflow(library, deerflow_sessions)
print(f"Seeded {deerflow_count} DeerFlow templates")

acg_count = seed_library_from_acg_survey(library)
print(f"Seeded {acg_count} ACG survey templates")

total = swarms_count + deerflow_count + acg_count
print(f"\nTotal: {total} templates in library\n")

# ---------------------------------------------------------------------------
# 2. Rank templates by fingerprint
# ---------------------------------------------------------------------------

fp = TaskFingerprint(
    task_shape="parallel",
    tool_count=3,
    subtask_count=4,
    required_roles=("researcher", "developer", "reviewer"),
)

ranked = library.top_templates_for(fp, limit=5)
print("=== Top 5 templates for parallel research task ===")
for template, score in ranked:
    print(f"  {score:.3f}  {template.name} ({template.topology})")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert total >= 18  # 10 swarms + 1 deerflow + 8 acg
assert len(ranked) > 0
assert all(score >= 0.0 for _, score in ranked)
print("\n--- all assertions passed ---")
