"""
Example 116: Atomic Skills Composition
========================================

Demonstrates composing Ma et al.'s five atomic coding skills
(localize, edit, test, reproduce, review) into multi-stage organisms.

1. Seed a PatternLibrary with the 5 atomic skills
2. Retrieve and inspect individual skill templates
3. Compose localize → edit → test as a serial pipeline
4. Show that review is naturally parallel (4 concurrent auditors)
5. Build a composed organism from the serial pipeline

The key insight: these 5 skills are composable basis vectors for
software engineering tasks. Any coding workflow can be expressed as
a composition of these atomic operations.

References:
  Ma et al. (arXiv:2604.05013) — atomic coding skills

Usage: python examples/116_atomic_skills_composition.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import MockProvider, Nucleus, PatternLibrary, SkillStage, skill_organism
from operon_ai.convergence import seed_library_from_atomic_skills, get_atomic_skill_patterns
from operon_ai.patterns.repository import TaskFingerprint


def main():
    print("=" * 60)
    print("Atomic Skills Composition")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Seed library with the 5 atomic coding skills
    # ------------------------------------------------------------------
    library = PatternLibrary()
    count = seed_library_from_atomic_skills(library)

    print(f"\n--- Atomic Skills ({count}) ---")
    patterns = get_atomic_skill_patterns()
    for p in patterns:
        print(f"  {p['name']:12s}  shape={p['task_shape']:10s}  "
              f"roles={p['roles']}")

    # ------------------------------------------------------------------
    # 2. Rank templates for a bug-fix task (serial, 2 tools)
    # ------------------------------------------------------------------
    print("\n--- Ranking: bug-fix task ---")
    bugfix_fp = TaskFingerprint(
        task_shape="sequential",
        tool_count=2,
        subtask_count=2,
        required_roles=("searcher", "editor"),
    )
    ranked = library.top_templates_for(bugfix_fp, limit=5)
    for template, score in ranked:
        print(f"  {score:.3f}  {template.name} ({template.topology})")

    # ------------------------------------------------------------------
    # 3. Rank templates for a code review task (parallel, 4 reviewers)
    # ------------------------------------------------------------------
    print("\n--- Ranking: code review task ---")
    review_fp = TaskFingerprint(
        task_shape="parallel",
        tool_count=4,
        subtask_count=4,
        required_roles=("logic_auditor", "security_auditor"),
    )
    ranked_review = library.top_templates_for(review_fp, limit=5)
    for template, score in ranked_review:
        print(f"  {score:.3f}  {template.name} ({template.topology})")

    # ------------------------------------------------------------------
    # 4. Compose: localize → edit → test as a serial organism
    # ------------------------------------------------------------------
    print("\n--- Composed Pipeline: localize → edit → test ---")

    # Build stages from the atomic skills' role definitions
    pipeline_skills = ["localize", "edit", "test"]
    stages = []
    for skill in patterns:
        if skill["name"] in pipeline_skills:
            for role in skill["roles"]:
                stages.append(SkillStage(
                    name=f"{skill['name']}_{role}",
                    role=role.replace("_", " ").title(),
                    instructions=f"{skill['description']} ({role} phase)",
                    mode="fixed",
                ))

    print(f"  total stages: {len(stages)}")
    for s in stages:
        print(f"    {s.name:24s} role={s.role}")

    # Build the organism
    fast = Nucleus(provider=MockProvider(responses={
        "find": "Found: auth/session.py:42",
        "rank": "Top match: auth/session.py:42 (confidence 0.95)",
        "modify": "Applied: add token refresh logic",
        "valid": "Syntax OK, no regressions",
        "analyz": "Test coverage: session timeout, token refresh",
        "generat": "Generated 3 test cases",
        "runner": "All 3 tests passed",
    }))
    deep = Nucleus(provider=MockProvider(responses={}))

    org = skill_organism(
        stages=stages,
        fast_nucleus=fast,
        deep_nucleus=deep,
    )

    result = org.run("Fix session timeout not refreshing auth token")

    print(f"\n  stages executed: {len(result.stage_results)}")
    for sr in result.stage_results:
        output_preview = str(sr.output)[:50]
        print(f"    {sr.stage_name}: {output_preview}")

    # ------------------------------------------------------------------
    # 5. Show review is parallel (4 independent auditors)
    # ------------------------------------------------------------------
    print("\n--- Review Skill (parallel topology) ---")
    review_skill = next(p for p in patterns if p["name"] == "review")
    print(f"  shape: {review_skill['task_shape']}")
    print(f"  roles: {review_skill['roles']}")
    print(f"  Note: 4 auditors can run concurrently — no data dependency")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    # 5 atomic skills seeded
    assert count == 5, f"Expected 5 atomic skills, got {count}"

    # Bug-fix task ranks localize and edit highest
    top_names = [t.name for t, _ in ranked[:3]]
    assert any("localize" in n for n in top_names), (
        f"localize should rank high for bug-fix, got {top_names}"
    )

    # Review task ranks review highest
    top_review = [t.name for t, _ in ranked_review[:2]]
    assert any("review" in n for n in top_review), (
        f"review should rank high for review task, got {top_review}"
    )

    # Composed pipeline has correct stage count
    # localize(2) + edit(2) + test(3) = 7 stages
    assert len(stages) == 7, f"Expected 7 stages, got {len(stages)}"
    assert len(result.stage_results) == 7

    # Review is parallel
    assert review_skill["task_shape"] == "parallel"

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
