# Example 89: Seeded Pattern Library

## Wiring Diagram

```
  [Swarms builtins]       [DeerFlow sessions]       [ACG survey]
    10+ patterns             session configs          8+ archetypes
        |                         |                        |
        v                         v                        v
  [seed_library_           [seed_library_           [seed_library_
   from_swarms]             from_deerflow]           from_acg_survey]
        |                         |                        |
        └─────────────┬───────────┘────────────────────────┘
                      v
              PatternLibrary
              (18+ templates)
                      |
                      v
            [top_templates_for]
              fingerprint query
                      |
                      v
            Ranked Results
              [(template, score), ...]
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | seed_library_from_swarms() | Bulk-loads Swarms builtin patterns |
| 2 | get_builtin_swarms_patterns() | Returns 10+ canonical Swarms topologies |
| 3 | seed_library_from_deerflow() | Converts DeerFlow session configs to templates |
| 4 | seed_library_from_acg_survey() | Loads academic ACG archetypes |
| 5 | PatternLibrary | Unified store for all seeded templates |
| 6 | top_templates_for() | Fingerprint-based ranking of best-match templates |
| 7 | TaskFingerprint | Query key: shape, tool_count, subtask_count, roles |

## Data Flow

```
Sources:
  ├─ get_builtin_swarms_patterns() → 10+ pattern dicts
  ├─ DeerFlow session configs      → 1+ session dicts
  └─ ACG survey (builtin)          → 8+ archetype dicts
       ↓
seed_library_from_*() functions
  └─ each converts source format → PatternTemplate → library.add()
       ↓
PatternLibrary (18+ templates total)
       ↓
TaskFingerprint(
  task_shape="parallel",
  tool_count=3,
  subtask_count=4,
  required_roles=("researcher", "developer", "reviewer")
)
       ↓
library.top_templates_for(fp, limit=5)
  └─ [(template_1, 0.85), (template_2, 0.72), ...]
       ranked by fingerprint similarity score
```

## Seed Sources

| Source | Count | Origin | Templates Cover |
|--------|-------|--------|----------------|
| Swarms builtins | 10+ | get_builtin_swarms_patterns() | Sequential, hierarchical, concurrent |
| DeerFlow | 1+ per session | User-provided session configs | Research, coding, review workflows |
| ACG survey | 8+ | Academic archetypes | Debate, voting, pipeline, mixture |
