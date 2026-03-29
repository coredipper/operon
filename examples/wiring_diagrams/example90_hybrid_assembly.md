# Example 90: Hybrid Assembly

## Wiring Diagram

```
task + fingerprint
        |
        v
  [hybrid_skill_organism]
        |
        v
  ┌─ library.top_templates_for(fp) ─┐
  │                                  │
  │  score >= threshold?             │
  │                                  │
  ├── YES ──────────┐   ┌── NO ─────┤
  │                 v   v            │
  │     [AdaptiveSkill   [default_template   │
  │      Organism]        _generator]        │
  │     (library match)       |              │
  │                           v              │
  │                    [ManagedOrganism]      │
  │                    (generated template    │
  │                     registered in lib)    │
  └──────────────────────────────────────────┘
        |
        v
   Organism (ready to run)
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | hybrid_skill_organism() | Single entry point; picks library or generator path |
| 2 | PatternLibrary | Checked first for matching templates |
| 3 | TaskFingerprint | Query key for library lookup |
| 4 | score_threshold | Minimum similarity score to use library match |
| 5 | default_template_generator | Fallback: creates template from fingerprint |
| 6 | AdaptiveSkillOrganism | Returned when library has a good match |
| 7 | ManagedOrganism | Returned when generator fallback is used |

## Data Flow

```
Inputs:
  ├─ task: str ("Write a Python function...")
  ├─ library: PatternLibrary (empty or seeded)
  ├─ fingerprint: TaskFingerprint(sequential, 0, 3, roles)
  ├─ fast_nucleus: Nucleus
  ├─ deep_nucleus: Nucleus
  └─ template_generator: Callable (optional)
       ↓
hybrid_skill_organism()
       ↓
  [library check] → top_templates_for(fp)
       ↓
  ┌─ score >= threshold ─────────────────────┐
  │  → AdaptiveSkillOrganism                 │
  │    (uses best-match template)            │
  │                                          │
  ├─ score < threshold or library empty ─────┤
  │  → default_template_generator(fp)        │
  │    → new PatternTemplate                 │
  │    → registered into library             │
  │    → ManagedOrganism                     │
  └──────────────────────────────────────────┘
```

## Path Selection

| Condition | Path | Returns | Side Effect |
|-----------|------|---------|------------|
| Library has match above threshold | Adaptive | AdaptiveSkillOrganism | None |
| Library empty or no match | Generator fallback | ManagedOrganism | Template added to library |
