# Example 91: DeerFlow Skill Bridge

## Wiring Diagram

```
Markdown Skill                    Markdown Skill
  (frontmatter +                  (roundtrip export)
   numbered steps)                       ^
        |                                |
        v                         [template_to_skill]
  [parse_skill_frontmatter]              ^
        |                                |
        v                                |
  frontmatter dict               PatternTemplate
  (name, category, ...)          (name, topology,
        |                         stage_specs,
        |                         fingerprint)
  [extract_workflow_steps]               ^
        |                                |
        v                                |
  steps: list[str]               [skill_to_template]
                                         ^
                                         |
                                  Markdown Skill ──┘
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | parse_skill_frontmatter() | Extracts YAML frontmatter from Markdown skill |
| 2 | extract_workflow_steps() | Parses numbered steps from Markdown body |
| 3 | skill_to_template() | Full Markdown -> PatternTemplate conversion |
| 4 | template_to_skill() | PatternTemplate -> Markdown export |
| 5 | Roundtrip fidelity | skill -> template -> skill preserves stage count |
| 6 | Topology inference | Step count determines topology (5 steps -> "skill_organism") |

## Data Flow

```
Markdown Skill:
  ---
  name: competitive_analysis
  category: research
  ---
  1. Identify competitors
  2. Gather data
  3. Analyze strengths
  4. Produce matrix
  5. Write summary
       ↓
parse_skill_frontmatter()
  └─ {name: "competitive_analysis", category: "research", ...}
       ↓
extract_workflow_steps()
  └─ ["Identify competitors", "Gather data", ..., "Write summary"]
       ↓
skill_to_template(skill_md)
  ├─ name: "competitive_analysis"
  ├─ topology: "skill_organism"
  ├─ stage_specs: 5 stages (one per step)
  └─ fingerprint: task_shape="mixed", subtask_count=5
       ↓
template_to_skill(template)
  └─ Markdown with frontmatter + numbered steps
       ↓
skill_to_template(exported)  [roundtrip]
  └─ stage_specs count matches original (5 == 5)
```

## Roundtrip Verification

| Property | Original | Roundtrip | Preserved? |
|----------|----------|-----------|-----------|
| Stage count | 5 | 5 | Yes |
| Topology | skill_organism | skill_organism | Yes |
| Frontmatter | present | present | Yes |
| Task shape | mixed | mixed | Yes |
