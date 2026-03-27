# Example 74: Adaptive Assembly

## Wiring Diagram

```mermaid
flowchart TB
    subgraph library["PatternLibrary"]
        T1[Sequential Review Pipeline<br/>topology=skill_organism]
        T2[Enterprise Analysis Organism<br/>topology=skill_organism]
        T3[Research Swarm<br/>topology=specialist_swarm]
    end

    subgraph assembly["Adaptive Assembly Loop"]
        FP[TaskFingerprint<br/>shape=sequential, tools=3<br/>roles=researcher,strategist<br/>tags=enterprise] --> SELECT[top_templates_for]
        T1 --> SELECT
        T2 --> SELECT
        T3 --> SELECT
        SELECT --> BEST[Best: Enterprise Analysis<br/>score=highest]
        BEST --> BUILD[adaptive_skill_organism<br/>build from template]
        BUILD --> ORG[SkillOrganism<br/>intake -> research -> strategy]
    end

    subgraph execution["Execution + Recording"]
        ORG --> RUN1[Run 1: Q4 earnings analysis]
        RUN1 --> RES1[Result + WatcherSummary]
        RES1 --> REC1[record_run<br/>success=true]
        REC1 --> LIB[(Library Updated)]

        ORG --> RUN2[Run 2: Q3 compliance report]
        RUN2 --> RES2[Result + WatcherSummary]
        RES2 --> REC2[record_run<br/>success=true]
        REC2 --> LIB
    end

    subgraph feedback["Feedback Loop"]
        LIB --> RERANK[Re-ranked Templates]
        RERANK --> SHIFT[Enterprise: 100% success<br/>score boosted]
    end

    style BEST fill:#c8e6c9
    style LIB fill:#e0f7fa
    style SHIFT fill:#fff9c4
```

```
Adaptive Assembly Loop:

  TaskFingerprint(sequential, tools=3, roles=[researcher, strategist], tags=[enterprise])
       |
       v
  PatternLibrary.top_templates_for() --> ranked:
       1. Enterprise Analysis Organism  (best match)
       2. Sequential Review Pipeline
       3. Research Swarm
       |
       v
  adaptive_skill_organism() builds organism from template:
       [intake: Normalizer] --> [research: Researcher(fuzzy)] --> [strategy: Strategist(deep)]
       + WatcherComponent (auto-attached)
       + handlers bound from user-provided dict
       |
       v
  Run 1 --> success --> record_run(template_id="enterprise", success=True)
  Run 2 --> success --> record_run(template_id="enterprise", success=True)
       |
       v
  Re-rank: Enterprise success_rate=100% --> score boosted in future selections
```

## Key Patterns

### Closed-Loop Adaptive Assembly
The full assembly cycle: fingerprint a task, select the best template from the
library, build an organism from it, execute with a watcher, record the outcome,
and let success rates influence future selections.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | TaskFingerprint | Structured task description for matching |
| 2 | PatternLibrary | Template registry with similarity ranking |
| 3 | adaptive_skill_organism() | Factory that selects template + builds organism |
| 4 | Template-to-organism mapping | stage_specs -> SkillStage instances with handlers |
| 5 | WatcherComponent (auto) | Automatically attached for monitoring |
| 6 | AdaptiveResult | Run result + record + watcher summary |
| 7 | record_run() (auto) | Outcome automatically fed back to library |
| 8 | Success rate feedback | Future rankings boosted by past success |

### Biological Analogy
Like the adaptive immune system's clonal selection: when a pathogen (task) arrives,
the system selects the best-matching B-cell clone (template), amplifies it
(builds organism), tests the response (execution), and records whether it
succeeded (memory cell formation). Successful clones are preferentially selected
for future challenges.

### Handler Binding
The user provides a dict mapping stage names to handler functions. The
adaptive_skill_organism() factory binds these to the stages from the selected
template, bridging the gap between abstract templates and concrete execution.

## Data Flow

```
adaptive_skill_organism() inputs:
  ├─ task: str
  ├─ fingerprint: TaskFingerprint
  ├─ library: PatternLibrary
  ├─ fast_nucleus: Nucleus
  ├─ deep_nucleus: Nucleus
  └─ handlers: dict[str, Callable]
       ↓
AdaptiveOrganism
  ├─ template: PatternTemplate (selected)
  ├─ template_score: float
  └─ organism: SkillOrganism (built from template)
       ↓
AdaptiveResult (from .run())
  ├─ run_result: OrganismResult
  │     ├─ stage_results: list[StageResult]
  │     └─ final_output: Any
  ├─ record: PatternRunRecord (auto-created)
  │     ├─ success: bool
  │     ├─ latency_ms: float
  │     └─ tokens_used: int
  └─ watcher_summary: dict
```

## Pipeline Stages (Enterprise Analysis Organism)

| Stage | Role | Mode | Handler | Input | Output |
|-------|------|------|---------|-------|--------|
| intake | Normalizer | deterministic | lambda task: {"parsed": task} | Raw task string | Parsed dict |
| research | Researcher | fuzzy | lambda with state/outputs | Parsed task | "Revenue up 12%..." |
| strategy | Strategist | deep | lambda with state/outputs | Research findings | "Recommend: hold..." |
