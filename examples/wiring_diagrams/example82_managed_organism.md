# Example 82: Managed Organism

## Wiring Diagram

```mermaid
flowchart TB
    subgraph factory["managed_organism() Factory"]
        call["managed_organism(<br/>task, library, fingerprint,<br/>fast_nucleus, deep_nucleus,<br/>handlers, substrate, telomere,<br/>organism_id)"]
    end

    subgraph stack["Full v0.19-0.23 Stack (auto-wired)"]
        lib["PatternLibrary<br/>template: 'pipeline'<br/>1 success record"] --> matcher["Template Matching<br/>by TaskFingerprint"]
        matcher --> organism["SkillOrganism<br/>intake → process"]
        watcher["WatcherComponent<br/>(auto-created)"] -.->|observe| organism
        substrate["BiTemporalMemory<br/>(substrate)"] -.->|record facts| organism
        telomere["Telomere<br/>max_ops=100"] --> dev["DevelopmentController<br/>(auto-created)"]
        dev -.->|stage gating| organism
        sl["SocialLearning<br/>organism_id='org-A'"] -.->|export/import| lib
    end

    subgraph run["Execution"]
        organism --> result["ManagedResult"]
        result --> run_result["run_result:<br/>RunResult"]
        result --> tmpl["template_used:<br/>PatternTemplate"]
        result --> ws["watcher_summary:<br/>dict"]
        result --> ds["development_status:<br/>DevelopmentStatus"]
    end

    subgraph lifecycle["Lifecycle Operations"]
        consolidate["m.consolidate()"] --> cr["ConsolidationResult"]
        status["m.status()"] --> status_dict["Status Dict"]
        export["m.export_templates()"] --> exchange["TemplateExchange"]
    end

    call --> stack

    style call fill:#e0f7fa
    style result fill:#c8e6c9
    style organism fill:#fff9c4
```

```
managed_organism(
    task, library, fingerprint,
    fast_nucleus, deep_nucleus,
    handlers={intake: lambda, process: lambda},
    substrate=BiTemporalMemory(),
    telomere=Telomere(100),
    organism_id="org-A",
)
  │
  │  One function call wires the full stack:
  │
  ├── [PatternLibrary] ── template matching by fingerprint
  ├── [SkillOrganism] ── intake → process
  │       ├── handler: intake  → lambda task: {"parsed": task}
  │       └── handler: process → lambda: "Report processed successfully."
  ├── [WatcherComponent] ── monitors all stages
  ├── [BiTemporalMemory] ── substrate attached (use emit_output_fact or fact_extractor to record)
  ├── [Telomere] + [DevelopmentController] ── developmental gating
  └── [SocialLearning] ── export/import templates
         │
         v
  ManagedResult
    ├── run_result.final_output
    ├── template_used.name
    ├── watcher_summary
    └── development_status.stage
```

## Key Patterns

### Single-Call Full Stack Wiring
The `managed_organism()` factory replaces 5-7 manual component wirings with
a single function call. It auto-creates and connects the watcher, development
controller, social learning, and substrate.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | managed_organism() | Factory that wires the complete v0.19-0.23 stack |
| 2 | PatternLibrary | Template store with fingerprint-based matching |
| 3 | TaskFingerprint | Matches task shape to available templates |
| 4 | WatcherComponent | Auto-created, monitors all stages |
| 5 | BiTemporalMemory | Substrate records stage outputs as bi-temporal facts |
| 6 | DevelopmentController | Auto-created from Telomere, gates capabilities |
| 7 | SocialLearning | Enables template export/import between organisms |
| 8 | consolidate() | Triggers sleep consolidation on the managed stack |

### Biological Parallel
A fully-formed organism with all organ systems connected: nervous system
(watcher), immune system (substrate validation), endocrine system (development
controller), and social behavior (social learning). The factory acts like
embryonic development -- one genome (config) produces a complete organism.

## Data Flow

```
managed_organism() inputs
  ├─ task: str
  ├─ library: PatternLibrary
  ├─ fingerprint: TaskFingerprint("sequential", 2, 2, ...)
  ├─ fast_nucleus: Nucleus
  ├─ deep_nucleus: Nucleus
  ├─ handlers: dict[str, Callable]
  ├─ substrate: BiTemporalMemory
  ├─ telomere: Telomere(100)
  └─ organism_id: "org-A"
       ↓
ManagedOrganism (m)
       ↓ m.run(task)
ManagedResult
  ├─ run_result: RunResult
  │   ├─ final_output: str
  │   └─ stage_results: list[StageResult]
  ├─ template_used: PatternTemplate
  ├─ watcher_summary: dict
  └─ development_status: DevelopmentStatus
       ↓ m.consolidate()
ConsolidationResult
       ↓ m.export_templates()
TemplateExchange
```

## Pipeline Stages

| Stage | Handler | Input | Output | Managed By |
|-------|---------|-------|--------|-----------|
| intake | lambda (handlers dict) | raw task str | {"parsed": task} | Watcher + Substrate |
| process | lambda (handlers dict) | task + state + outputs | "Report processed successfully." | Watcher + Substrate |

## Lifecycle Operations

| Operation | Method | Returns | Purpose |
|-----------|--------|---------|---------|
| Run | m.run(task) | ManagedResult | Execute the pipeline |
| Consolidate | m.consolidate() | ConsolidationResult | Sleep consolidation cycle |
| Status | m.status() | dict | Full stack status snapshot |
| Export | m.export_templates() | TemplateExchange | Social learning export |
