# Example 84: CLI Organism

## Wiring Diagram

```mermaid
flowchart TB
    subgraph factory["cli_organism() Factory"]
        call["cli_organism(<br/>commands={generate, transform, count},<br/>input_mode='stdin',<br/>watcher=True,<br/>substrate=BiTemporalMemory())"]
    end

    subgraph pipeline["Auto-Wired CLI Pipeline"]
        task["'hello from the cli organism'"] --> generate
        generate["generate<br/>echo<br/>cli_handler"] -->|stdout via stdin| transform
        transform["transform<br/>tr a-z A-Z<br/>cli_handler"] -->|stdout via stdin| count
        count["count<br/>wc -c<br/>cli_handler"] --> output["Final Output"]
    end

    subgraph managed["Managed Stack (auto-wired)"]
        watcher["WatcherComponent<br/>(watcher=True)"] -.->|observe 3 stages| pipeline
        substrate["BiTemporalMemory<br/>(substrate)"] -.->|record facts| pipeline
    end

    subgraph results["Results"]
        output --> mr["ManagedResult"]
        mr --> rr["run_result:<br/>3 stage_results"]
        mr --> ws["watcher_summary:<br/>total_stages_observed=3"]
        mr --> facts["substrate facts<br/>recorded"]
    end

    call --> pipeline

    style call fill:#e0f7fa
    style output fill:#c8e6c9
    style generate fill:#fff9c4
    style transform fill:#fff9c4
    style count fill:#fff9c4
```

```
cli_organism(
    commands={
        "generate": "echo",
        "transform": ["tr", "a-z", "A-Z"],
        "count": "wc -c",
    },
    input_mode="stdin",
    watcher=True,
    substrate=BiTemporalMemory(),
)
  │
  │  One dict → full managed pipeline:
  │
  ├── [generate]  echo           → "hello from the cli organism"
  │       | (stdin)
  ├── [transform] tr a-z A-Z     → "HELLO FROM THE CLI ORGANISM"
  │       | (stdin)
  ├── [count]     wc -c          → character count
  │
  ├── [WatcherComponent]         → total_stages_observed: 3
  └── [BiTemporalMemory]         → facts recorded per stage
         │
         v
  ManagedResult
    ├── run_result.stage_results (3)
    ├── watcher_summary
    └── substrate facts
```

## Key Patterns

### Dict-to-Pipeline Factory
`cli_organism()` takes a simple dict of `{name: command}` and builds a complete
managed organism with watcher, substrate, and all v0.19-0.23 components. This
is the highest-level abstraction for CLI-backed pipelines.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | cli_organism() | Factory: dict of commands → full managed organism |
| 2 | cli_handler (auto-created) | Each command wrapped as a SkillStage handler |
| 3 | input_mode="stdin" | All stages receive previous output via stdin pipe |
| 4 | WatcherComponent | Auto-created when watcher=True, monitors all stages |
| 5 | BiTemporalMemory | Substrate records stage outputs as bi-temporal facts |
| 6 | ManagedOrganism | Returned object with run(), status(), consolidate() |

### Unix Pipeline as Organism
The command dict maps directly to a Unix pipeline analogy:
`echo | tr a-z A-Z | wc -c`, but with organism-level monitoring, fact recording,
and convergence detection wrapping each stage.

### Biological Parallel
Assembly-line organelle: like the endoplasmic reticulum where proteins pass
through sequential processing stations. Each station (CLI command) transforms
the input, and the cell (organism) monitors the entire assembly line.

## Data Flow

```
cli_organism() inputs
  ├─ commands: {"generate": "echo", "transform": ["tr","a-z","A-Z"], "count": "wc -c"}
  ├─ input_mode: "stdin"
  ├─ fast_nucleus: Nucleus (MockProvider)
  ├─ deep_nucleus: Nucleus (MockProvider)
  ├─ watcher: True
  └─ substrate: BiTemporalMemory()
       ↓
ManagedOrganism (m)
       ↓ m.run("hello from the cli organism")
ManagedResult
  ├─ run_result: RunResult
  │   ├─ stage_results[0]: generate → "hello from the cli organism"
  │   ├─ stage_results[1]: transform → "HELLO FROM THE CLI ORGANISM"
  │   └─ stage_results[2]: count → character count
  ├─ watcher_summary: {"total_stages_observed": 3, ...}
  └─ substrate._facts: recorded outputs
```

## Pipeline Stages

| Stage | Command | Input Mode | Input | Output |
|-------|---------|-----------|-------|--------|
| generate | `echo` | stdin | raw task string | echoed text |
| transform | `tr a-z A-Z` | stdin | echoed text | uppercased text |
| count | `wc -c` | stdin | uppercased text | character count |
