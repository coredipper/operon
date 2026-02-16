# Example 51: Epiplexity Healing Cascade

## Wiring Diagram

```mermaid
flowchart TB
    subgraph monitor["Epiplexity Monitor"]
        output[Agent Output] --> EM[EpiplexityMonitor.measure]
        EM --> status{HealthStatus}
    end

    subgraph cascade["Escalating Interventions"]
        status -->|HEALTHY/EXPLORING| continue[Continue Normally]
        status -->|STAGNANT| stage1["Stage 1: Autophagy<br/>Prune stale context"]
        stage1 --> recheck1{Still stuck?}
        recheck1 -->|no| continue
        recheck1 -->|yes| stage2["Stage 2: Regeneration<br/>Kill worker, spawn fresh"]
        stage2 --> recheck2{Still stuck?}
        recheck2 -->|no| continue
        recheck2 -->|yes| stage3["Stage 3: Abort<br/>Diagnostic report"]
    end

    subgraph healing["Healing Resources"]
        APD[AutophagyDaemon] --> stage1
        LYSO[Lysosome] --> APD
        RS[RegenerativeSwarm] --> stage2
        HS[HistoneStore] --> stage2
    end

    style continue fill:#c8e6c9
    style stage1 fill:#fff9c4
    style stage2 fill:#ffccbc
    style stage3 fill:#ffcdd2
```

```
[Agent Output] --text--> [EpiplexityMonitor]
                              |
           +--HEALTHY-------> [continue]
           |
           +--STAGNANT------> [AutophagyDaemon] --prune--> [Lysosome]
           |                       |
           |                  still stuck?
           |                       |
           +--CRITICAL-------> [RegenerativeSwarm]
           |                       ├── kill worker
           |                       ├── summary -> [HistoneStore]
           |                       └── spawn fresh worker
           |                       |
           |                  still stuck?
           |                       |
           +--ABORT----------> [Diagnostic Report]
```

## Key Patterns

### Escalating Interventions
The cascade mirrors biological immune response escalation:
1. **Mild** (Autophagy): Prune stale context — minimal disruption
2. **Moderate** (Regeneration): Kill and replace worker — fresh start with summary
3. **Severe** (Abort): Give up with full diagnostic data

### Epiplexity as Health Signal
Low Bayesian surprise (embedding novelty) combined with high perplexity indicates
the agent is in a pathological loop — not converging to a solution but repeating
itself with slight variations.

## Data Flow

```
EpiplexityMonitor
  ├─ measure(text) -> EpiplexityResult
  │     ├─ status: HealthStatus
  │     ├─ epiplexity: float
  │     └─ trend: str
       ↓
StagnationDetector
  ├─ consecutive_stagnant: int
  ├─ intervention_history: list[str]
  └─ should_escalate() -> InterventionLevel
       ↓
HealingCascade
  ├─ stage_1_autophagy() -> bool (resolved?)
  ├─ stage_2_regeneration() -> bool
  └─ stage_3_abort() -> DiagnosticReport
```

## Health Status Thresholds

| Status | Epiplexity | Duration | Action |
|--------|-----------|----------|--------|
| HEALTHY | Low | — | Continue |
| EXPLORING | Moderate | — | Continue |
| STAGNANT | High | Short | Autophagy |
| CRITICAL | High | Sustained | Regeneration |
| — | High | Extended | Abort |
