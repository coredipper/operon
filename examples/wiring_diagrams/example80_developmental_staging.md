# Example 80: Developmental Staging

## Wiring Diagram

```mermaid
flowchart LR
    subgraph lifecycle["Developmental Lifecycle"]
        embryonic["EMBRYONIC<br/>tick 0"] -->|"10% lifespan"| juvenile["JUVENILE<br/>~tick 10"]
        juvenile -->|"35% lifespan"| adolescent["ADOLESCENT<br/>~tick 35"]
        adolescent -->|"70% lifespan"| mature["MATURE<br/>~tick 70"]
    end

    subgraph controller["DevelopmentController"]
        telomere["Telomere<br/>max_operations=100"] --> dc["DevelopmentController"]
        config["DevelopmentConfig<br/>juv=0.10, adol=0.35, mat=0.70"] --> dc
        dc --> stage["current stage"]
        dc --> plasticity["learning_plasticity"]
    end

    subgraph periods["Critical Periods"]
        cp1["rapid_learning<br/>EMBRYONIC → JUVENILE"] -->|closes at JUVENILE| closed1["CLOSED"]
        cp2["tool_exploration<br/>JUVENILE → ADOLESCENT"] -->|closes at ADOLESCENT| closed2["CLOSED"]
    end

    subgraph gating["Capability Gating"]
        dc --> gate{"can_acquire_stage?"}
        gate -->|"current >= required"| allow["ALLOW"]
        gate -->|"current < required"| deny["DENY"]
    end

    style embryonic fill:#e8f5e9
    style juvenile fill:#fff9c4
    style adolescent fill:#ffe0b2
    style mature fill:#c8e6c9
    style closed1 fill:#ffcdd2
    style closed2 fill:#ffcdd2
    style allow fill:#c8e6c9
    style deny fill:#ffcdd2
```

```
  Telomere (max_operations=100)
       |
       v
  DevelopmentController
       |
       +─── tick() ──→  stage transitions:
       |                   EMBRYONIC ──(10%)──→ JUVENILE
       |                   JUVENILE ──(35%)──→ ADOLESCENT
       |                   ADOLESCENT ─(70%)──→ MATURE
       |
       +─── critical_periods:
       |      "rapid_learning"    open: EMBRYONIC..JUVENILE    → CLOSED after JUVENILE
       |      "tool_exploration"  open: JUVENILE..ADOLESCENT   → CLOSED after ADOLESCENT
       |
       +─── capability_gating:
       |      can_acquire_stage(EMBRYONIC) → True  (always)
       |      can_acquire_stage(MATURE)    → True  (only when stage >= MATURE)
       |
       +─── learning_plasticity:  decreases as organism matures
```

## Key Patterns

### Telomere-Driven Maturation
The Telomere tracks total operations as a lifespan counter. The
DevelopmentController reads the telomere's consumption ratio and triggers stage
transitions at configured thresholds.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | Telomere | Lifespan counter (max_operations=100) |
| 2 | DevelopmentController | Manages stage transitions based on telomere ratio |
| 3 | DevelopmentConfig | Thresholds for juvenile (0.10), adolescent (0.35), mature (0.70) |
| 4 | CriticalPeriod | Time-bounded windows that close permanently |
| 5 | DevelopmentalStage enum | EMBRYONIC, JUVENILE, ADOLESCENT, MATURE |
| 6 | Capability gating | Restricts tool/template access by maturity level |

### Biological Parallel
- **Telomere**: Biological telomeres shorten with each cell division, acting as a lifespan clock
- **Critical Periods**: In neuroscience, critical periods (e.g., language acquisition) close permanently after a developmental window
- **Plasticity decay**: Young brains have high synaptic plasticity that decreases with age
- **Capability gating**: Developmental milestones must be reached before certain abilities emerge

## Data Flow

```
Telomere(max_operations=100)
  └─ consumption_ratio: float (0.0 → 1.0)
       ↓
DevelopmentController
  ├─ stage: DevelopmentalStage
  ├─ learning_plasticity: float
  ├─ transitions: list[Transition]
  └─ critical_periods: tuple[CriticalPeriod]
       ↓
DevelopmentStatus
  ├─ stage: DevelopmentalStage
  ├─ transitions: list
  └─ closed_periods: list[str]
```

## Stage Transitions

| Threshold | Stage | Plasticity | Critical Periods Open |
|-----------|-------|------------|----------------------|
| 0.00 - 0.10 | EMBRYONIC | High | rapid_learning, tool_exploration |
| 0.10 - 0.35 | JUVENILE | Medium-High | tool_exploration |
| 0.35 - 0.70 | ADOLESCENT | Medium | None |
| 0.70 - 1.00 | MATURE | Low | None |
