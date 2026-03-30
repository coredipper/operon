# Example 97: Ralph Hat Analysis

## Wiring Diagram

```
Ralph config YAML                        AdapterResult
  (hats, transitions, events)           (risk_score, warnings, template)
        |                                       ^
        v                                       |
  [parse_ralph_config]                  [analyze_external_topology]
        |                                       ^
        v                                       |
  ExternalTopology ─────────────────────────────┘
    source="ralph"
    agents (one per hat), edges (transitions)
        |
        v
  ralph_hats_to_stages()
        |
        v
  list[StageSpec]
    name=hat_name, role=hat_role
```

## Hat-to-Stage Mapping

```
Ralph Config                           Operon Stages
┌──────────────────────┐              ┌──────────────────────┐
│ hat: "researcher"    │──────────────│ StageSpec("researcher",│
│   role: "Research"   │              │   role="Research")     │
│   transitions:       │              └──────────────────────┘
│     - analyst        │                       │
│     - writer         │                       │ edge
└──────────────────────┘                       v
┌──────────────────────┐              ┌──────────────────────┐
│ hat: "analyst"       │──────────────│ StageSpec("analyst",  │
│   role: "Analysis"   │              │   role="Analysis")    │
│   transitions:       │              └──────────────────────┘
│     - writer         │                       │
└──────────────────────┘                       │ edge
                                               v
┌──────────────────────┐              ┌──────────────────────┐
│ hat: "writer"        │──────────────│ StageSpec("writer",   │
│   role: "Writing"    │              │   role="Writing")     │
└──────────────────────┘              └──────────────────────┘
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | parse_ralph_config() | Converts Ralph YAML config into ExternalTopology |
| 2 | ExternalTopology | Framework-agnostic graph (hats as agents, transitions as edges) |
| 3 | analyze_external_topology() | Applies epistemic theorems to score structural risk |
| 4 | AdapterResult | Risk score, warnings, topology advice |
| 5 | ralph_hats_to_stages() | Maps hat definitions to Operon StageSpec list |

## Data Flow

```
dict: {hats: [{name, role, transitions, events}]}
       |
parse_ralph_config()
  ├─ 3-hat pipeline (researcher → analyst → writer)
  │    → ExternalTopology(source="ralph", agents=3, edges=2)
  └─ Event-driven hat with multiple transitions
       → ExternalTopology(source="ralph", agents=N, edges=M)
       |
analyze_external_topology(topology)
  ├─ risk_score: float
  ├─ warnings: list[str]
  └─ topology_advice.recommended_pattern: str
       |
ralph_hats_to_stages(config)
  └─ list[StageSpec] — one stage per hat, ordered by transition graph
```
