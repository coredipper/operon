# Example 98: A-Evolve Workspace Analysis

## Wiring Diagram

```
A-Evolve manifest                        AdapterResult
  (workspace, skills, score)            (risk_score, warnings, template)
        |                                       ^
        v                                       |
  [parse_aevolve_workspace]             [analyze_external_topology]
        |                                       ^
        v                                       |
  ExternalTopology ─────────────────────────────┘
    source="aevolve"
    agents=[workspace_node], edges=[]
        |
        v
  aevolve_skills_to_stages()
        |
        v
  list[StageSpec]
    name=skill_name, role="evolved"
```

## Evolution Loop Mapping

```
A-Evolve Loop                          Operon Analysis
┌─────────────────┐
│  Solve          │   workspace snapshot
│  (run task)     │──────────┐
└─────────────────┘          │
        │                    v
        v            ┌──────────────────────┐
┌─────────────────┐  │ parse_aevolve_        │
│  Observe        │  │   workspace()         │
│  (score result) │  │                       │
└─────────────────┘  │ ExternalTopology      │
        │            │   source="aevolve"    │
        v            │   single agent node   │
┌─────────────────┐  │   skills as metadata  │
│  Evolve         │  └──────────────────────┘
│  (mutate code)  │          │
└─────────────────┘          v
        │            ┌──────────────────────┐
        v            │ analyze_external_    │
┌─────────────────┐  │   topology()         │
│  Gate           │  │                       │
│  (accept/reject)│  │ AdapterResult         │
└─────────────────┘  │   risk_score, warnings│
        │            └──────────────────────┘
        v                    │
┌─────────────────┐          v
│  Reload         │  ┌──────────────────────┐
│  (deploy new)   │  │ aevolve_skills_to_   │
└─────────────────┘  │   stages()           │
                     │                       │
                     │ list[StageSpec]        │
                     │   one per skill        │
                     └──────────────────────┘
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | parse_aevolve_workspace() | Converts A-Evolve manifest into ExternalTopology |
| 2 | ExternalTopology | Single-node graph representing the evolved workspace |
| 3 | analyze_external_topology() | Applies epistemic theorems to score structural risk |
| 4 | AdapterResult | Risk score, warnings, topology advice |
| 5 | aevolve_skills_to_stages() | Maps evolved skills to Operon StageSpec list |
| 6 | EvolutionGating.tla | TLA+ spec for the Gate safety properties |

## Data Flow

```
dict: {workspace_id, skills: [{name, score}], version, benchmark_score}
       |
parse_aevolve_workspace()
  └─ ExternalTopology(source="aevolve", agents=1, edges=0)
     (single node — evolution acts on one workspace at a time)
       |
analyze_external_topology(topology)
  ├─ risk_score: float (typically low — single node, no handoff risk)
  ├─ warnings: list[str]
  └─ topology_advice.recommended_pattern: str
       |
aevolve_skills_to_stages(manifest)
  └─ list[StageSpec] — one stage per evolved skill
```
