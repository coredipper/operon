# Example 86: Swarms Topology Analysis

## Wiring Diagram

```
Swarms pattern dict                      AdapterResult
  (agent_specs, edges)                   (risk_score, warnings, template)
        |                                       ^
        v                                       |
  [parse_swarm_topology]               [analyze_external_topology]
        |                                       ^
        v                                       |
  ExternalTopology ─────────────────────────────┘
    source="swarms"
    pattern_name, agents, edges
        |
        v
  [swarm_to_template]
        |
        v
  PatternTemplate
    topology, stage_specs, fingerprint
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | parse_swarm_topology() | Converts plain dict config into ExternalTopology |
| 2 | ExternalTopology | Framework-agnostic graph (agents + edges + metadata) |
| 3 | analyze_external_topology() | Applies epistemic theorems to score structural risk |
| 4 | AdapterResult | Risk score, warnings, topology advice |
| 5 | swarm_to_template() | Converts topology to PatternTemplate for library use |
| 6 | TaskFingerprint | Shape/role metadata on generated template |

## Data Flow

```
dict: {pattern_name, agent_specs, edges}
       ↓
parse_swarm_topology()
  ├─ Sequential (2 agents, 1 edge)    → ExternalTopology(source="swarms")
  ├─ Deep Chain (8 agents, 7 edges)   → ExternalTopology(source="swarms")
  └─ Hierarchical (4 agents, 3 edges) → ExternalTopology(source="swarms")
       ↓
analyze_external_topology(topology)
  ├─ risk_score: float (0.0 = safe, 1.0 = dangerous)
  ├─ warnings: list[str]  (e.g. deep-chain error amplification)
  └─ topology_advice.recommended_pattern: str
       ↓
swarm_to_template(topology)
  ├─ template_id: str
  ├─ topology: "specialist_swarm" | "skill_organism" | ...
  ├─ stage_specs: list (one per agent)
  └─ fingerprint: TaskFingerprint(task_shape, subtask_count)
```

## Risk Scoring

| Topology | Agents | Risk | Why |
|----------|--------|------|-----|
| Sequential (2) | 2 | Low | Short chain, minimal amplification |
| Deep Chain (8) | 8 | High | 7-hop chain amplifies errors |
| Hierarchical | 4 | Moderate | Fan-out from manager limits depth |
