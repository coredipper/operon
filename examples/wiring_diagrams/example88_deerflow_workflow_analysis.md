# Example 88: DeerFlow Workflow Analysis

## Wiring Diagram

```
session_config                       AdapterResult
  {assistant_id, skills,             (risk_score, warnings,
   sub_agents, sandbox}               topology_advice)
        |                                   ^
        v                                   |
  [parse_deerflow_session]       [analyze_external_topology]
        |                                   ^
        v                                   |
  ExternalTopology ─────────────────────────┘
    source="deerflow"
    metadata.sandbox="docker"
        |
        ├──────────────────┐
        v                  v
  [deerflow_skills_       [deerflow_to_template]
   to_stages]                    |
        |                        v
        v                  PatternTemplate
  SkillStages                topology, stage_specs
    (mode per category)
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | parse_deerflow_session() | Converts DeerFlow 2.0 session config to ExternalTopology |
| 2 | ExternalTopology | Lead agent + sub-agents as graph; sandbox in metadata |
| 3 | analyze_external_topology() | Epistemic risk scoring on the topology |
| 4 | deerflow_skills_to_stages() | Maps skill categories to cognitive modes |
| 5 | deerflow_to_template() | Full session-to-template conversion |
| 6 | Skill category mapping | research -> fixed; code -> fuzzy; verification -> fixed |

## Data Flow

```
session_config: {
  assistant_id: "lead_agent",
  skills: [...],
  sub_agents: [{name, role, skills}, ...],
  recursion_limit: 100,
  sandbox: "docker"
}
       ↓
parse_deerflow_session(session_config)
  ├─ agents: 4 (lead + 3 sub-agents)
  ├─ edges: lead → each sub-agent
  ├─ metadata: {sandbox: "docker", recursion_limit: 100}
  └─ source: "deerflow"
       ↓
analyze_external_topology(topology)
  ├─ risk_score: float
  ├─ warnings: list[str]
  └─ topology_advice.recommended_pattern: str
       ↓
deerflow_skills_to_stages(skill_dicts)
  ├─ web_research: mode=fixed  (research category)
  ├─ code_gen:     mode=fuzzy  (code category)
  └─ quality_check: mode=fixed (verification category)
       ↓
deerflow_to_template(session_config)
  ├─ topology: "specialist_swarm" | "skill_organism" | "single_worker"
  └─ stage_specs: list
```

## Skill Category Mapping

| DeerFlow Category | Operon Mode | Rationale |
|------------------|-------------|-----------|
| research | fixed | Observational, retrieval-based |
| code | fuzzy | Generative, action-oriented |
| verification | fixed | Deterministic quality checks |
