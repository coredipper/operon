# Example 87: AnimaWorks Role Mapping

## Wiring Diagram

```
org_config                         SkillStages
  {supervisor, agents,              (name, mode, role)
   communication}                        ^
        |                                |
        v                                |
  [parse_animaworks_org]     [animaworks_roles_to_stages]
        |                                ^
        v                                |
  ExternalTopology                  roles list
    source="animaworks"         (from org_config)
    agents, edges
        |
        v
  [animaworks_to_template]
        |
        v
  PatternTemplate
    topology, stage_specs, tags
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | parse_animaworks_org() | Builds ExternalTopology from supervisor hierarchy |
| 2 | ExternalTopology | Graph: supervisor -> each agent as edges |
| 3 | animaworks_roles_to_stages() | Maps roles to SkillStages with cognitive modes |
| 4 | SkillStage | Named stage with mode (fixed/fuzzy) based on role |
| 5 | animaworks_to_template() | Full conversion to PatternTemplate with tags |
| 6 | Cognitive mode mapping | reviewer/analyst/auditor -> fixed; engineer/writer/manager -> fuzzy |

## Data Flow

```
org_config: {
  supervisor: {name, role},
  agents: [{name, role, skills}, ...],
  communication: "hierarchical"
}
       ↓
parse_animaworks_org(org_config)
  ├─ agents: 5 (4 agents + supervisor)
  ├─ edges: 4 (supervisor → each agent)
  └─ source: "animaworks"
       ↓
animaworks_roles_to_stages(roles)
  ├─ backend_dev:  mode=fuzzy  (engineer — action-oriented)
  ├─ frontend_dev: mode=fuzzy  (engineer)
  ├─ qa_analyst:   mode=fixed  (reviewer — observational)
  ├─ tech_writer:  mode=fuzzy  (writer — action-oriented)
  └─ tech_lead:    mode=fuzzy  (manager)
       ↓
animaworks_to_template(org_config)
  ├─ topology: "specialist_swarm" | "skill_organism"
  ├─ stage_specs: list (one per agent)
  └─ tags: list[str]
```

## Role-to-Mode Mapping

| AnimaWorks Role | Operon Mode | Rationale |
|----------------|-------------|-----------|
| engineer | fuzzy | Action-oriented, generative work |
| manager | fuzzy | Decision-making under uncertainty |
| reviewer | fixed | Observational, deterministic checks |
| writer | fuzzy | Generative, creative output |
