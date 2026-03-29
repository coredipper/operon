# Convergence

Operon's convergence package bridges external agent orchestration systems into Operon's structural analysis layer. Currently supports [Swarms](https://github.com/kyegomez/swarms), [DeerFlow](https://github.com/bytedance/deer-flow), and [AnimaWorks](https://github.com/AnimaWorks/AnimaWorks).

## Four-Layer Architecture

```
┌─────────────────────────────────────────────┐
│  AnimaWorks (cognitive layer)               │
├─────────────────────────────────────────────┤
│  AsyncThink (thinking layer)                │
├─────────────────────────────────────────────┤
│  DeerFlow / Swarms (orchestration layer)    │
├─────────────────────────────────────────────┤
│  Operon (structural layer)                  │
└─────────────────────────────────────────────┘
```

## Adapters (Phase C1)

Type-level bridges to [Swarms](https://github.com/kyegomez/swarms), [DeerFlow](https://github.com/bytedance/deer-flow), and [AnimaWorks](https://github.com/AnimaWorks/AnimaWorks). All adapters produce `ExternalTopology`, which `analyze_external_topology()` consumes to apply Operon's four epistemic theorems as a structural linter.

- `parse_swarm_topology()` — Swarms workflow patterns
- `parse_animaworks_org()` — AnimaWorks org hierarchies
- `parse_deerflow_session()` — DeerFlow session configs

## Template Exchange (Phase C2)

Seed Operon's PatternLibrary from external catalogs:

- `seed_library_from_swarms()` — 10 built-in Swarms patterns
- `seed_library_from_deerflow()` — DeerFlow sessions
- `seed_library_from_acg_survey()` — 8 ACG survey method categories
- `skill_to_template()` / `template_to_skill()` — bidirectional DeerFlow bridge
- `hybrid_skill_organism()` — library-first + generator fallback

## Memory + Thinking (Phase C3)

- `PrimingView` — multi-channel SubstrateView with trust, telemetry, experience
- `bridge_animaworks_memory()` / `bridge_deerflow_memory()` — import into BiTemporalMemory
- `HeartbeatDaemon` — idle-time consolidation
- `AsyncOrganizer` — Fork/Join execution within stages

## Formal Verification (Phase C4)

- 3 TLA+ specs: TemplateExchangeProtocol, DevelopmentalGating, ConvergenceDetection
- `DesignProblem` + `compose_series/parallel` + `feedback_fixed_point` — Zardini co-design

## Examples

- [86–88](https://github.com/coredipper/operon/blob/main/examples/): Adapter demos
- [89–91](https://github.com/coredipper/operon/blob/main/examples/): Template exchange
- [92–95](https://github.com/coredipper/operon/blob/main/examples/): Memory, PrimingView, heartbeat, async
- [96](https://github.com/coredipper/operon/blob/main/examples/96_codesign_composition.py): Co-design composition
