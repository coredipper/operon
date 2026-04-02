# Convergence

Operon's convergence package bridges external agent orchestration systems into Operon's structural analysis layer. Currently supports [Swarms](https://github.com/kyegomez/swarms), [DeerFlow](https://github.com/bytedance/deer-flow), [AnimaWorks](https://github.com/AnimaWorks/AnimaWorks), [Ralph](https://github.com/mikeyobrien/ralph-orchestrator), and [A-Evolve](https://github.com/A-EVO-Lab/a-evolve).

## Five-Layer Architecture

```
┌─────────────────────────────────────────────┐
│  A-Evolve (evolution layer)                 │
├─────────────────────────────────────────────┤
│  AnimaWorks (cognitive layer)               │
├─────────────────────────────────────────────┤
│  AsyncThink (thinking layer)                │
├─────────────────────────────────────────────┤
│  Ralph / DeerFlow / Swarms (orchestration)  │
├─────────────────────────────────────────────┤
│  Operon (structural layer)                  │
└─────────────────────────────────────────────┘
```

## Adapters (Phase C1)

Type-level bridges to [Swarms](https://github.com/kyegomez/swarms), [DeerFlow](https://github.com/bytedance/deer-flow), [AnimaWorks](https://github.com/AnimaWorks/AnimaWorks), [Ralph](https://github.com/mikeyobrien/ralph-orchestrator), and [A-Evolve](https://github.com/A-EVO-Lab/a-evolve). All adapters produce `ExternalTopology`, which `analyze_external_topology()` consumes to apply Operon's four epistemic theorems as a structural linter.

- `parse_swarm_topology()` — Swarms workflow patterns
- `parse_animaworks_org()` — AnimaWorks org hierarchies
- `parse_deerflow_session()` — DeerFlow session configs

### Phase C1 (Extended) — Ralph + A-Evolve Adapters

- `parse_ralph_config()` — [Ralph](https://github.com/mikeyobrien/ralph-orchestrator) hat definitions to ExternalTopology
- `ralph_hats_to_stages()` — maps Ralph hats to Operon StageSpec list
- `parse_aevolve_workspace()` — [A-Evolve](https://github.com/A-EVO-Lab/a-evolve) workspace manifests to ExternalTopology
- `aevolve_skills_to_stages()` — maps evolved skills to Operon StageSpec list

## Template Exchange (Phase C2)

Seed Operon's PatternLibrary from external catalogs:

- `seed_library_from_swarms()` — 10 built-in Swarms patterns
- `seed_library_from_deerflow()` — DeerFlow sessions
- `seed_library_from_acg_survey()` — 8 ACG survey method categories
- `seed_library_from_ralph()` — Ralph hat-based patterns
- `seed_library_from_aevolve()` — A-Evolve evolved workspace patterns
- `skill_to_template()` / `template_to_skill()` — bidirectional DeerFlow bridge
- `hybrid_skill_organism()` — library-first + generator fallback

## Memory + Thinking (Phase C3)

- `PrimingView` — multi-channel SubstrateView with trust, telemetry, experience
- `bridge_animaworks_memory()` / `bridge_deerflow_memory()` — import into BiTemporalMemory
- `HeartbeatDaemon` — idle-time consolidation
- `AsyncOrganizer` — Fork/Join execution within stages

## Formal Verification (Phase C4)

- 4 TLA+ specs: TemplateExchangeProtocol, DevelopmentalGating, ConvergenceDetection, EvolutionGating
- `EvolutionGating.tla` — models the A-Evolve Solve->Observe->Evolve->Gate->Reload loop with monotonic score safety
- `DesignProblem` + `compose_series/parallel` + `feedback_fixed_point` — Zardini co-design

## Production Runtime (Phase C5)

- 4 compilers (`organism_to_swarms`, `organism_to_deerflow`, `organism_to_ralph`, `organism_to_scion`) producing serializable dicts for each deployment target
- `DistributedWatcher` with transport abstraction (`InMemoryTransport` for single-process, `HttpTransport` as a webhook payload stub — callers send the queued requests via their own HTTP client)
- `operon_watcher_node()` — LangGraph-compatible node for DeerFlow integration, with `create_watcher_config()` helper
- [Scion](https://github.com/GoogleCloudPlatform/scion) compiler supports container isolation via `runtime="docker"` parameter

## Evaluation Harness (Phase C6)

- 20 benchmark tasks across 7 configurations (single, pipeline, fan-out, fan-in, diamond, full, stress)
- `MockEvaluator` using real structural analysis (topology advice, epistemic bounds, pattern scoring)
- Structural variation analysis: measures how topology metrics change across task configurations
- Credit assignment: attributes evaluation outcomes to individual stage contributions

## Prompt Optimization + Workflow Generation (Phase C7)

- `PromptOptimizer` protocol with `NoOpOptimizer` reference implementation; `EvolutionaryOptimizer` extended protocol
- `attach_optimizer` — attach optimizer to any SkillStage for prompt-level tuning
- `WorkflowGenerator` protocol with `HeuristicGenerator` reference implementation; `ReasoningGenerator` extended protocol
- `generate_and_register` — generate workflow topology and register it in PatternLibrary

## Harness Search + Live Evaluation (Phase C8) — Planned

Integrates [Meta-Harness](https://arxiv.org/abs/2603.28052) (Lee et al., Stanford/MIT 2026) insights into the convergence stack. Meta-Harness shows that searching over full agent harness code with filesystem access to all prior candidates, traces, and scores dramatically outperforms compressed-feedback optimizers (AlphaEvolve, OPRO, TextGrad).

### Harness Search

- `FilesystemOptimizer` — extends `EvolutionaryOptimizer` protocol with filesystem-backed trace store (full source + execution traces + scores per candidate, not compressed feedback)
- `HarnessSearchDP` — wraps Meta-Harness outer loop as a Zardini `DesignProblem`, composable with existing adapter DPs via `compose_series/parallel`
- `seed_library_from_meta_harness()` — decompose discovered harness programs into `PatternTemplate` entries for cross-framework reuse
- Pareto convergence criterion — extend `ConvergenceDetection.tla` with frontier stabilization check (no new non-dominated candidates for N iterations)
- Causal diagnosis safety property — extend `EvolutionGating.tla` to require causal attribution before accepting post-regression proposals

### Live Evaluation

- `LiveEvaluator` — runs real LLM calls through SkillOrganism pipelines (API providers + CLI agents)
- CLI provider support: `claude -p` and `codex exec` via `cli_handler()` as single-shot evaluation backends
- LLM-as-judge quality scoring with cross-provider comparison
- Guided vs unguided measurement across Gemini API, Claude CLI, Codex CLI

### Key Finding from C6 Live Evaluation

Structural guidance gives a consistent +6.2% quality improvement for multi-stage pipelines but has no effect on single-agent CLI execution — confirming that guidance helps when there is topology to guide, and the compile→decompile round-trip reveals where structural information gets lost.

## Examples

- [86–88](https://github.com/coredipper/operon/blob/main/examples/): Adapter demos
- [89–91](https://github.com/coredipper/operon/blob/main/examples/): Template exchange
- [92–95](https://github.com/coredipper/operon/blob/main/examples/): Memory, PrimingView, heartbeat, async
- [96](https://github.com/coredipper/operon/blob/main/examples/96_codesign_composition.py): Co-design composition
- [97–98](https://github.com/coredipper/operon/blob/main/examples/): Ralph hat analysis, A-Evolve workspace analysis
- [99–103](https://github.com/coredipper/operon/blob/main/examples/): Production runtime — 4 compilers and distributed watcher
- [104](https://github.com/coredipper/operon/blob/main/examples/104_evaluation_harness.py): Evaluation harness with MockEvaluator
- [105](https://github.com/coredipper/operon/blob/main/examples/105_prompt_optimization_interface.py): Prompt optimization protocols
- [106](https://github.com/coredipper/operon/blob/main/examples/106_workflow_generation_interface.py): Workflow generation and registration
- [107](https://github.com/coredipper/operon/blob/main/examples/107_live_evaluation.py): Live evaluation with real LLM providers
