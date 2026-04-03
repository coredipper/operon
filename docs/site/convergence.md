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

## Meta-Evolution (Phase C8 — Phase A)

Evolves organism configurations (modes, models, thresholds) using biological primitives at the meta-level. Tests whether Operon's abstractions generalize from running organisms to evolving them.

### Architecture

- `FilesystemOptimizer` — new protocol distinct from C7's `EvolutionaryOptimizer`. Operates on `CandidateConfig` (organism configurations), not prompts.
- `EvolutionLoop` — main glue: proposes candidates, evaluates via `LiveEvaluator`, persists to filesystem store, uses `EpiplexityMonitor` for stall detection
- `CandidateConfig` maps to/from `Genome` — tests whether the gene abstraction covers configuration space (lossless round-trip confirmed)
- Candidate-first filesystem layout with append-only `index.jsonl` for efficient querying

### Proposers

Hybrid strategy: `TournamentMutator` (programmatic, fast) + `LLMProposer` (Gemini/local, rich context).

**Dual stall detection** triggers the LLM proposer:
1. Config novelty stall — `EpiplexityMonitor` with pluggable `DistanceProvider` (tests scale-invariance of epistemic health monitoring)
2. Score plateau — best score not improved in `threshold * n_tasks` steps

### Live Evaluation

- `LiveEvaluator` — real LLM calls through SkillOrganism pipelines (Gemini, OpenAI, Anthropic, Ollama, LM Studio, Claude CLI, Codex CLI)
- LLM-as-judge quality scoring with rubric (correctness 50%, completeness 30%, clarity 20%)
- Cross-judging support for provider-independent evaluation

### Key Findings

**Biological abstraction generalization:**
- Genome mapping is clean (~5 lines). Gene abstraction covers full configuration space with lossless round-trip.
- EpiplexityMonitor generalizes across scales. `ConfigHammingDistance` triggers STAGNANT/EXPLORING just like embedding cosine distance.
- `DesignProblem` wrapping of evolution steps is natural. Co-design composition works at the meta-level.
- Boundaries found: `feedback_fixed_point` doesn't fit (evolution isn't convergent iteration), `TrustRegistry` overkill for 2 proposers.

**Ao et al. (arXiv:2603.26993) test — exogenous signals:**
- Compressed history (index entries only): LLM proposer mean 0.15 — worse than tournament (0.32)
- Rich filesystem context (configs + trace metadata): LLM proposer mean 0.49 — matches tournament (0.44)
- Finding: rich context helps 3x, but config-space evolution doesn't strongly benefit from LLM reasoning over blind mutation. Topology evolution (Phase B) is where structural reasoning should provide genuine advantage.

**Related work:** de los Riscos, Corbacho & Arbib ([arXiv:2603.28906](https://arxiv.org/abs/2603.28906)) provide a category-theoretic framework (ArchAgents) that maps tightly to Operon's architecture: objects = organism architectures, morphisms = compilers, agents = configured organisms.

### Phase B — Topology Mutations (Implemented)

Topology mutations extend the evolution loop to add/remove stages and rewire edges. `TournamentMutator` applies 70% config / 30% topology mutations. `CandidateConfig.edges` stores wiring pairs with lossless Genome round-trip.

**Key Phase B findings:**
- Tournament *improved* with topology (0.44 → 0.59): blind mutation handles structural changes productively
- LLM proposer *degraded* (0.49 → 0.37): reasoning about edges that don't affect sequential execution wastes capacity
- Finding: topology mutations need execution support (`_build_organism` using edges for non-sequential wiring) before LLM structural reasoning can help. Currently organisms always run sequentially regardless of edges.

### Remaining Work

- Topology-aware organism execution (use edges for non-sequential wiring)
- Pareto convergence criterion
- Per-stage model routing (currently one model per tier)
- TrustRegistry for >2 proposer strategies

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
- [108](https://github.com/coredipper/operon/blob/main/examples/108_meta_evolution.py): Meta-evolution of organism configurations
