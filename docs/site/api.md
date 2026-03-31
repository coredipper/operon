# API Overview

This is a lightweight map of the current library surface.

## Recommended Public Layer

- `operon_ai.patterns`
  - `advise_topology(...)`
  - `reviewer_gate(...)`
  - `specialist_swarm(...)`
  - `skill_organism(...)`
  - `PatternLibrary`, `TaskFingerprint`, `PatternTemplate`, `PatternRunRecord`
  - `WatcherComponent`, `WatcherConfig`, `WatcherSignal`, `SignalCategory`
  - `InterventionKind`, `WatcherIntervention`
  - `adaptive_skill_organism(...)`, `assemble_pattern(...)`
  - `AdaptiveSkillOrganism`, `AdaptiveRunResult`, `ExperienceRecord`
  - `CognitiveMode`, `resolve_cognitive_mode()`
  - `SleepConsolidation`, `ConsolidationResult`, `CounterfactualResult`, `counterfactual_replay()`
  - `SocialLearning`, `PeerExchange`, `TrustRegistry`, `AdoptionResult`, `AdoptionOutcome`

## Core Runtime and Analysis

- `operon_ai.core`
  - typed wiring
  - epistemic analysis
  - coalgebraic utilities
  - denaturation and optics

## Organelles

- `operon_ai.organelles`
  - `Nucleus`
  - `Membrane`
  - `Mitochondria`
  - `Chaperone`
  - `Ribosome`
  - `Lysosome`

## State / Memory

- `operon_ai.state`
  - ATP / metabolism
  - genome
  - histone memory
  - lifecycle-related state

- `operon_ai.memory`
  - episodic memory
  - bi-temporal memory (`BiTemporalMemory`, `BiTemporalFact`, `BiTemporalQuery`)

- `operon_ai.patterns`
  - `SubstrateView` — read-only view passed to stage handlers during substrate-backed runs

## Coordination / Surveillance / Healing

- `operon_ai.coordination`
- `operon_ai.surveillance`
- `operon_ai.health`
- `operon_ai.healing`

For concrete usage, start from the examples rather than reading the namespaces in isolation.

## Convergence

- `operon_ai.convergence`
  - adapters: `parse_swarm_topology`, `parse_animaworks_org`, `parse_deerflow_session`
  - analysis: `analyze_external_topology`, `topology_to_template`
  - types: `ExternalTopology`, `AdapterResult`, `RuntimeConfig`
  - catalog: `seed_library_from_swarms`, `seed_library_from_deerflow`, `seed_library_from_acg_survey`
  - skill bridge: `skill_to_template`, `template_to_skill`
  - hybrid: `hybrid_skill_organism`, `default_template_generator`
  - async thinking: `AsyncOrganizer`, `AsyncThinkResult`, `async_stage_handler`
  - memory bridge: `bridge_animaworks_memory`, `bridge_deerflow_memory`
  - co-design: `DesignProblem`, `compose_series`, `compose_parallel`, `feedback_fixed_point`

- `operon_ai.patterns.priming`
  - `PrimingView` — multi-channel SubstrateView subclass
  - `build_priming_view` — promote SubstrateView to PrimingView

- `operon_ai.patterns.heartbeat`
  - `HeartbeatDaemon` — WatcherComponent with idle-time consolidation

## Ralph Adapter

- `operon_ai.convergence.ralph_adapter`
  - `parse_ralph_config` — convert Ralph hat config to ExternalTopology
  - `ralph_hats_to_stages` — map Ralph hats to SkillStages with CognitiveMode

- `operon_ai.convergence.catalog`
  - `seed_library_from_ralph` — seed PatternLibrary from Ralph hat patterns

## A-Evolve Adapter

- `operon_ai.convergence.aevolve_adapter`
  - `parse_aevolve_workspace` — convert A-Evolve workspace manifest to ExternalTopology
  - `aevolve_skills_to_stages` — map evolved skills to SkillStages

- `operon_ai.convergence.aevolve_skills`
  - `import_aevolve_skills` — import SKILL.md strings into PatternLibrary
  - `seed_library_from_aevolve` — seed from workspace manifests

All functions are also re-exported from `operon_ai.convergence` directly.

## Compilers (C5)

- `operon_ai.convergence.swarms_compiler`
  - `organism_to_swarms`, `managed_to_swarms`

- `operon_ai.convergence.deerflow_compiler`
  - `organism_to_deerflow`, `managed_to_deerflow`

- `operon_ai.convergence.ralph_compiler`
  - `organism_to_ralph`, `managed_to_ralph`

- `operon_ai.convergence.scion_compiler`
  - `organism_to_scion`, `managed_to_scion`

## Distributed Watcher

- `operon_ai.convergence.distributed_watcher`
  - `DistributedWatcher`, `InMemoryTransport`, `HttpTransport`

- `operon_ai.convergence.langgraph_watcher`
  - `operon_watcher_node`, `create_watcher_config`

## Prompt Optimization (C7)

- `operon_ai.convergence.prompt_optimization`
  - `PromptOptimizer` — protocol for prompt-level optimization
  - `EvolutionaryOptimizer` — evolutionary prompt mutation with fitness gating
  - `NoOpOptimizer` — pass-through optimizer for baselines
  - `attach_optimizer` — attach an optimizer instance to a SkillStage

## Workflow Generation (C7)

- `operon_ai.convergence.workflow_generation`
  - `WorkflowGenerator` — protocol for workflow topology generation
  - `ReasoningGenerator` — reasoning-based workflow construction
  - `HeuristicGenerator` — heuristic-based workflow construction
  - `generate_and_register` — generate a workflow and register it in PatternLibrary
