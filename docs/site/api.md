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
