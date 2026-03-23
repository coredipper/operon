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
  - bi-temporal memory (`BiTemporalMemory`, `BiTemporalFact`, `BiTemporalQuery`, `SubstrateView`)

## Coordination / Surveillance / Healing

- `operon_ai.coordination`
- `operon_ai.surveillance`
- `operon_ai.health`
- `operon_ai.healing`

For concrete usage, start from the examples rather than reading the namespaces in isolation.
