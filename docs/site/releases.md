# Release Notes

This page tracks the recent direction of the project.

## v0.25.0

Focus:

- evaluation harness, prompt optimization protocols, workflow generation (Phases C6+C7)
- 20 benchmark tasks x 7 configurations with MockEvaluator using real structural analysis
- PromptOptimizer and WorkflowGenerator protocol families

New:

- `MockEvaluator` — evaluation harness with structural variation and credit assignment
- `PromptOptimizer`, `EvolutionaryOptimizer`, `NoOpOptimizer` — prompt optimization protocols
- `attach_optimizer` — attach optimizer to SkillStage
- `WorkflowGenerator`, `ReasoningGenerator`, `HeuristicGenerator` — workflow generation protocols
- `generate_and_register` — generate workflow and register in PatternLibrary
- 20 benchmark tasks across 7 configurations (single, pipeline, fan-out, fan-in, diamond, full, stress)
- Structural variation analysis and credit assignment in evaluation
- Examples 104–106

## v0.24.1

Focus:

- production runtime compilers, distributed watcher, LangGraph integration (Phase C5)
- 4 deployment compilers (Swarms, DeerFlow, Ralph, Scion) plus 6 external adapter integrations

New:

- `organism_to_swarms()`, `managed_to_swarms()` — compile organism to Swarms workflow config
- `organism_to_deerflow()`, `managed_to_deerflow()` — compile organism to DeerFlow session config
- `organism_to_ralph()`, `managed_to_ralph()` — compile organism to Ralph event-driven hat config
- `organism_to_scion()`, `managed_to_scion()` — compile organism to Scion containerized grove config
- `DistributedWatcher` with `InMemoryTransport` and `HttpTransport` (webhook payload stub) — transport-abstracted convergence detection
- `operon_watcher_node()` — LangGraph-compatible convergence detection node
- `create_watcher_config()` — helper for LangGraph watcher configuration
- Examples 99–103

## v0.24.0

Focus:

- convergence adapters for [Swarms](https://github.com/kyegomez/swarms), [DeerFlow](https://github.com/bytedance/deer-flow), and [AnimaWorks](https://github.com/AnimaWorks/AnimaWorks)
- template exchange, [DeerFlow](https://github.com/bytedance/deer-flow) skill bridge, hybrid assembly
- PrimingView multi-channel context, memory bridge, HeartbeatDaemon
- AsyncThink Fork/Join execution, TLA+ formal verification, co-design theory

New:

- `operon_ai.convergence` package with 12 modules
- `ExternalTopology`, `AdapterResult` — shared adapter types
- `analyze_external_topology()` — epistemic theorems as structural linter
- `seed_library_from_swarms/deerflow/acg_survey` — catalog seeding
- `skill_to_template()`, `template_to_skill()` — bidirectional DeerFlow skill bridge
- `hybrid_skill_organism()` — library-first + LLM generator fallback
- `PrimingView` — multi-channel SubstrateView subclass (immutable via MappingProxyType)
- `HeartbeatDaemon` — idle-time consolidation via WatcherComponent extension
- `AsyncOrganizer`, `async_stage_handler()` — Fork/Join within stages
- `DesignProblem`, `compose_series/parallel`, `feedback_fixed_point` — Zardini co-design
- 3 TLA+ specifications (TemplateExchange, DevelopmentalGating, ConvergenceDetection)
- `prompt_optimizer` hook on SkillStage (interface for future DSPy integration)
- `parse_ralph_config()`, `ralph_hats_to_stages()` — Ralph adapter
- `parse_aevolve_workspace()`, `aevolve_skills_to_stages()` — A-Evolve adapter
- `seed_library_from_ralph/aevolve` — catalog seeding
- `EvolutionGating.tla` — TLA+ spec for evolution loop
- Examples 86–98

## v0.23.3

Focus:

- CLI stage handler for external tool integration
- Shell out to any CLI tool (Claude Code, Copilot, ruff, custom scripts) as organism stages

New:

- `cli_handler()` — factory that wraps any CLI command as a SkillStage handler
- `cli_organism()` — convenience for multi-CLI workflows via managed_organism
- `CLIResult` — structured output with stdout, stderr, returncode, latency, timed_out
- `_action_type` convention in handler output for signaling FAILURE to the watcher
- Output parsers: `parse_json()`, `parse_lines()`
- `examples/83_cli_stage_handler.py`

## v0.23.2

Focus:

- pattern-first ergonomics pass for the v0.19-0.23 subsystems
- one-call `managed_organism()` factory wiring the full stack
- top-level `consolidate()` convenience function

New:

- `ManagedOrganism`, `ManagedRunResult` — full-stack organism with run/consolidate/export/scaffold
- `managed_organism()` — batteries-included factory with sensible defaults
- `consolidate()` — one-call sleep consolidation
- `advise_topology()` gains optional `library` and `fingerprint` params
- `examples/82_managed_organism.py`

## v0.23.1

Focus:

- release integration and publication polish
- bi-temporal memory adapters (HistoneStore → BiTemporal, EpisodicMemory → BiTemporal)
- cross-subsystem integration tests (5 end-to-end tests)
- article rewrite (abstract, conclusion updated for full v0.19–v0.23 scope)

New:

- `histone_to_bitemporal()`, `episodic_to_bitemporal()` — memory bridge adapters
- Integration tests covering substrate+watcher, adaptive+consolidation, social+development, full lifecycle
- Article abstract covering six-layer progression
- Article conclusion with roadmap arc and updated future work

## v0.23.0

Focus:

- developmental staging (EMBRYONIC → JUVENILE → ADOLESCENT → MATURE)
- critical periods that close as organisms mature
- capability gating on Plasmid acquisition
- teacher-learner scaffolding

New:

- `DevelopmentController`, `DevelopmentConfig`, `DevelopmentalStage`, `DevelopmentStatus`
- `CriticalPeriod`, `StageTransition`, `stage_reached()`
- `Plasmid.min_stage` — developmental gating on tool acquisition
- `SocialLearning.scaffold_learner()` + `ScaffoldingResult`
- Watcher developmental signals (SOMATIC/development)
- `examples/80_developmental_staging.py` — lifecycle progression and gating
- `examples/81_critical_periods.py` — teacher-learner scaffolding
- [Developmental Staging Space](https://huggingface.co/spaces/coredipper/operon-development)
- Article updates: critical periods (§6), developmental staging impl (§8)

## v0.22.1

Focus:

- social learning with trust-weighted template exchange across organisms
- epistemic vigilance (TrustRegistry) for peer output trust scoring
- curiosity signals in WatcherComponent for novelty-seeking escalation

New:

- `SocialLearning`, `PeerExchange`, `TrustRegistry`, `AdoptionResult`, `AdoptionOutcome`
- Watcher curiosity signals (EPISTEMIC/curiosity) + `curiosity_escalation_threshold`
- `examples/78_social_learning.py` — template sharing with trust
- `examples/79_curiosity_driven_exploration.py` — curiosity-driven escalation
- [Social Learning Space](https://huggingface.co/spaces/coredipper/operon-social)
- Article updates: social learning + curiosity (§6, §8)

## v0.22.0

Focus:

- cognitive mode annotations (System A/B on SkillStage)
- sleep consolidation cycle (replay, compress, counterfactual, histone promotion)
- counterfactual replay over bi-temporal corrections

New:

- `CognitiveMode` enum, `resolve_cognitive_mode()` helper
- `SleepConsolidation`, `ConsolidationResult`, `CounterfactualResult`
- `counterfactual_replay()` — static analysis of corrected facts
- Watcher `mode_balance()` for System A/B distribution
- `examples/76_cognitive_modes.py` — mode annotations and watcher balance
- `examples/77_sleep_consolidation.py` — full consolidation cycle
- [Consolidation Space](https://huggingface.co/spaces/coredipper/operon-consolidation)
- Article updates: cognitive modes (§6, §8), sleep consolidation (§8)

## v0.21.1

Focus:

- adaptive assembly loop (fingerprint → template → assemble → run → record)
- experience pool on WatcherComponent for cross-run intervention learning

New:

- `AdaptiveSkillOrganism`, `AdaptiveRunResult` — compose-run-record lifecycle wrapper
- `adaptive_skill_organism()` — public factory for adaptive assembly
- `assemble_pattern()` — convert PatternTemplate into runnable topology
- `ExperienceRecord` — cross-run intervention memory on WatcherComponent
- `record_experience()`, `retrieve_similar_experiences()`, `recommend_intervention()`
- `examples/74_adaptive_assembly.py` — full adaptive loop
- `examples/75_experience_driven_watcher.py` — experience-driven recommendations
- [Adaptive Assembly Space](https://huggingface.co/spaces/coredipper/operon-adaptive)
- Article updates: evo-devo inner loop (§6), adaptive assembly impl (§8)

## v0.21.0

Focus:

- pattern repository for reusable collaboration templates
- watcher component with three-category signal taxonomy
- run-loop intervention mechanism (retry, escalate, halt)

New:

- `PatternLibrary`, `TaskFingerprint`, `PatternTemplate`, `PatternRunRecord`
- `WatcherComponent`, `WatcherConfig`, `WatcherSignal`, `SignalCategory`
- `InterventionKind`, `WatcherIntervention` — run-loop intervention types
- `examples/72_pattern_repository.py` — register, score, and retrieve templates
- `examples/73_watcher_component.py` — signal classification and interventions
- [Watcher Dashboard Space](https://huggingface.co/spaces/coredipper/operon-watcher)
- Article updates: adaptive assembly (§2, §6), watcher + pattern library (§8)

## v0.20.0

Focus:

- bi-temporal memory integration with SkillOrganism
- three-layer context model (topology, ephemeral, bi-temporal)
- HuggingFace Space for bi-temporal memory explorer

New:

- `SubstrateView` — frozen read-only envelope for substrate queries
- `SkillStage` fields: `read_query`, `fact_extractor`, `emit_output_fact`, `fact_tags`
- `SkillOrganism.substrate` — optional `BiTemporalMemory` for auditable shared facts
- `examples/71_bitemporal_skill_organism.py` — enterprise workflow with substrate
- [Bi-Temporal Memory Space](https://huggingface.co/spaces/coredipper/operon-bitemporal)
- Article updates: three-layer context model (§6), substrate integration (§8)

## v0.19.0

Focus:

- bi-temporal memory (valid time vs record time)
- append-only correction semantics
- belief-state reconstruction for compliance auditing
- article updates: temporal databases, temporal coalgebra, temporal epistemics

New:

- `BiTemporalMemory`, `BiTemporalFact`, `BiTemporalQuery`, `FactSnapshot`, `CorrectionResult`
- `examples/69_bitemporal_memory.py` — core API demo
- `examples/70_bitemporal_compliance_audit.py` — enterprise audit scenario
- [Bi-Temporal Memory docs](bitemporal-memory.md)

## v0.18

Focus:

- thinner front door
- pattern-first API
- provider-bound skill organisms
- attachable telemetry

Related writing:

- [Medium: Operon v0.18](https://medium.com/@coredipper/operon-v0-18-the-point-where-it-started-feeling-usable-b284d7b7317f)
- [Blog: Operon v0.18](https://banu.be/blog/operon-v018-pattern-first-skills/)

## v0.17

Focus:

- epistemic topology
- architecture-level analysis
- practical comparison to Kim et al.

Related writing:

- [Blog: Operon v0.17](https://banu.be/blog/operon-v017-epistemic-topology/)
