# Convergence Implementation Roadmap

Date: 2026-03-24
Status: Active
Depends on: v0.23.1 (complete), convergence investigation doc

> **Goal:** Implement the four-layer convergence architecture (Operon structural /
> AsyncThink thinking / Swarms orchestration / AnimaWorks cognitive) through
> phased adapter development, co-design formalization, TLA+ formal verification,
> and shared evaluation.

This roadmap turns the convergence investigation doc into a dependency-ordered
implementation plan with concrete deliverables, success criteria, and version
targets.

### Theoretical foundations

Two additional sources inform this roadmap beyond the original investigation:

**Zardini (2023), "Co-Design of Complex Systems" (ETH Zurich thesis).**
Provides a monotone theory of co-design using category theory — design problems
as monotone maps between posets, composed via series/parallel/feedback. Directly
applicable to formalizing the adapter composition in the four-layer stack:
each adapter is a design problem, the full convergence is their composition,
and Zardini's fixed-point theory proves whether the adaptive assembly loop
converges. Key concepts: feasibility relations (what resources are sufficient
for what capabilities), functorial solution schemes (queries as functors),
and assume-guarantee contracts for compositional reasoning.

**Chi et al. (2025), "AsyncThink: The Era of Agentic Organization" (Microsoft Research).**
Introduces learned Fork/Join thinking structure within a single LLM via RL-
optimized organizer-worker protocol. Fills the gap between Operon's inter-agent
topology and individual agent reasoning: Operon decides stage structure,
AsyncThink decides how each stage thinks internally. Key concepts: thinking
concurrency ratio (η), critical-path latency optimization, and organization
policy generalization to unseen tasks.

### Four-layer architecture

```
┌─────────────────────────────────────────────┐
│  AnimaWorks (cognitive layer)               │
│  Identity, memory consolidation, forgetting,│
│  heartbeat, social coordination, priming    │
├─────────────────────────────────────────────┤
│  AsyncThink (thinking layer)                │
│  Fork/Join policy, concurrency optimization,│
│  learned execution structure within agents  │
├─────────────────────────────────────────────┤
│  Swarms (orchestration layer)               │
│  Enterprise deployment, scaling, monitoring,│
│  pattern execution, multi-provider routing  │
├─────────────────────────────────────────────┤
│  Operon (structural layer)                  │
│  Topology advice, epistemic analysis,       │
│  pattern scoring, watcher, auditability,    │
│  co-design formalization (Zardini)          │
└─────────────────────────────────────────────┘
```

Operon designs and validates the structure. AsyncThink optimizes how each
agent thinks within that structure. Swarms executes it at scale. AnimaWorks
gives it persistent cognition. Each layer is independently useful.

---

## Narrative Arc

| Release | Theme | One-line summary |
|---------|-------|------------------|
| v0.24.0 | **Foundation adapters** | Swarms topology analysis, AnimaWorks role mapping, shared types |
| v0.24.x | **Template exchange** | Swarms catalog → PatternLibrary seeding, AutoSwarmBuilder hybrid |
| v0.25.0 | **Memory + thinking convergence** | AnimaWorks memory bridge, PrimingView, AsyncThink thinking mode |
| v0.25.x | **Formal verification** | TLA+ specs, Zardini co-design formalization, fixed-point proofs |
| v0.26.0 | **Production runtime** | Swarms execution backend, distributed watcher, async stages |
| v0.26.x | **Evaluation + publication** | Shared benchmarks, convergence paper, documentation |

The progression: **adapters → exchange → memory/thinking → verification → runtime → evaluation**.

---

## Phase C1 — Foundation Adapters (v0.24.0)

**Theme:** Build the type-level bridges between the three projects without
runtime coupling.

**Justification:** Before any operational integration, we need adapter types
that can translate between Operon's typed dataclasses and the configuration
formats of Swarms and AnimaWorks. These are pure data transformations with
no runtime dependency on either external project.

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/convergence/__init__.py` | New convergence package |
| `operon_ai/convergence/swarms_adapter.py` | `SwarmTopologySpec` ↔ Operon `TopologyAdvice` mapping |
| `operon_ai/convergence/animaworks_adapter.py` | AnimaWorks role template ↔ Operon `SkillStage` mapping |
| `operon_ai/convergence/types.py` | Shared convergence types: `ExternalTopology`, `RuntimeConfig`, `AdapterResult` |
| `tests/unit/convergence/test_swarms_adapter.py` | Adapter tests (no Swarms dependency — test against dict specs) |
| `tests/unit/convergence/test_animaworks_adapter.py` | Adapter tests (no AnimaWorks dependency) |
| `examples/82_swarms_topology_analysis.py` | Analyze Swarms patterns with Operon's epistemic theorems |
| `examples/83_animaworks_role_mapping.py` | Map AnimaWorks role templates to Operon SkillStages |

**Key design principle:** Adapters must NOT import Swarms or AnimaWorks. They
operate on serializable dict/JSON representations of external configs. This
keeps Operon dependency-free.

**Type definitions:**

```python
@dataclass(frozen=True)
class ExternalTopology:
    """Serializable representation of an external system's agent topology."""
    source: str                          # "swarms" or "animaworks"
    pattern_name: str                    # e.g., "SequentialWorkflow", "HierarchicalSwarm"
    agents: tuple[dict[str, Any], ...]   # agent specs as dicts
    edges: tuple[tuple[str, str], ...]   # directed edges (from, to)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AdapterResult:
    """Result of adapting an external topology to Operon types."""
    topology_advice: TopologyAdvice
    suggested_template: PatternTemplate | None
    warnings: tuple[str, ...]            # structural concerns from epistemic analysis
    risk_score: float                    # 0.0 = safe, 1.0 = high risk
```

**Swarms adapter functions:**
- `parse_swarm_topology(pattern_name, agent_specs, edges) -> ExternalTopology`
- `analyze_external_topology(topology: ExternalTopology) -> AdapterResult`
  - Uses `advise_topology()` and epistemic theorems to assess structure
  - Flags error-amplifying topologies (Theorem 1), excessive handoff cost (Theorem 2)
- `swarm_to_template(topology: ExternalTopology) -> PatternTemplate`
  - Converts Swarms pattern spec to a registerable PatternTemplate

**AnimaWorks adapter functions:**
- `parse_animaworks_org(org_config: dict) -> ExternalTopology`
  - Reads supervisor hierarchy, role templates, agent count
- `animaworks_roles_to_stages(roles: list[dict]) -> tuple[SkillStage, ...]`
  - Maps role templates (engineer, manager, writer, etc.) to SkillStages with CognitiveMode

**Success criteria:**
- `analyze_external_topology()` correctly flags at least 3 of Swarms' 10+ patterns as structurally risky
- AnimaWorks role mapping produces valid SkillStages that pass organism assembly
- No runtime dependency on either external project

**Article updates:**

| Section | Change |
|---------|--------|
| `02-related-work.tex` | New subsection: "Convergence with Operational Runtimes" citing AnimaWorks, Swarms |
| `08-implementation.tex` | New subsection: convergence adapter architecture, ExternalTopology type |

**Documentation updates:**
- `docs/site/releases.md` — v0.24.0 section
- `docs/site/api.md` — new convergence package section
- `docs/site/concepts.md` — convergence layer in architecture
- New page: `docs/site/convergence.md` — overview of four-layer architecture
- `examples/README.md` — examples 82-83
- Blog post: "Operon v0.24: Convergence Begins"

---

## Phase C2 — Template Exchange (v0.24.x)

**Theme:** Seed Operon's PatternLibrary with Swarms' proven patterns and
build the AutoSwarmBuilder hybrid.

**Justification:** Operon's `PatternLibrary` starts empty. Swarms has 10+
battle-tested patterns. Seeding the library with converted templates gives
Operon's adaptive assembly a strong starting catalog.

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/convergence/catalog.py` | `seed_library_from_swarms(library, swarm_patterns)` |
| `operon_ai/convergence/hybrid_assembly.py` | `hybrid_skill_organism()` — AutoSwarmBuilder → template → library → adaptive |
| `examples/84_seeded_library.py` | Library seeded with Swarms patterns, scored retrieval |
| `examples/85_hybrid_assembly.py` | Hybrid: LLM-generated template → library registration → scoring |
| Tests | Template conversion, seeding, hybrid assembly |

**`seed_library_from_swarms()`:**
1. Convert each Swarms pattern to `ExternalTopology` via Phase C1 adapter
2. Run `analyze_external_topology()` to compute risk scores
3. Convert to `PatternTemplate` with appropriate `TaskFingerprint`
4. Register in library with risk_score as metadata

**`hybrid_skill_organism()`:**
1. Accept a task description + optional LLM for one-shot generation
2. If library has templates with score > threshold, use `adaptive_skill_organism()`
3. Otherwise, generate a template via LLM (mimicking AutoSwarmBuilder)
4. Register the generated template in the library
5. Run and record — the library learns from the LLM-generated template

**Success criteria:**
- Seeded library contains valid templates for all 10+ Swarms patterns
- `top_templates_for()` ranks seeded templates sensibly for test fingerprints
- Hybrid assembly falls back to LLM generation for novel tasks, then improves with scoring

**Article updates:**

| Section | Change |
|---------|--------|
| `06-discussion.tex` | Extend adaptive structure: seeded libraries, hybrid assembly, LLM-generated templates |
| `08-implementation.tex` | New subsection: template exchange protocol, hybrid assembly pipeline |

**Documentation updates:**
- `docs/site/releases.md` — v0.24.x section
- `docs/site/convergence.md` — template exchange section
- `examples/README.md` — examples 84-85
- HF Space: `operon-convergence-catalog` — browse seeded templates, compare rankings
- Blog post: "Operon v0.24: Seeding the Library from Swarms"

---

## Phase C3 — Memory + Thinking Convergence (v0.25.0)

**Theme:** Bridge AnimaWorks' memory systems into Operon's bi-temporal store,
evolve SubstrateView into a richer PrimingView, and add AsyncThink-inspired
intra-stage thinking structure.

**Justification:** AnimaWorks' 6-channel priming and consolidation pipeline
represent operational evidence for memory patterns Operon has formalized but
not deployed at scale. AsyncThink's organizer-worker protocol demonstrates
that intra-stage thinking structure (Fork/Join) can be learned via RL and
significantly reduce latency while improving accuracy. The bridge creates
auditable records from AnimaWorks' mutable memory; the async thinking mode
gives stages a richer internal execution model.

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/convergence/memory_bridge.py` | AnimaWorks episodic/semantic/procedural → BiTemporalMemory |
| `operon_ai/patterns/priming.py` | `PrimingView` — multi-channel frozen envelope replacing flat SubstrateView |
| `operon_ai/patterns/heartbeat.py` | `HeartbeatDaemon` — always-on watcher with periodic consolidation |
| `operon_ai/patterns/async_thinking.py` | `AsyncThinkingMode` — Fork/Join execution within a single stage (inspired by Chi et al.) |
| `examples/86_memory_bridge.py` | AnimaWorks memory → bi-temporal facts |
| `examples/87_priming_view.py` | Multi-channel stage priming |
| `examples/88_heartbeat_daemon.py` | Idle-time consolidation and curiosity |
| `examples/88b_async_thinking.py` | Stage with Fork/Join sub-queries and concurrency tracking |
| Tests | Bridge auditability, PrimingView channels, heartbeat scheduling, async thinking |

**AsyncThinkingMode (from Chi et al. "AsyncThink"):**

A new execution mode for `SkillStage` that allows a stage to internally Fork
sub-queries to parallel workers within the same nucleus, then Join results.
The organizer (stage handler) decides when to fork and when to join based on
an organization policy. Key additions:

- `mode="async"` on `SkillStage` — signals that this stage uses Fork/Join
- `AsyncOrganizer` manages sub-query dispatch within a single stage execution
- `ThinkingConcurrencyRatio` (η = avg active workers / capacity) as a new
  watcher signal source — low concurrency on parallelizable tasks triggers
  escalation
- `CriticalPathLatency` metric for async stages — minimum sequential depth
  of the Fork/Join DAG, computed via dynamic programming per Chi et al. §4.1
- Organization policy stored as `intervention_policy` in `PatternTemplate`,
  so `adaptive_skill_organism()` selects both topology and thinking structure

**PrimingView (evolves SubstrateView):**

```python
@dataclass(frozen=True)
class PrimingView:
    """Multi-channel context envelope for stage handlers."""
    facts: tuple[BiTemporalFact, ...]          # bi-temporal facts (existing)
    query: BiTemporalQuery | str | None        # original query
    record_time: datetime                       # record-time horizon
    # New channels (Phase C3)
    recent_outputs: tuple[dict[str, Any], ...]  # last N stage outputs
    telemetry: tuple[TelemetryEvent, ...]       # recent telemetry events
    experience: tuple[ExperienceRecord, ...]    # relevant past interventions
    developmental_status: DevelopmentStatus | None  # current maturity
    trust_context: dict[str, float]             # peer trust scores
```

**Breaking-change note:** `SubstrateView` is currently a concrete exported
dataclass (see `operon_ai/patterns/types.py`), and downstream tests assert
`isinstance(view, SubstrateView)`. `PrimingView` must subclass or alias
`SubstrateView` to preserve backward compatibility, or this becomes a breaking
change requiring a migration path.

**HeartbeatDaemon:**
- Extends `WatcherComponent` with a periodic `heartbeat()` method
- When idle (between runs), collects somatic/developmental signals
- Triggers `SleepConsolidation` when configurable conditions are met
  (time since last consolidation, accumulated run records, etc.)
- Optionally emits curiosity-driven exploration suggestions

**Success criteria:**
- Memory bridge creates valid bi-temporal facts from AnimaWorks-format dicts
- PrimingView provides all 6 AnimaWorks priming channels via Operon data sources
- HeartbeatDaemon triggers consolidation autonomously after accumulated runs

**Article updates:**

| Section | Change |
|---------|--------|
| `06-discussion.tex` | Extend memory discussion: PrimingView multi-channel model, heartbeat as always-on System M |
| `08-implementation.tex` | New subsection: memory bridge architecture, PrimingView channels, HeartbeatDaemon lifecycle |

**Documentation updates:**
- `docs/site/releases.md` — v0.25.0 section
- `docs/site/bitemporal-memory.md` — memory bridge section
- `docs/site/skill-organisms.md` — PrimingView replacing SubstrateView
- `docs/site/convergence.md` — memory convergence section
- `examples/README.md` — examples 86-88
- HF Space: `operon-priming` — visualize multi-channel priming for stages
- Blog post: "Operon v0.25: Memory Convergence and the Heartbeat"

---

## Phase C4 — Formal Verification + Co-Design Theory (v0.25.x)

**Theme:** TLA+ specifications for the three candidate protocols, Zardini
co-design formalization for the adapter composition, model-checked with TLC.

**Justification:** The convergence introduces distributed coordination problems
(template exchange, developmental gating, convergence detection across
organisms) that require formal verification. TLA+ specs prove the safety
invariants hold across all interleavings.

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `specs/TemplateExchangeProtocol.tla` | Template exchange with trust-weighted adoption |
| `specs/DevelopmentalGating.tla` | Lifecycle progression with critical periods |
| `specs/ConvergenceDetection.tla` | Intervention-rate convergence across organisms |
| `specs/README.md` | How to install TLA+ toolbox and run TLC model checker |
| `operon_ai/convergence/codesign.py` | Zardini co-design formalization: adapters as monotone maps, composition as monoidal product |
| Documentation | Safety invariants verified, liveness properties checked, co-design fixed-point convergence proved |

**Zardini co-design formalization (from Zardini 2023, ETH thesis):**

Each convergence adapter is formalized as a **design problem (DP)** in Zardini's
framework — a monotone map between a poset of resources (inputs/constraints)
and a poset of functionalities (outputs/capabilities):

- `SwarmTopologyAdapter` as a DP: resources = Swarms pattern spec; functionalities = Operon TopologyAdvice + risk score
- `MemoryBridgeAdapter` as a DP: resources = AnimaWorks memory entries; functionalities = BiTemporalFact records
- `AsyncThinkingAdapter` as a DP: resources = agent pool capacity + latency budget; functionalities = thinking concurrency ratio + accuracy

The full convergence stack is the **series composition** of these DPs (Ch. 5 of
Zardini). The adaptive assembly loop (`run → record → score → select`) is a
**feedback composition** (§5.4), and Zardini's domain-theoretic fixed-point
theory (§7.5) proves whether this loop converges to stable template rankings.

Key deliverable: `codesign.py` provides:
- `AdapterDP` protocol — monotone map interface for convergence adapters
- `compose_series(dp1, dp2)` — series composition
- `compose_parallel(dp1, dp2)` — parallel composition
- `feedback_fixed_point(dp, initial)` — iterative fixed-point computation
- `feasibility_check(dp, resources)` — does the adapter produce valid output for given input?

This gives the convergence stack a mathematical proof framework beyond TLA+:
TLA+ verifies temporal properties (safety/liveness over interleavings);
co-design theory verifies structural properties (feasibility, compositionality,
convergence of the adaptive loop).

**TemplateExchangeProtocol.tla:**
- State: `library[org]`, `trust[org][peer]`, `stage[org]`, `records[org]`
- Actions: `Export(org)`, `Import(org, peer)`, `RecordOutcome(org, tmpl, success)`
- Safety: template adoption respects `min_stage`; trust only changes via `RecordOutcome`
- Liveness: qualifying template eventually adopted; trust converges

**DevelopmentalGating.tla:**
- State: `telomere[org]`, `stage[org]`, `periods[org]`, `tools[org]`
- Actions: `Tick(org)`, `AcquireTool(org, tool)`, `Scaffold(teacher, learner)`
- Safety: capability gating never violated; critical periods never reopen; stages never regress
- Liveness: organism that keeps ticking eventually reaches MATURE

**ConvergenceDetection.tla:**
- State: `interventions[org]`, `stages[org]`, `halted[org]`
- Actions: `StageResult(org)`, `Intervene(org, kind)`, `CheckConvergence(org)`
- Safety: non-convergence always triggers HALT within bounded steps
- Liveness: convergent organism eventually completes

**Success criteria:**
- TLC model checker verifies all safety invariants for small state spaces (2-3 organisms, 5-10 stages)
- Liveness properties checked with fairness assumptions
- At least one atomicity refinement (coarse → fine-grained) verified

**Article updates:**

| Section | Change |
|---------|--------|
| `06-discussion.tex` | New subsection: "Formal Verification of Convergence Properties" — TLA+ methodology, locally stable guards |
| `08-implementation.tex` | New subsection: TLA+ spec descriptions, invariant catalog, TLC verification results |
| `09-conclusion.tex` | Update future work: formal verification now done, reference results |

**Documentation updates:**
- `docs/site/releases.md` — v0.25.x section
- `docs/site/convergence.md` — formal verification section with invariant catalog
- Update `docs/site/theory.md` — extend with TLA+ verification methodology
- Blog post: "Operon v0.25: Proving Convergence with TLA+"

---

## Phase C5 — Production Runtime (v0.26.0)

**Theme:** Compile Operon organisms into Swarms workflows for production
deployment, with AsyncThink-optimized stages.

**Justification:** Operon designs topologies; AsyncThink optimizes how each
stage thinks internally; Swarms deploys the result at scale.
This phase builds the compiler that translates `SkillOrganism` into Swarms'
`SequentialWorkflow` / `GraphWorkflow`, preserving watcher signals and
convergence detection.

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/convergence/swarms_compiler.py` | `organism_to_swarms(SkillOrganism) -> SwarmWorkflow` |
| `operon_ai/convergence/distributed_watcher.py` | Watcher that operates via async signals (for heartbeat runtimes) |
| `examples/90_swarms_deployment.py` | Operon-designed topology running in Swarms |
| `examples/91_distributed_watcher.py` | Watcher operating across process boundaries |
| `examples/92_async_stage_execution.py` | AsyncThink-style concurrent stage execution |
| Integration tests | End-to-end: design in Operon → deploy in Swarms → record in library |

**Compiler:**
- Maps `SkillStage` to Swarms `Agent` with provider config
- Maps `SkillOrganism.run()` loop to `SequentialWorkflow` or `GraphWorkflow`
- Injects watcher as Swarms monitoring hook (if Swarms supports hooks)
- Falls back to wrapper pattern if Swarms doesn't expose hooks

**Distributed watcher:**
- Instead of writing to `shared_state` dict, publishes interventions via
  configurable transport (in-memory queue, Redis, HTTP callback)
- Consumes signals from async sources (AnimaWorks heartbeat, Swarms metrics)
- Maintains the same decision priority chain (convergence → immune → epiplexity → curiosity → failure)

**Success criteria:**
- Operon-designed topology runs successfully through Swarms infrastructure
- Watcher detects non-convergence in a Swarms-deployed workflow
- Performance overhead of the Operon structural layer < 5% of total execution time

**Article updates:**

| Section | Change |
|---------|--------|
| `03-mapping.tex` | Extend typed composition: organism-to-workflow compilation as functor between categories |
| `08-implementation.tex` | New subsection: Swarms compiler, distributed watcher transport layer |

**Documentation updates:**
- `docs/site/releases.md` — v0.26.0 section
- `docs/site/convergence.md` — production runtime section
- `docs/site/skill-organisms.md` — distributed watcher documentation
- `examples/README.md` — examples 89-90
- HF Space: `operon-deployment` — visualize Operon topology compiled to Swarms workflow
- Blog post: "Operon v0.26: From Library to Production Runtime"

---

## Phase C6 — Evaluation + Publication (v0.26.x)

**Theme:** Shared benchmarks, convergence paper, final documentation.

**Justification:** The convergence must be evaluated empirically. Does Operon's
structural layer actually improve Swarms' default configurations? Does
AnimaWorks' consolidation benefit from bi-temporal auditability?

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| Shared eval harness | BFCL + AgentDojo benchmarks across all three projects |
| Comparative analysis | Operon topology advice vs. Swarms SwarmRouter vs. AnimaWorks defaults |
| Convergence paper | Article or blog series covering the four-layer architecture |
| Documentation | Updated docs site, API reference for convergence package |

**Evaluation protocol:**
1. Select 20 benchmark tasks spanning sequential, parallel, and mixed shapes
2. For each task, generate agent teams via:
   - Operon `adaptive_skill_organism()` with seeded library
   - Swarms `AutoSwarmBuilder` (one-shot LLM generation)
   - AnimaWorks default configuration
   - Hybrid (AutoSwarmBuilder → Operon library → scoring)
3. Run each through BFCL (function calling) and AgentDojo (prompt injection)
4. Measure: success rate, token cost, latency, intervention count, convergence rate
5. Compare: does the seeded library outperform LLM generation after N runs?
6. Measure: does bi-temporal auditability add value over mutable memory?

**Success criteria:**
- Clear recommendation on whether integration is worth pursuing (per workstream)
- At least two working proof-of-concept adapters demonstrated
- Seeded PatternLibrary outperforms empty library on benchmark tasks
- Convergence paper submitted or published

**Article updates:**

| Section | Change |
|---------|--------|
| `00-abstract.tex` | Final rewrite to include convergence results |
| `08-implementation.tex` | New subsection: evaluation protocol, benchmark results table |
| `09-conclusion.tex` | Rewrite to include convergence findings, four-layer architecture as validated result |
| `references.bib` | Add AnimaWorks, Swarms, Zardini, Chi et al. as references |

**Documentation updates:**
- `docs/site/releases.md` — v0.26.x section
- `docs/site/convergence.md` — evaluation results, recommendations
- Full docs site mirror to `coredipper.github.io`
- Blog post: "Operon v0.26: Convergence Complete — Four Layers Validated"

---

## HuggingFace Space Allocation

| Space | Phase | Purpose |
|-------|-------|---------|
| `operon-convergence-catalog` | C2 | Browse seeded templates, compare Operon vs. Swarms rankings |
| `operon-priming` | C3 | Visualize multi-channel PrimingView for stages |
| `operon-deployment` | C5 | Operon topology compiled to Swarms workflow visualization |

---

## Blog Post Allocation

| Blog Post | Phase | Topic |
|-----------|-------|-------|
| "Convergence Begins" | C1 | Foundation adapters, topology analysis, four-layer vision |
| "Seeding the Library from Swarms" | C2 | Template exchange, hybrid assembly |
| "Memory, Thinking, and the Heartbeat" | C3 | PrimingView, memory bridge, AsyncThink thinking mode, heartbeat |
| "Proving Convergence: TLA+ and Co-Design" | C4 | TLA+ verification, Zardini co-design formalization, fixed-point proofs |
| "From Library to Production Runtime" | C5 | Swarms compiler, async stages, distributed watcher |
| "Convergence Complete" | C6 | Evaluation results, four-layer validation, recommendations |

---

## Documentation Site Updates

Each phase updates `coredipper.github.io`:

| Page | Phase | Change |
|------|-------|--------|
| `operon/releases/` | All | New version section per phase |
| `operon/convergence/` (new) | C1 | Created; extended each subsequent phase |
| `operon/api/` | C1, C3, C5 | New convergence types |
| `operon/skill-organisms/` | C3 | PrimingView replacing SubstrateView |
| `operon/bitemporal-memory/` | C3 | Memory bridge section |
| `operon/concepts/` | C1, C6 | Architecture evolution |
| `operon/theory/` | C4 | TLA+ verification methodology + co-design theory |
| `operon/spaces/` | C2, C3, C5 | New spaces |
| `operon/examples/` | All | Examples 82-90 |
| `blog/` | All | One blog post per phase |

---

## Example Allocation

> **Note:** Examples 82–85 are already committed in the repository
> (`82_managed_organism.py`, `83_cli_stage_handler.py`, `84_cli_organism.py`,
> `85_claude_code_pipeline.py`). The convergence examples below need to be
> renumbered starting from 86 to avoid collisions.

| # | Example | Phase |
|---|---------|-------|
| 82 | `82_swarms_topology_analysis.py` | C1 |
| 83 | `83_animaworks_role_mapping.py` | C1 |
| 84 | `84_seeded_library.py` | C2 |
| 85 | `85_hybrid_assembly.py` | C2 |
| 86 | `86_memory_bridge.py` | C3 |
| 87 | `87_priming_view.py` | C3 |
| 88 | `88_heartbeat_daemon.py` | C3 |
| 88b | `88b_async_thinking.py` | C3 |
| 89 | `89_codesign_composition.py` | C4 |
| 90 | `90_swarms_deployment.py` | C5 |
| 91 | `91_distributed_watcher.py` | C5 |
| 92 | `92_async_stage_execution.py` | C5 |

---

## Risk Summary

| Risk | Impact | Mitigation |
|------|--------|------------|
| External APIs change | Adapters break | Operate on serializable dicts, not imports; pin to specific versions |
| Swarms doesn't expose monitoring hooks | Can't inject watcher | Wrapper pattern: Operon wraps Swarms rather than injecting into it |
| AnimaWorks memory format undocumented | Bridge guesses wrong | Start with documented REST/JSON endpoints; iterate |
| TLA+ model state space explosion | TLC doesn't terminate | Limit to 2-3 organisms, 5-10 stages; use symmetry reduction |
| Agency Tax too high | Operon layer adds unacceptable latency | Measure overhead explicitly; optimize hot paths; cache topology analysis |
| AsyncThink training data unavailable | Can't learn organization policies | Start with rule-based Fork/Join heuristics; add RL later when training infra available |
| Co-design fixed-point doesn't converge | Adaptive loop oscillates | Zardini's domain theory gives sufficient conditions; add damping if needed |
| Convergence paper rejected | No publication | Blog series as fallback; the code is the primary artifact |

---

## References

### External
- AnimaWorks: https://github.com/xuiltul/animaworks (Apache 2.0)
- Swarms: https://github.com/kyegomez/swarms (MIT)
- Swarms docs: https://docs.swarms.world
- Dupoux, LeCun & Malik (arXiv:2603.15381) — System A/B/M taxonomy
- Hao et al. (arXiv:2603.15371, BIGMAS) — intervention count as convergence signal
- Demirbas, "TLA+ Mental Models" (March 2026) — formal specification methodology
- Jiang et al. (arXiv:2602.19320) — Anatomy of Agentic Memory (Agency Tax, append-only robustness)
- Lin et al. (arXiv:2603.18718) — MemMA (probe-based memory verification)
- Feng et al. (arXiv:2602.04411) — Self-evolving Embodied AI (five co-evolving components)
- **Zardini (2023), "Co-Design of Complex Systems" (ETH Zurich thesis)** — monotone co-design theory, category of design problems, feasibility relations, functorial solution schemes, feedback fixed points
- **Chi et al. (arXiv:2510.26658), "AsyncThink: The Era of Agentic Organization"** — learned Fork/Join thinking structure, organizer-worker protocol, thinking concurrency ratio, critical-path latency, RL-optimized organization policy

### Internal
- Convergence investigation: `docs/plans/2026-03-23-operon-animaworks-convergence.md`
- Operon roadmap: `docs/plans/roadmap.md`
- Large-scope bitemporal: `docs/plans/2026-03-16-bitemporal-memory-large-implementation.md`
- MASFly integration: `docs/plans/2026-03-17-masfly-integration.md`
