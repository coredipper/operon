# Convergence Implementation Roadmap

Date: 2026-03-24
Status: Active
Depends on: v0.23.1 (complete), convergence investigation doc

> **Goal:** Implement the three-layer convergence architecture (Operon structural /
> Swarms orchestration / AnimaWorks cognitive) through phased adapter development,
> TLA+ formal verification, and shared evaluation.

This roadmap turns the convergence investigation doc into a dependency-ordered
implementation plan with concrete deliverables, success criteria, and version
targets.

---

## Narrative Arc

| Release | Theme | One-line summary |
|---------|-------|------------------|
| v0.24.0 | **Foundation adapters** | Swarms topology analysis, AnimaWorks role mapping, shared types |
| v0.24.x | **Template exchange** | Swarms catalog → PatternLibrary seeding, AutoSwarmBuilder hybrid |
| v0.25.0 | **Memory convergence** | AnimaWorks memory bridge, PrimingView, heartbeat daemon |
| v0.25.x | **Formal verification** | TLA+ specs for template exchange, developmental gating, convergence |
| v0.26.0 | **Production runtime** | Swarms execution backend, distributed watcher, scaling tests |
| v0.26.x | **Evaluation + publication** | Shared benchmarks, convergence paper, documentation |

The progression: **adapters → exchange → memory → verification → runtime → evaluation**.

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

---

## Phase C3 — Memory Convergence (v0.25.0)

**Theme:** Bridge AnimaWorks' memory systems into Operon's bi-temporal store
and evolve SubstrateView into a richer PrimingView.

**Justification:** AnimaWorks' 6-channel priming and consolidation pipeline
represent operational evidence for memory patterns Operon has formalized but
not deployed at scale. The bridge creates auditable records from AnimaWorks'
mutable memory.

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/convergence/memory_bridge.py` | AnimaWorks episodic/semantic/procedural → BiTemporalMemory |
| `operon_ai/patterns/priming.py` | `PrimingView` — multi-channel frozen envelope replacing flat SubstrateView |
| `operon_ai/patterns/heartbeat.py` | `HeartbeatDaemon` — always-on watcher with periodic consolidation |
| `examples/86_memory_bridge.py` | AnimaWorks memory → bi-temporal facts |
| `examples/87_priming_view.py` | Multi-channel stage priming |
| `examples/88_heartbeat_daemon.py` | Idle-time consolidation and curiosity |
| Tests | Bridge auditability, PrimingView channels, heartbeat scheduling |

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

Backward-compatible: existing handlers that accept `SubstrateView` as 5th arg
continue to work via arity-aware dispatch. `PrimingView` is a superset.

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

---

## Phase C4 — Formal Verification (v0.25.x)

**Theme:** TLA+ specifications for the three candidate protocols, model-checked
with TLC.

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
| Documentation | Safety invariants verified, liveness properties checked |

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

---

## Phase C5 — Production Runtime (v0.26.0)

**Theme:** Compile Operon organisms into Swarms workflows for production
deployment.

**Justification:** Operon designs topologies; Swarms deploys them at scale.
This phase builds the compiler that translates `SkillOrganism` into Swarms'
`SequentialWorkflow` / `GraphWorkflow`, preserving watcher signals and
convergence detection.

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/convergence/swarms_compiler.py` | `organism_to_swarms(SkillOrganism) -> SwarmWorkflow` |
| `operon_ai/convergence/distributed_watcher.py` | Watcher that operates via async signals (for heartbeat runtimes) |
| `examples/89_swarms_deployment.py` | Operon-designed topology running in Swarms |
| `examples/90_distributed_watcher.py` | Watcher operating across process boundaries |
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
| Convergence paper | Article or blog series covering the three-layer architecture |
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

---

## Example Allocation

| # | Example | Phase |
|---|---------|-------|
| 82 | `82_swarms_topology_analysis.py` | C1 |
| 83 | `83_animaworks_role_mapping.py` | C1 |
| 84 | `84_seeded_library.py` | C2 |
| 85 | `85_hybrid_assembly.py` | C2 |
| 86 | `86_memory_bridge.py` | C3 |
| 87 | `87_priming_view.py` | C3 |
| 88 | `88_heartbeat_daemon.py` | C3 |
| 89 | `89_swarms_deployment.py` | C5 |
| 90 | `90_distributed_watcher.py` | C5 |

---

## Risk Summary

| Risk | Impact | Mitigation |
|------|--------|------------|
| External APIs change | Adapters break | Operate on serializable dicts, not imports; pin to specific versions |
| Swarms doesn't expose monitoring hooks | Can't inject watcher | Wrapper pattern: Operon wraps Swarms rather than injecting into it |
| AnimaWorks memory format undocumented | Bridge guesses wrong | Start with documented REST/JSON endpoints; iterate |
| TLA+ model state space explosion | TLC doesn't terminate | Limit to 2-3 organisms, 5-10 stages; use symmetry reduction |
| Agency Tax too high | Operon layer adds unacceptable latency | Measure overhead explicitly; optimize hot paths; cache topology analysis |
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

### Internal
- Convergence investigation: `docs/plans/2026-03-23-operon-animaworks-convergence.md`
- Operon roadmap: `docs/plans/roadmap.md`
- Large-scope bitemporal: `docs/plans/2026-03-16-bitemporal-memory-large-implementation.md`
- MASFly integration: `docs/plans/2026-03-17-masfly-integration.md`
