# Comprehensive Convergence Plan

## Context

This plan merges five source documents into one authoritative implementation
roadmap for Operon's convergence architecture — integrating with Swarms,
DeerFlow, AnimaWorks, Ralph, A-Evolve, AsyncThink, and Scion through
phased adapter development, co-design formalization, and shared evaluation.

**Current state (v0.24.1, as of 2026-03-30):**
- Phases C1-C4 complete: 14 convergence modules, 4 TLA+ specs
- 5 adapters: Swarms, DeerFlow, AnimaWorks, Ralph, A-Evolve
- 98 examples (86-98 are convergence), 1404 tests
- Convergence paper written (`article/paper2/main.tex`)
- Published to PyPI as `operon-ai==0.24.1`

**Remaining:** C5 (compilers for 4 deployment targets), C6 (evaluation), C7 (future)

**Key decisions:**
- 6-layer architecture: A-Evolve / AnimaWorks / AsyncThink / Ralph+DeerFlow+Swarms / Operon / Scion
- Scion is the infrastructure layer (below Operon), not an orchestration peer
- Scion's philosophy ("less structure") is in productive tension with Operon's thesis — C6 evaluates this
- Prompt optimization (DSPy) seeded in C3, full implementation via A-Evolve in C7

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  A-Evolve (evolution layer)                 │
│  Workspace mutation, skill forging, EGL     │
├─────────────────────────────────────────────┤
│  AnimaWorks (cognitive layer)               │
│  Identity, memory consolidation, priming    │
├─────────────────────────────────────────────┤
│  AsyncThink (thinking layer)                │
│  Fork/Join policy, concurrency optimization │
├─────────────────────────────────────────────┤
│  Ralph / DeerFlow / Swarms (orchestration)  │
│  Event-driven / LangGraph / graph workflows │
├─────────────────────────────────────────────┤
│  Operon (structural layer)                  │
│  Topology advice, epistemic analysis,       │
│  pattern scoring, watcher, auditability     │
├─────────────────────────────────────────────┤
│  Scion (infrastructure layer)               │
│  Container provisioning, git worktrees,     │
│  Hub/Broker, OTEL telemetry, Kubernetes     │
└─────────────────────────────────────────────┘
```

---

## Phase C1 — Foundation Adapters (v0.24.0)

**Goal:** Type-level bridges to Swarms, DeerFlow, AnimaWorks with zero runtime coupling.

| Deliverable | Path |
|-------------|------|
| Convergence package | `operon_ai/convergence/__init__.py` |
| Shared types | `operon_ai/convergence/types.py` — `ExternalTopology`, `AdapterResult` |
| Swarms adapter | `operon_ai/convergence/swarms_adapter.py` |
| AnimaWorks adapter | `operon_ai/convergence/animaworks_adapter.py` |
| DeerFlow adapter | `operon_ai/convergence/deerflow_adapter.py` |
| Tests (4 files) | `tests/unit/convergence/test_{types,swarms,animaworks,deerflow}_adapter.py` |
| Examples 86-88 | Swarms topology, AnimaWorks roles, DeerFlow workflow analysis |

**Core types** (`convergence/types.py`):
```python
@dataclass(frozen=True)
class ExternalTopology:
    source: str                          # "swarms" | "animaworks" | "deerflow"
    pattern_name: str
    agents: tuple[dict[str, Any], ...]
    edges: tuple[tuple[str, str], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AdapterResult:
    topology_advice: TopologyAdvice
    suggested_template: PatternTemplate | None
    warnings: tuple[str, ...]
    risk_score: float
```

**Design principle:** Adapters operate on dicts/JSON only — no imports of
Swarms, DeerFlow, or AnimaWorks. All three adapters produce the same
`ExternalTopology` → `analyze_external_topology()` consumes it.

**Success criteria:**
- `analyze_external_topology()` flags 3+ Swarms patterns as structurally risky
- DeerFlow sessions and AnimaWorks org configs parse to valid `ExternalTopology`
- `import operon_ai.convergence` works without installing any external project

---

## Phase C2 — Template Exchange (v0.24.x)

**Goal:** Seed PatternLibrary from Swarms, DeerFlow skills, and ACG survey patterns.

| Deliverable | Path |
|-------------|------|
| Catalog seeding | `operon_ai/convergence/catalog.py` — `seed_library_from_swarms()`, `seed_library_from_deerflow()`, `seed_library_from_acg_survey()` |
| DeerFlow skill bridge | `operon_ai/convergence/deerflow_skills.py` — bidirectional `skill_to_template()` / `template_to_skill()` |
| Hybrid assembly | `operon_ai/convergence/hybrid_assembly.py` — `hybrid_skill_organism()` |
| Tests (3 files) | `tests/unit/convergence/test_{catalog,deerflow_skills,hybrid_assembly}.py` |
| Examples 89-91 | Seeded library, hybrid assembly, DeerFlow skill bridge |

**Hybrid assembly logic:** Library templates > threshold → adaptive assembly;
otherwise LLM generates one-shot → registers → scoring refines over time.

**ACG survey integration:** `seed_library_from_acg_survey()` registers templates
from the survey's 40+ comparison cards with metadata (`source="acg_survey"`,
`graph_determination_time`, `graph_plasticity_mode`).

---

## Phase C3 — Memory + Thinking Convergence (v0.25.0)

**Goal:** Bridge memory systems, evolve SubstrateView → PrimingView, add AsyncThink Fork/Join.

| Deliverable | Path |
|-------------|------|
| Memory bridge | `operon_ai/convergence/memory_bridge.py` — AnimaWorks + DeerFlow → BiTemporalMemory |
| PrimingView | `operon_ai/patterns/priming.py` — subclass of `SubstrateView` (line 121, `types.py`) |
| HeartbeatDaemon | `operon_ai/patterns/heartbeat.py` |
| AsyncThinkingMode | `operon_ai/convergence/async_thinking.py` — Fork/Join within stages |
| Prompt optimizer hook | Add `prompt_optimizer: Callable | None = None` to `SkillStage` in `patterns/types.py` |
| Tests (4 files) | `tests/unit/{convergence,patterns}/test_{memory_bridge,priming,heartbeat,async_thinking}.py` |
| Examples 92-95 | Memory bridge, PrimingView, heartbeat, async thinking |

**PrimingView backward compatibility:** Must subclass `SubstrateView` so
`isinstance(view, SubstrateView)` continues to pass:
```python
@dataclass(frozen=True)
class PrimingView(SubstrateView):
    recent_outputs: tuple[dict[str, Any], ...] = ()
    telemetry: tuple[TelemetryEvent, ...] = ()
    experience: tuple[ExperienceRecord, ...] = ()
    developmental_status: DevelopmentStatus | None = None
    trust_context: dict[str, float] = field(default_factory=dict)
```

**Prompt optimizer hook (ACG survey gap — interface only):** Defines the
extension point for future DSPy-style optimization without implementing it.

---

## Phase C4 — Formal Verification + Co-Design (v0.25.x)

**Goal:** TLA+ specs for 3 protocols, Zardini co-design formalization.

| Deliverable | Path |
|-------------|------|
| TLA+ specs (3) | `specs/TemplateExchangeProtocol.tla`, `DevelopmentalGating.tla`, `ConvergenceDetection.tla` |
| TLA+ readme | `specs/README.md` |
| Co-design module | `operon_ai/convergence/codesign.py` — `AdapterDP`, `compose_series`, `feedback_fixed_point` |
| Tests | `tests/unit/convergence/test_codesign.py` |
| Example 96 | Co-design adapter composition |

**Safety invariants:** Template adoption respects `min_stage`, trust changes
only via `record_outcome`, critical periods never reopen, convergence budget
triggers HALT.

**Co-design:** Each C1 adapter is a design problem (monotone map). The full
stack is series composition. The adaptive loop is feedback composition.
Zardini's fixed-point theory proves whether scoring stabilizes.

---

## Phase C5 — Production Runtime (v0.26.0)

**Goal:** Compile Operon organisms into executable configs for 4 deployment
targets: Swarms, DeerFlow, Ralph, and Scion.

| Deliverable | Path |
|-------------|------|
| Swarms compiler | `operon_ai/convergence/swarms_compiler.py` — `organism_to_swarms()` |
| DeerFlow compiler | `operon_ai/convergence/deerflow_compiler.py` — `organism_to_deerflow()` |
| Ralph compiler | `operon_ai/convergence/ralph_compiler.py` — `organism_to_ralph()` |
| Scion compiler | `operon_ai/convergence/scion_compiler.py` — `organism_to_scion()` |
| LangGraph watcher | `operon_ai/convergence/langgraph_watcher.py` — native LangGraph node |
| Distributed watcher | `operon_ai/convergence/distributed_watcher.py` — Redis/HTTP/OTEL transport |
| Tests (5 files) | `test_{swarms,deerflow,ralph,scion}_compiler.py`, `test_distributed_watcher.py` |
| Examples 99-103 | Swarms, DeerFlow, Ralph, Scion deployment, distributed watcher |

**Compilers output serializable dicts** (no framework imports). Users install
the target framework separately and pass the dict to their API.

**DeerFlow advantage:** `operon_watcher_node()` is a native LangGraph node —
deepest integration possible. Progressive skill loading maps to developmental staging.

**Ralph advantage:** Event subscriptions compiled from topology edges. Backpressure
gates mapped from watcher intervention thresholds.

**[Scion](https://github.com/GoogleCloudPlatform/scion) advantage:** The only
compilation target that provisions *isolated execution environments*, not just
workflow configs. Each stage becomes a containerized agent with its own git
worktree, credentials, and resource limits. The Hub/Runtime Broker enables
multi-machine deployment. Scion's OpenTelemetry integration feeds directly
into the distributed watcher's signal taxonomy.

`organism_to_scion()` output:
- Grove config (project namespace, git repo)
- Per-stage agent definitions (template, system prompt, skills, runtime profile)
- Inter-agent messaging topology (via `scion message`)
- Watcher as a dedicated monitoring agent consuming OTEL telemetry
- Kubernetes runtime config for production deployment

**Scion philosophical tension:** Scion's design philosophy ("Less is More",
"Isolation Over Constraints", "Agent Lifecycles are Dynamic") argues that
as frontier models improve, explicit structural complexity matters less than
flexible collaboration substrates. This is in direct tension with Operon's
thesis that topology-level guarantees prevent failure modes that better
models alone cannot fix. The integration is productive precisely because of
this tension: Scion provides the runtime substrate where Operon's structural
analysis is either validated or invalidated by real agent behavior. If
Scion's agents self-organize effectively without structural guidance, that
is evidence against Operon's thesis. If error amplification and sequential
overhead are real in practice, that is evidence for it. Phase C6 evaluation
should explicitly test this question.

---

## Phase C6 — Evaluation + Publication (v0.26.x)

**Goal:** Benchmark across all projects, structural credit assignment, convergence paper.
The central evaluation question: **does Operon's structural analysis actually
help, or do agents self-organize effectively without it?** Scion's "action
over pondering" philosophy makes this the most honest test of the framework.

| Deliverable | Description |
|-------------|-------------|
| Eval harness | BFCL + AgentDojo across all 7 configurations (Operon, Swarms, DeerFlow, AnimaWorks, Ralph, Scion unguided, Scion + Operon) |
| Structural variation | ACG gap: measure graph topology variation across inputs |
| Credit assignment | ACG gap: per-stage attribution of success/failure |
| Convergence paper | Six-layer architecture with evaluation results |

**Evaluation:** 20 tasks × 7 configurations. Measure success rate, token
cost, latency, intervention count, convergence rate, structural variation.

| # | Configuration | Structural guidance | Runtime |
|---|---------------|--------------------|---------|
| 1 | Operon adaptive | Full (epistemic theorems + template scoring) | Local |
| 2 | Swarms AutoSwarmBuilder | None (one-shot LLM generation) | Local |
| 3 | DeerFlow default | None (LangGraph default) | Docker |
| 4 | AnimaWorks default | None (supervisor hierarchy) | Local |
| 5 | Ralph default | None (event-driven self-organization) | Local |
| 6 | **Scion unguided** | **None** (agents self-organize via CLI) | Container |
| 7 | **Scion + Operon** | **Full** (Operon compiles, Scion deploys) | Container |

Configurations 6 vs 7 directly test the Scion philosophical question: does
Operon's structural analysis improve outcomes when agents have full autonomy
in isolated containers? If config 7 outperforms config 6, structural
guidance helps even with capable models. If config 6 matches or exceeds
config 7, Scion's "less is more" philosophy holds.

**Citation strategy:**
- Introduction: Evans et al. (plural intelligence motivation)
- Related work: ACG survey (definitive taxonomy), note ACG ↔ wiring diagram correspondence
- Theory: Zardini co-design, Chi et al. AsyncThink
- Discussion: DeerFlow integration as engineering validation

---

## Phase C7 — Future Work (deferred)

1. DSPy-style prompt optimization (interface seeded in C3)
2. Workflow generation from trained models (WorkflowLLM, FlowReasoner)
3. RL-trained organization policies for AsyncThink
4. Diffusion-based workflow generation

---

## Example Allocation (86-103)

| # | Phase | Status | Description |
|---|-------|--------|-------------|
| 86-88 | C1 | Done | Swarms/AnimaWorks/DeerFlow topology analysis |
| 89-91 | C2 | Done | Seeded library, hybrid assembly, skill bridge |
| 92-95 | C3 | Done | Memory bridge, PrimingView, heartbeat, async thinking |
| 96 | C4 | Done | Co-design composition |
| 97-98 | C1+ | Done | Ralph hat analysis, A-Evolve workspace analysis |
| 99 | C5 | Planned | Swarms deployment |
| 100 | C5 | Planned | DeerFlow deployment |
| 101 | C5 | Planned | Ralph deployment |
| 102 | C5 | Planned | Scion deployment |
| 103 | C5 | Planned | Distributed watcher |

---

## Dependency Graph

```
C1 (Foundation)
 ├──→ C2 (Template Exchange)
 └──→ C3 (Memory + Thinking)
           └──→ C4 (Verification)
                    └──→ C5 (Runtime)
                              └──→ C6 (Evaluation)
                                        └──→ C7 (Future)
```

C2 and C3 can proceed in parallel after C1.

---

## Critical Files

| File | Why |
|------|-----|
| `operon_ai/convergence/types.py` | All adapters produce and all phases consume `ExternalTopology` |
| `operon_ai/patterns/types.py:121` | `SubstrateView` must be preserved; `PrimingView` subclasses it |
| `operon_ai/patterns/adaptive.py` | Extended for `hybrid_skill_organism()` in C2 |
| `operon_ai/convergence/codesign.py` | Theoretical anchor: proves adaptive loop converges |
| `operon_ai/convergence/deerflow_adapter.py` | Primary new deliverable validating the abstraction |

---

## Verification

After each phase:
1. `python -m pytest tests/ -x -q` — all tests pass
2. `python examples/NN_*.py` — new examples run without error
3. No new imports of Swarms/DeerFlow/AnimaWorks/Ralph/A-Evolve/Scion in operon_ai/
4. For C4: `tlc specs/*.tla` — model checker verifies invariants
5. For C5: compiler output is valid JSON/dict (no framework objects)
6. For C6: eval harness produces comparative metrics table

Final: convergence paper cites all three overlap analyses, positions Operon
within the ACG taxonomy, and reports evaluation results.
