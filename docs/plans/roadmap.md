# Operon Development Roadmap

> v0.18.5 → v0.23.x | March 2026 – onward

This document sequences all planned work into dependency-ordered phases,
absorbs insights from Dupoux, LeCun & Malik (arXiv:2603.15381), and maps
deliverables to examples, HuggingFace Spaces, and evaluation infrastructure.

It **references** but does not duplicate the detailed plan documents in this
directory. Implementers should read the referenced plans for full specifications.

---

## Narrative Arc

| Release | Theme | One-line summary |
|---------|-------|------------------|
| v0.15–0.17 | Structural epistemics | Typed wiring, topology advice, coalgebras, optics |
| v0.18 | Pattern-first runtime | `reviewer_gate`, `specialist_swarm`, `skill_organism` |
| v0.19 | Temporal epistemics (Phase 1) | Bi-temporal memory with dual time axes |
| v0.20 | Temporal epistemics (Phase 2) | Substrate integration, three-layer context model |
| **v0.21** | **Adaptive structure** | Pattern repository, watcher, dynamic assembly |
| **v0.22** | **Cognitive architecture** | System A/B modes, sleep/consolidation, social learning |
| **v0.23** | **Developmental staging** | Critical periods, capability gating, publication |

The progression: **structure → memory → adaptation → cognition → development**.
Each layer assumes the previous one is stable.

---

## Dupoux / LeCun / Malik Mapping

The paper proposes three interacting systems for autonomous learning.
Below is how each maps onto Operon's existing and planned components.

| Paper concept | Operon mapping | Phase |
|---------------|---------------|-------|
| **System A** (observational, statistical, cheap) | `fast_nucleus` stages, mode="fixed" | Existing |
| **System B** (action-oriented, goal-directed, expensive) | `deep_nucleus` stages, mode="fuzzy" | Existing |
| **System M** (meta-control, routes data, switches modes) | `WatcherComponent` + `advise_topology()` | 3–4 |
| Bilevel evo-devo (outer loop optimizes init) | `PatternRepository` + `Genome` | 3–4 |
| Inner loop (development within one life) | Single `SkillOrganism` run with stages | Existing |
| Sleep / memory consolidation | `AutophagyDaemon` → `SleepConsolidation` | 5 |
| Imagination / counterfactual replay | `counterfactual_replay()` over bitemporal memory | 5 |
| Social learning / epistemic vigilance | `SocialLearning` extension of `QuorumSensing` | 6 |
| Intrinsic motivation / curiosity | Novelty signals in `Membrane` / `Watcher` | 6 |
| Critical periods / developmental gating | `Telomere` maturation + capability staging | 7 |
| Meta-state taxonomy (epistemic/somatic/species) | `WatcherComponent` signal classification | 3 |

---

## Phase 1 — Bi-Temporal Memory (v0.19.0) ✅

**Theme:** Temporal epistemics — "what did the agent know at time T?"

**Justification:** Foundation for auditable decision-making. Every subsequent
phase depends on durable, correctable fact storage. Compliance, healthcare,
and finance use cases demand this before adaptive features.

**Prerequisites:** v0.18.5 pattern-first API (done).

**Plan documents:**
- `2026-03-16-bitemporal-memory-design.md` — motivation and design space
- `2026-03-16-bitemporal-memory-implementation.md` — medium-scope core subsystem

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/memory/bitemporal.py` | `BiTemporalFact`, `BiTemporalQuery`, `BiTemporalMemory` |
| Unit tests | Temporal reconstruction, correction chains, snapshot queries |
| `examples/69_bitemporal_memory.py` | Core API: assert, query, correct, snapshot |
| `examples/70_bitemporal_compliance_audit.py` | Enterprise audit trail scenario |
| HF Space: `operon-bitemporal` | Interactive temporal query explorer |

**Article updates** (done):

| Section | Change |
|---------|--------|
| `02-related-work.tex` | New subsection: temporal databases, bi-temporal data models (Snodgrass, SQL:2011) |
| `03-mapping.tex` | New paragraph in coalgebra subsection: temporal state as bi-temporal coalgebra |
| `06-discussion.tex` | New subsubsection: "Temporal Epistemics — What Did the Agent Know?" |
| `08-implementation.tex` | New subsection (§8.4): `BiTemporalMemory` implementation, API, audit trail |
| `references.bib` | Added `snodgrass2000temporal`, `kulkarni2012temporal` |

**Success criteria:**
- `query_at(valid_time, record_time)` returns correct belief state
- `correct()` preserves full history (append-only)
- `snapshot_at()` reconstructs complete agent belief at any point
- All existing tests pass (no regressions)

---

## Phase 2 — Bi-Temporal + SkillOrganism Integration (v0.20.0) ✅

**Theme:** Auditable shared substrate for multi-stage workflows.

**Justification:** Generic bi-temporal memory answers "what was true at time T?"
but SkillOrganism needs "how should multiple stages share and revise knowledge
during a workflow?" — the three-layer context model (topology + ephemeral state +
durable bitemporal facts).

**Prerequisites:** Phase 1 core subsystem.

**Plan document:**
- `2026-03-17-bitemporal-skill-organism-integration.md`

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `SkillOrganism.substrate` parameter | Optional `BiTemporalMemory` binding |
| Stage-aware fact lifecycle | Stages assert/query/correct facts through shared substrate |
| `examples/71_bitemporal_skill_organism.py` | Multi-stage workflow with auditable shared facts |
| Integration tests | Cross-stage fact visibility, correction semantics |

**Article updates:**

| Section | Change |
|---------|--------|
| `06-discussion.tex` | Extend multi-cellular subsection: three-layer context model (topology + ephemeral + bitemporal) |
| `08-implementation.tex` | Extend SkillOrganism subsection: `substrate` parameter, stage-aware fact lifecycle |

**Success criteria:**
- Evaluator stages can demote/correct facts without destroying history
- `explain_decision_at(stage, time)` traces what was known when
- Zero overhead when `substrate=None` (backward compatible)

---

## Phase 3 — MASFly Static Foundations (v0.21.0)

**Theme:** Pattern repository and watcher — the static scaffolding for adaptation.

**Justification:** Before dynamic assembly, Operon needs a place to store
successful patterns and a component that can observe runtime execution. The
paper's System M provides the theoretical backbone: the watcher is a meta-
control orchestrator with a principled signal taxonomy (epistemic / somatic /
species-specific).

**Prerequisites:** Phase 2 (bitemporal substrate available for watcher memory).

**Plan document:**
- `2026-03-17-masfly-integration.md` — workstreams 1 (repository) and 3 (watcher)

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/patterns/repository.py` | `PatternLibrary`: `TaskFingerprint`, `PatternTemplate`, `PatternRunRecord` |
| `operon_ai/patterns/watcher.py` | `WatcherComponent(SkillRuntimeComponent)` with three-category signal intake |
| `examples/72_pattern_repository.py` | Register, retrieve, and score pattern templates |
| `examples/73_watcher_component.py` | Runtime monitoring with retry/escalation/halt interventions |
| HF Space: `operon-watcher` | Live watcher dashboard showing signal classification |
| Unit + integration tests | Template CRUD, watcher signal routing, intervention triggers |

**Paper-enriched design decisions:**
- Watcher signal inputs classified as **epistemic** (epiplexity, prediction error),
  **somatic** (ATP/metabolic state), or **species-specific** (immune threats, membrane alerts)
- **Intervention count as convergence signal:** when the Watcher's cumulative
  retry/escalate count exceeds a threshold relative to stage count, emit a
  non-convergence epistemic signal that can trigger `halt`. Empirically validated
  by Hao et al. (arXiv:2603.15371, "BIGMAS"): incorrect runs systematically
  require more routing decisions than successful ones (e.g., 9.4 vs 7.3 mean
  on planning tasks). Cheap to implement (~20 LOC in `WatcherComponent`)
- Pattern repository framed as **evolutionary memory** (bilevel evo-devo outer loop)

**Article updates:**

| Section | Change |
|---------|--------|
| `02-related-work.tex` | Add Dupoux, LeCun & Malik (2603.15381) and Hao et al. (2603.15371, BIGMAS) to Related Work; add MASFly reference |
| `03-mapping.tex` | New subsection: meta-control as System M — watcher signal taxonomy mapping |
| `04-operad.tex` | Extend immunity/quorum sections: watcher as meta-control operator in the operad |
| `06-discussion.tex` | New subsection: "Adaptive Structure Selection — From Static Advice to Learned Patterns" |
| `08-implementation.tex` | New subsections: `PatternLibrary`, `WatcherComponent` implementation |
| `references.bib` | Add Dupoux/LeCun/Malik and Hao et al. (BIGMAS) bib entries |

**Success criteria:**
- `PatternLibrary.top_templates_for(fingerprint)` returns ranked matches
- `WatcherComponent` consumes `EpiplexityMonitor`, `TelemetryProbe`, `ImmuneSystem` signals
- Intervention count triggers non-convergence signal when rate exceeds threshold
- Phase 1 interventions work: retry, escalate to deep model, halt
- No mid-flight graph mutation yet (that's Phase 4)

---

## Phase 4 — Adaptive Assembly (v0.21.x)

**Theme:** Dynamic organism construction from experience — the evo-devo inner loop.

**Justification:** With patterns stored and a watcher observing, Operon can now
automatically select and instantiate the right topology for a new task. This
closes the loop between outcome and future pattern selection.

**Prerequisites:** Phase 3 (repository + watcher).

**Plan document:**
- `2026-03-17-masfly-integration.md` — workstreams 2 (adaptive assembly),
  4 (experience pool), 5 (adaptive wrapper)

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `operon_ai/patterns/adaptive.py` | `adaptive_skill_organism()`, `assemble_pattern()` |
| Experience pool | `record_intervention()`, `retrieve_similar_interventions()` in watcher |
| `AdaptiveSkillOrganism` wrapper | construct → run → record → patch futures |
| `examples/74_adaptive_assembly.py` | Task fingerprinting → template selection → execution → recording |
| `examples/75_experience_driven_watcher.py` | Watcher using intervention history to recommend actions |
| HF Space: `operon-adaptive` | Adaptive assembly visualization |

**Article updates:**

| Section | Change |
|---------|--------|
| `03-mapping.tex` | Extend Genome subsection: bilevel evo-devo — Genome as genetic code φ, PatternRepository as evolutionary memory |
| `06-discussion.tex` | Extend adaptive structure subsection: experience pool, intervention memory, closed-loop adaptation |
| `08-implementation.tex` | New subsection: `AdaptiveSkillOrganism` wrapper, `assemble_pattern()` |

**Success criteria:**
- `adaptive_skill_organism(task)` selects template from repository
- Run outcomes feed back into repository scores
- Watcher recommends interventions based on similar past situations
- `SkillOrganism` API unchanged (adapter pattern, not modification)

---

## Phase 5 — Cognitive Modes + Sleep/Consolidation (v0.22.0)

**Theme:** System A/B cognitive reframing and autophagy extension into
sleep/consolidation cycles.

**Justification:** The paper reframes Operon's fast/deep nucleus distinction
from a cost optimization into a cognitive architecture principle. System A
discovers representations; System B discovers causal structure. Sleep
consolidation goes beyond pruning to include pattern replay and schema
formation.

**Prerequisites:** Phase 4 (pattern repository to consolidate into).

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `CognitiveMode` annotations on `SkillStage` | `observational` vs `action_oriented` mode declarations |
| `SleepConsolidation` cycle | Extends `AutophagyDaemon`: prune + replay + compress + counterfactual |
| `counterfactual_replay()` | Replay decisions with updated facts over bitemporal memory |
| `examples/76_cognitive_modes.py` | Stage mode annotations with watcher reasoning about mode balance |
| `examples/77_sleep_consolidation.py` | Sleep cycle: prune context, replay patterns, compress to templates |
| HF Space: `operon-consolidation` | Consolidation cycle visualization |

**Paper-enriched design decisions:**
- Stages declare cognitive mode; watcher reasons about A/B mode balance
- Sleep cycle replays successful patterns from session into `PatternLibrary`
- Counterfactual replay uses bitemporal `diff_between()` to compare outcomes
- Histone tier promotion (WORKING → EPISODIC → LONGTERM) happens during sleep

**Article updates:**

| Section | Change |
|---------|--------|
| `05-pathology.tex` | Extend cognitive healing (autophagy) subsection: sleep consolidation as extended autophagy |
| `06-discussion.tex` | New subsection: "Cognitive Modes — System A/B as Fast/Deep Stage Annotations" |
| `06-discussion.tex` | Extend bioenergetic intelligence subsection: counterfactual replay, imagination-based learning |
| `08-implementation.tex` | New subsection: `SleepConsolidation` cycle, `counterfactual_replay()` |

**Success criteria:**
- Mode annotations are queryable at runtime
- Sleep consolidation produces new pattern templates from episodic memory
- Counterfactual replay detects cases where updated facts would change outcome
- AutophagyDaemon backward compatible (consolidation is opt-in extension)

---

## Phase 6 — Social Learning + Curiosity (v0.22.x)

**Theme:** New capabilities directly motivated by the paper — agents that learn
from peers and seek informative inputs.

**Justification:** The paper's appendix on social learning (B.1) and intrinsic
motivation describes capabilities that go beyond Operon's current coordination
model. Agents should not just coordinate — they should learn from observing each
other's successes and actively seek novel situations.

**Prerequisites:** Phase 5 (cognitive modes for curiosity signals).

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| Social learning extension | `SocialLearning` on `QuorumSensing` — learn from peer agent runs |
| Epistemic vigilance | Trust scoring for peer agent outputs based on track record |
| Curiosity signals | Novelty/prediction-error signals in `Membrane` and `Watcher` |
| `examples/78_social_learning.py` | Agents learning topology preferences from peer observations |
| `examples/79_curiosity_driven_exploration.py` | Active information seeking via novelty signals |

**Article updates:**

| Section | Change |
|---------|--------|
| `04-operad.tex` | Extend quorum sensing: from voting to learning — social learning as compositional pattern |
| `06-discussion.tex` | New subsection: "Social Learning and Epistemic Vigilance" |
| `06-discussion.tex` | Extend membrane discussion: curiosity-driven attention as active membrane filtering |

**Success criteria:**
- Agent A can adopt a pattern template that worked for Agent B
- Trust scores modulate how much weight is given to peer outputs
- Curiosity signals trigger watcher escalation on novel inputs
- No mandatory coupling — social learning is opt-in per organism

---

## Phase 7 — Developmental Staging + Critical Periods (v0.23.0)

**Theme:** Telomere maturation as capability gating — the paper's critical
periods applied to agent development.

**Justification:** The paper's biological critical periods suggest temporal
capability gating: young agents use simple tools, mature agents unlock complex
capabilities. This extends `Telomere` from lifecycle tracking to developmental
staging and extends `Plasmid` to respect developmental phase.

**Prerequisites:** Phase 6 (social learning for teacher-learner dynamics).

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| `DevelopmentalStage` on `Telomere` | Phase-aware capability gating (early/intermediate/mature) |
| `Plasmid` developmental awareness | Tool acquisition respects current developmental stage |
| Teacher-learner scaffolding | Mature agents scaffold younger agents' learning |
| `examples/80_developmental_staging.py` | Agent progressing through capability phases |
| `examples/81_critical_periods.py` | Time-limited learning windows with phase transitions |
| HF Space: `operon-development` | Developmental trajectory visualization |

**Article updates:**

| Section | Change |
|---------|--------|
| `03-mapping.tex` | Extend telomere subsection: developmental staging as lifecycle-gated capability acquisition |
| `05-pathology.tex` | New failure mode: premature capability exposure (developmental disorder analogy) |
| `06-discussion.tex` | New subsection: "Critical Periods and Developmental Gating" |

**Success criteria:**
- Agents start with restricted capabilities, unlock more over time
- Critical periods close: early-phase-only capabilities become fixed
- Teacher agents can accelerate learner development
- Telomere lifecycle events trigger phase transitions

---

## Phase 8 — Release Integration + Publication (v0.23.x)

**Theme:** Documentation, paper updates, and the large-scope bitemporal release.

**Prerequisites:** Phases 1–7 substantially complete.

**Plan document:**
- `2026-03-16-bitemporal-memory-large-implementation.md` — release-wide integration

**Deliverables:**

| Artifact | Description |
|----------|-------------|
| Large-scope bitemporal integration | Adapters from HistoneStore/EpisodicMemory/Healing into bitemporal |
| Updated paper (`article/main.tex`) | Final polish of all per-phase article updates; compile with `tectonic` |
| Documentation site updates | API reference, tutorials, architecture guide |
| Cross-subsystem integration tests | End-to-end: bitemporal + watcher + adaptive + consolidation |
| Publication-grade eval results | 100-seed runs across all suites |

**Article updates:**

| Section | Change |
|---------|--------|
| `00-abstract.tex` | Rewrite to reflect full v0.19–v0.23 scope (temporal + adaptive + cognitive + developmental) |
| `01-introduction.tex` | Update contributions list with new subsystems |
| `08-implementation.tex` | Update eval protocol subsection with BFCL/AgentDojo results, 100-seed publication runs |
| `09-conclusion.tex` | Rewrite conclusion to cover complete narrative arc |

**Success criteria:**
- All subsystems interoperate cleanly
- Paper narrative covers v0.19–v0.23 progression
- Documentation covers all new APIs with examples
- No regressions across 81 examples

---

## White Paper (`article/main.tex`)

The paper is built with [Tectonic](https://tectonic-typesetting.github.io/),
a self-contained LaTeX engine that downloads packages on demand and requires
no manual TeX distribution setup.

```bash
# Build the PDF (from repo root)
cd article && tectonic main.tex

# Output: article/main.pdf
```

Each phase includes an **Article updates** table listing the sections to
modify. Updates are drafted in-phase and compiled/polished in Phase 8
before publication.

**Current section structure:**

| File | Section |
|------|---------|
| `00-abstract.tex` | Abstract |
| `01-introduction.tex` | Introduction (biological heuristic, categorical bridge, contributions) |
| `02-related-work.tex` | Related Work (network motifs, ACT, agentic AI, epistemic logic, temporal databases) |
| `03-mapping.tex` | The Mapping: Biology ↔ Software (Poly, lenses, optics, coalgebras, temporal coalgebra) |
| `04-operad.tex` | Formal Syntax: The Agentic Operad (typing, composition, immunity) |
| `05-pathology.tex` | Failure Modes & Pathology (oncology, autoimmunity, prion, ischemia) |
| `06-discussion.tex` | Discussion: Towards Epigenetic Software (RAG, HGT, multi-cellular, epistemic topology, temporal epistemics) |
| `07-optimization.tex` | Diagram Optimization via Categorical Rewriting |
| `08-implementation.tex` | Reference Implementation (architecture, organelles, bi-temporal memory, eval protocol) |
| `09-conclusion.tex` | Conclusion |
| `references.bib` | Bibliography (BibTeX) |

---

## Parallel Tracks

These run alongside the phased work and are not blocked by it.

### Provider Ecosystem

Continue expanding provider support (currently Anthropic, Gemini, OpenAI, and
OpenAI-compatible backends). Each new provider gets a nucleus binding and at
least one example demonstrating it.

### Tissue Patterns

New multi-cellular patterns building on the existing `tissue.py` and
`cell_type.py` infrastructure. Driven by community use cases.

### Documentation Site

Ongoing improvements to `docs/site/`. New pages for each phase's APIs as
they ship. Space documentation updated with each new HF deployment.

### Evaluation Harness (BFCL / AgentDojo)

**Plan documents:**
- `2026-02-13-eval_harness_status.md` — current infrastructure
- `2026-02-14-bfcl-agentdojo-design.md` — design for BFCL + AgentDojo
- `2026-02-14-bfcl-agentdojo-implementation.md` — step-by-step implementation

Replaces StableToolBench with deterministic, pip-installable benchmarks:
- **BFCL** (Berkeley Function Calling): tests Chaperone folding/repair with real schemas
- **AgentDojo** (ETH Zurich): tests Immune System with prompt injection payloads

This work can proceed in parallel and should be substantially complete before
Phase 8 publication.

### Example Consistency

**Plan document:**
- `2025-12-25-examples-consistency.md`

Standardize all existing examples (imports, error handling, documentation).
This was originally targeted for before v0.19.0; the current priority is
to complete it before adding further examples beyond 85.

---

## Example Allocation

Existing examples: 01–85 (85 files; no gaps).

| # | Example | Phase |
|---|---------|-------|
| 69 | `69_bitemporal_memory.py` | 1 |
| 70 | `70_bitemporal_compliance_audit.py` | 1 |
| 71 | `71_bitemporal_skill_organism.py` | 2 |
| 72 | `72_pattern_repository.py` | 3 |
| 73 | `73_watcher_component.py` | 3 |
| 74 | `74_adaptive_assembly.py` | 4 |
| 75 | `75_experience_driven_watcher.py` | 4 |
| 76 | `76_cognitive_modes.py` | 5 |
| 77 | `77_sleep_consolidation.py` | 5 |
| 78 | `78_social_learning.py` | 6 |
| 79 | `79_curiosity_driven_exploration.py` | 6 |
| 80 | `80_developmental_staging.py` | 7 |
| 81 | `81_critical_periods.py` | 7 |

---

## HuggingFace Space Allocation

Existing spaces: `operon-epistemic`, `operon-diagram-builder`.

| Space | Phase | Purpose |
|-------|-------|---------|
| `operon-bitemporal` | 1 | Interactive temporal query explorer |
| `operon-watcher` | 3 | Live watcher dashboard with signal classification |
| `operon-adaptive` | 4 | Adaptive assembly visualization |
| `operon-consolidation` | 5 | Sleep/consolidation cycle visualization |
| `operon-development` | 7 | Developmental trajectory visualization |

---

## Risk Summary

| Risk | Impact | Mitigation |
|------|--------|------------|
| Bitemporal memory scope creep | Delays all downstream phases | Ship medium scope first (Phase 1), large scope in Phase 8 |
| Watcher complexity | Runtime overhead, hard to debug | Phase 1 interventions only (retry/escalate/halt), no graph mutation |
| Paper concepts don't translate cleanly | Forced abstractions | Map onto existing shapes, don't create new subsystems just for terminology |
| Example count growth (81) | Maintenance burden | Example consistency plan must complete before v0.19.0 |
| Adaptive assembly overfitting | Patterns that work on benchmarks but not real tasks | Experience pool with decay; eval harness as reality check |
| Social learning trust gaming | Agents colluding to inflate trust scores | Epistemic vigilance + immune system cross-check |

---

## Recommended Reading Order

For implementers approaching this roadmap:

1. **This document** — overall sequence and rationale
2. **`2026-03-16-bitemporal-memory-design.md`** — understand the temporal epistemics motivation
3. **`2026-03-16-bitemporal-memory-implementation.md`** — Phase 1 specification
4. **`2026-03-17-bitemporal-skill-organism-integration.md`** — Phase 2 specification
5. **`2026-03-17-masfly-integration.md`** — Phases 3–4 specification
6. **Dupoux, LeCun & Malik (arXiv:2603.15381)** — theoretical grounding for Phases 5–7
7. **`2026-03-16-bitemporal-memory-large-implementation.md`** — Phase 8 release integration
8. **`2026-02-14-bfcl-agentdojo-design.md`** + **implementation** — parallel eval track
9. **`2025-12-25-examples-consistency.md`** — parallel example cleanup
10. **`2026-03-16-pattern-first-api-v018.md`** — current API surface (reference)

---

## References

### Plan Documents (this directory)

- `2025-12-25-examples-consistency.md` — example standardization
- `2026-02-13-eval_harness_status.md` — eval infrastructure status
- `2026-02-14-bfcl-agentdojo-design.md` — BFCL + AgentDojo design
- `2026-02-14-bfcl-agentdojo-implementation.md` — BFCL + AgentDojo implementation
- `2026-03-16-bitemporal-memory-design.md` — bitemporal memory motivation
- `2026-03-16-bitemporal-memory-implementation.md` — bitemporal core subsystem
- `2026-03-16-bitemporal-memory-large-implementation.md` — release-wide bitemporal
- `2026-03-16-pattern-first-api-v018.md` — v0.18 pattern-first API
- `2026-03-17-bitemporal-skill-organism-integration.md` — bitemporal + SkillOrganism
- `2026-03-17-masfly-integration.md` — MASFly adaptive layer

### External

- Dupoux, LeCun & Malik. "Why AI systems don't learn and what to do about it."
  arXiv:2603.15381, March 2026.
- Hao, Dai, Qin & Yu. "Brain-Inspired Graph Multi-Agent Systems for LLM Reasoning."
  arXiv:2603.15371, March 2026. (Intervention count as convergence signal in Phase 3.)
