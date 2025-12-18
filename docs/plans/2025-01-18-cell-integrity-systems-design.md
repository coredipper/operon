# Cell Integrity Systems Design

**Date:** 2025-01-18
**Status:** Draft
**Author:** Brainstorming session

---

## Overview

This document describes three interconnected systems for ensuring robustness in Operon agent architectures:

1. **Quality System (Ubiquitin-Proteasome)** — Prevents cascade failures through provenance tracking
2. **Surveillance System (Immune)** — Detects Byzantine agents through behavioral fingerprinting
3. **Coordination System (Cell Cycle)** — Prevents deadlocks through checkpoints and watchdogs

These systems are biologically inspired and integrate to form a complete "cell integrity" layer.

---

## Problem Statement

### Failure Modes Addressed

| Failure Mode | Description | System |
|--------------|-------------|--------|
| **Cascade failures** | Bad output from Agent A corrupts downstream agents | Quality |
| **Silent corruption** | Agent produces plausible but wrong outputs | Surveillance |
| **Adversarial takeover** | Agent is prompt-injected, works against goals | Surveillance |
| **Circular dependencies** | Agents wait on each other indefinitely | Coordination |
| **Resource contention** | Agents starve competing for limited resources | Coordination |
| **Priority inversion** | High-priority work blocked by low-priority locks | Coordination |

---

## System 1: Quality (Ubiquitin-Proteasome)

### Biological Model

The ubiquitin-proteasome system is the cell's quality control inspector:
- **E3 Ligases** attach ubiquitin tags to proteins based on quality signals
- **DUBs (Deubiquitinases)** remove tags, rescuing proteins
- **Proteasome** inspects tags and degrades proteins below threshold
- **Degrons** determine protein-specific degradation rates

### Components

#### UbiquitinTag

```python
@dataclass(frozen=True)
class UbiquitinTag:
    confidence: float           # 0.0 to 1.0, decays through chain
    origin: str                 # Agent that created the data
    generation: int             # How many agents it's passed through
    chain_type: ChainType       # K48 (degradation), K63 (signaling), etc.
    degron: DegronType          # Data-specific decay rate
    chain_length: int           # For recycling accounting
    integrity: IntegrityLabel   # Integration with existing Operon types
```

#### TaggedData

Wrapper that pairs data with its provenance tag:

```python
@dataclass
class TaggedData(Generic[T]):
    data: T
    tag: UbiquitinTag
```

#### UbiquitinPool

Manages tag allocation and recycling:
- Tracks available tags (limited resource)
- Handles pool exhaustion strategies (BLOCK, PASSTHROUGH, RECYCLE_OLDEST)
- Recycles tags when data is degraded

#### E3Ligase and Deubiquitinase

Context-sensitive taggers and erasers:

```python
@dataclass
class E3Ligase:
    name: str
    active: Callable[[ProvenanceContext], bool]
    substrate_match: Callable[[Any], bool]
    tag_strength: Callable[[ProvenanceContext], float]

@dataclass
class Deubiquitinase:
    name: str
    active: Callable[[ProvenanceContext], bool]
    rescue_condition: Callable[[UbiquitinTag, ProvenanceContext], bool]
    rescue_amount: float
```

#### ChaperoneRepair

Attempts repair before degradation (biological: chaperones refold misfolded proteins):

```python
@dataclass
class ChaperoneRepair:
    name: str
    can_repair: Callable[[Any, UbiquitinTag], bool]
    repair: Callable[[Any, UbiquitinTag], tuple[Any, bool]]
    confidence_boost: float = 0.3
```

#### Proteasome

Inspects tags and enforces quality thresholds:
- **Degradation threshold** — Below this, switch to fallback
- **Block threshold** — Below this, stop propagation entirely
- **Chaperone repair** — Try repair before degradation
- **DUB rescue** — Try confidence restoration first
- **Degron-adjusted thresholds** — Different data types have different sensitivity

#### ProvenanceWiringDiagram

Extends existing `WiringDiagram` with automatic provenance propagation:
- Tags flow through connections automatically
- Configurable decay rates per connection
- E3 ligases and DUBs can be attached to connections
- Proteasome checkpoints at critical junctions

#### ProvenanceRuntime

Executes wiring diagrams with provenance tracking:
- Handles fan-out (clones tags for each branch)
- Merges tags when paths converge
- Tracks execution trace for debugging

### Key Mechanisms

1. **Confidence decay** — Each hop reduces confidence by configurable rate
2. **Degron types** — STABLE, NORMAL, UNSTABLE, IMMEDIATE (different half-lives)
3. **Chaperone repair** — Attempt fix before degradation (JSON repair, retry, etc.)
4. **Ubiquitin recycling** — Reclaim tags when data is discarded
5. **Dynamic policies** — Decay rates can adjust based on runtime context

---

## System 2: Surveillance (Immune)

### Biological Model

The MHC/immune system detects compromised cells:
- **MHC Display** — Cells present protein fragments for inspection
- **Thymus** — T-cells learn "self" patterns during training
- **T-Cells** — Require two signals to activate (prevent false positives)
- **Tregs** — Suppress overactive immune responses
- **Memory** — Remember threat patterns for faster re-response

### Components

#### MHCPeptide

Behavioral fingerprint displayed by an agent:

```python
@dataclass(frozen=True)
class MHCPeptide:
    agent_id: str
    output_length_mean: float
    output_length_std: float
    response_time_mean: float
    response_time_std: float
    vocabulary_hash: str
    structure_hash: str
    confidence_mean: float
    error_rate: float
    canary_accuracy: Optional[float]
```

#### MHCDisplay

Collector that builds fingerprints from agent outputs:
- Rolling window of observations
- Tracks output lengths, response times, confidence, vocabulary, structure
- Records canary test results

#### Thymus (Baseline Training)

Trains surveillance on what "normal" looks like:
- Requires minimum training samples
- Calculates statistical bounds (mean ± tolerance × std)
- Stores valid vocabulary/structure hashes
- Selection results: POSITIVE (trained), NEGATIVE (would attack self), ANERGIC (useless)

#### BaselineProfile

Learned "self" pattern for an agent:

```python
@dataclass
class BaselineProfile:
    agent_id: str
    output_length_bounds: tuple[float, float]
    response_time_bounds: tuple[float, float]
    confidence_bounds: tuple[float, float]
    error_rate_max: float
    valid_vocabulary_hashes: set[str]
    valid_structure_hashes: set[str]
    canary_accuracy_min: float
```

#### TCell (Two-Signal Response)

Surveillance responder requiring confirmation before action:
- **Signal 1** — MHC recognition (anomaly detected)
- **Signal 2** — Co-stimulation (canary failed, cross-validation, repeated anomaly, manual flag)
- Both signals required for activation
- Anergy: repeated Signal 1 without Signal 2 → stop responding

#### RegulatoryTCell (Tolerance)

Prevents over-aggressive surveillance:
- Suppression rules (temporary tolerance after updates, known variance, etc.)
- Auto-tolerance for stable agents (100+ clean inspections)
- Prevents autoimmunity (shutting down healthy agents)

#### ImmuneMemory

Persistent storage of threat patterns:
- Stores threat signatures with violation types, hashes, response that worked
- Recall accelerates response on re-encounter
- Export/import for sharing between systems
- Pruning of old memories

#### ImmuneSystem (Complete Organelle)

Integrates all surveillance components:
- Agent registration with canary tests
- Training workflow
- Inspection flow: Memory recall → T-cell inspection → Treg filtering
- Tolerance management

### Key Mechanisms

1. **Thymic selection** — Learn baseline before deployment
2. **Two-signal activation** — Prevent false positives
3. **Regulatory tolerance** — Prevent autoimmunity
4. **Immune memory** — Faster response on re-encounter
5. **Canary tests** — Known-answer probes for compromise detection

---

## System 3: Coordination (Cell Cycle)

### Biological Model

Cell cycle checkpoints prevent cells from progressing when conditions aren't met:
- **G1 checkpoint** — Verify resources before committing
- **S checkpoint** — Verify progress during execution
- **G2 checkpoint** — Verify outputs before completing
- **M checkpoint** — Final commit-or-abort
- **Apoptosis** — Self-terminate if truly stuck

### Components

#### Phase (Cell Cycle)

```python
class Phase(Enum):
    G0 = "g0"    # Quiescent
    G1 = "g1"    # Acquiring resources
    S = "s"      # Executing
    G2 = "g2"    # Validating
    M = "m"      # Committing
```

#### ResourceLock

Lockable resource with ownership and priority:
- Try-acquire with optional preemption
- Waiting list sorted by priority
- Tracks hold duration

#### DependencyGraph

Tracks wait-for relationships:
- Detects deadlock cycles (DFS)
- Returns blocking chain for debugging

#### CellCycleController

Manages agent operations through checkpoints:
- Phase timeouts
- Checkpoint conditions per phase
- Resource acquisition with dependency tracking
- Deadlock detection

#### Watchdog (Apoptosis)

Terminates stuck operations:
- Max operation time
- Starvation detection (stuck waiting for resources)
- Progress timeout (no progress in S phase)
- Deadlock breaking (kill lowest priority or oldest in cycle)

#### PriorityInheritance

Prevents priority inversion:
- When high-priority waiter blocked by low-priority holder
- Holder temporarily inherits higher priority
- Original priority restored on release

#### CoordinationSystem (Complete Organelle)

Integrates all coordination components:
- Unified execution through all phases
- Automatic resource management
- Watchdog integration
- Custom checkpoint conditions

### Key Mechanisms

1. **Checkpoints** — Gates between phases
2. **Deadlock detection** — Cycle detection in dependency graph
3. **Watchdog termination** — Kill stuck operations
4. **Priority inheritance** — Prevent inversion
5. **Preemption** — High-priority can take resources from low-priority

---

## Integration: IntegratedCell

### Cross-System Interactions

| Event | Source System | Target System | Action |
|-------|---------------|---------------|--------|
| Low confidence output | Quality | Surveillance | Record anomaly, potentially trigger Signal 2 |
| Byzantine agent detected | Surveillance | Coordination | Lower agent priority |
| Agent terminated | Coordination | Quality | Recycle all agent's tags |
| Deadlock detected | Coordination | Surveillance | Flag involved agents |
| Threat pattern recalled | Surveillance | Quality | Pre-tag with low confidence |
| Canary failed | Surveillance | Coordination | Force-kill operation |

### Unified Execution Flow

1. **Pre-execution**: Surveillance checks agent health, adjusts priority
2. **Resource acquisition**: Coordination manages locks, detects deadlocks
3. **Execution**: Operation runs with provenance tracking
4. **Post-execution**: Quality validates output, surveillance records behavior
5. **Cleanup**: Coordination releases resources, tags recycled if needed

### CellExecutionResult

```python
@dataclass
class CellExecutionResult:
    agent_id: str
    success: bool
    output: Optional[TaggedData]
    degradation_result: Optional[DegradationResult]
    blocked_by: Optional[str]  # "surveillance", "coordination", "quality"
    coordination_result: Optional[CoordinationResult]
    immune_response: Optional[ImmuneResponse]
    execution_time: Optional[timedelta]
```

### CellHealth

```python
@dataclass
class CellHealth:
    healthy: bool
    surveillance_alerts: list[ImmuneResponse]
    apoptosis_events: list[ApoptosisEvent]
    pool_status: dict
    coordination_stats: dict
```

---

## Implementation Plan

### Phase 1: Quality System
1. Implement `UbiquitinTag` and `TaggedData`
2. Implement `UbiquitinPool` with exhaustion strategies
3. Implement `E3Ligase`, `Deubiquitinase`, `ChaperoneRepair`
4. Implement `Proteasome` with full inspection flow
5. Implement `ProvenancePolicy` and `ProvenanceConnection`
6. Extend `WiringDiagram` to `ProvenanceWiringDiagram`
7. Implement `ProvenanceRuntime`

### Phase 2: Surveillance System
1. Implement `MHCPeptide` and `MHCDisplay`
2. Implement `Thymus` and `BaselineProfile`
3. Implement `TCell` with two-signal activation
4. Implement `RegulatoryTCell` with suppression rules
5. Implement `ImmuneMemory` with export/import
6. Implement `ImmuneSystem` organelle

### Phase 3: Coordination System
1. Implement `ResourceLock` and `DependencyGraph`
2. Implement `CellCycleController` with checkpoints
3. Implement `Watchdog` with apoptosis triggers
4. Implement `PriorityInheritance`
5. Implement `CoordinationSystem` organelle

### Phase 4: Integration
1. Implement `IntegratedCell`
2. Wire cross-system callbacks
3. Implement unified execution flow
4. Implement health check
5. Add examples and tests

---

## Open Questions

1. **Persistence**: Should immune memory persist across restarts?
2. **Distribution**: How do these systems work in multi-node deployments?
3. **Metrics**: What observability/metrics should be exposed?
4. **Configuration**: How should thresholds be tuned for different use cases?

---

## References

- Ubiquitin-proteasome system: https://en.wikipedia.org/wiki/Proteasome
- MHC and immune recognition: https://en.wikipedia.org/wiki/Major_histocompatibility_complex
- Cell cycle checkpoints: https://en.wikipedia.org/wiki/Cell_cycle_checkpoint
- Priority inheritance: https://en.wikipedia.org/wiki/Priority_inheritance
