"""
Epistemic Analyzer: Knowledge Operators from Wiring Topology
============================================================

Biological Analogy:
Cells in a tissue don't all see the same signals — each cell's
"knowledge" of the tissue state depends on which morphogen gradients
reach it. This module derives Kripke-style observation profiles from
wiring diagram structure to predict error amplification, coordination
overhead, and parallelism bounds.

References:
- Article Section 6.5.4: Epistemic Topology of Wiring Diagrams
- Theorems 1-4: Error amplification, sequential penalty, parallel
  speedup, tool density scaling

These outputs are simplified architecture-level predictions derived
from diagram structure. They are not benchmark-fitted estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .analyzer import (
    critical_path,
    dependency_graph,
    find_independent_groups,
    total_cost,
)
from .wagent import ResourceCost, Wire, WiringDiagram


# ── Enums ──────────────────────────────────────────────────────────


class TopologyClass(Enum):
    """Classification of wiring diagram topology.

    Biological Analogy:
    Tissue architecture — independent cells (epithelial sheet),
    centralized hub (nervous system star topology), sequential
    pipeline (digestive tract), or hybrid (immune system with
    both local and centralized coordination).
    """

    INDEPENDENT = "independent"   # No wires: fully parallel, no coordination
    CENTRALIZED = "centralized"   # Fan-in: one hub aggregates all sources
    SEQUENTIAL = "sequential"     # Linear chain: each module feeds the next
    HYBRID = "hybrid"             # Mixed: fan-out/fan-in with parallelism


# ── Data Types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObservationProfile:
    """Per-module epistemic observation profile.

    Biological Analogy:
    A cell's receptor profile — which signals it can detect and
    from how far away. A cell with many receptor types has a wide
    observation profile; one with few receptors is epistemically
    narrow.

    In Kripke semantics, this is the accessibility relation for
    a single agent: which worlds (module states) it can distinguish.
    """

    module_name: str                       # Module this profile describes
    direct_sources: frozenset[str]         # Immediate predecessors (one hop)
    transitive_sources: frozenset[str]     # All reachable predecessors (any path)
    input_wires: tuple[Wire, ...]          # Actual wire objects carrying data in
    has_optic_filter: bool                 # True if any input wire has an optic
    has_denature_filter: bool              # True if any input wire has a denature filter

    @property
    def observation_width(self) -> int:
        """Number of transitively observable modules (epistemic reach)."""
        return len(self.transitive_sources)


@dataclass(frozen=True)
class EpistemicPartition:
    """Groups modules by observation equivalence.

    Two modules are observation-equivalent if they see the same
    transitive sources through the same filter types. Equivalent
    modules cannot distinguish between worlds that differ only
    outside their shared observation set.

    Biological Analogy:
    Cell types defined by shared receptor profiles — two cells
    with identical receptor sets respond to the same signals
    and are functionally equivalent for signal processing.
    """

    equivalence_classes: tuple[frozenset[str], ...]  # Groups of equivalent modules


@dataclass(frozen=True)
class TopologyClassification:
    """Result of classifying a diagram's wiring topology.

    Classification rules (checked in order):
    - INDEPENDENT: no wires at all
    - SEQUENTIAL: critical path spans all modules
    - CENTRALIZED: one hub with all other modules as sources (fan-in)
    - HYBRID: everything else (fan-out + fan-in, mixed patterns)
    """

    topology_class: TopologyClass  # The classified topology type
    hub_module: str | None         # Central module name if CENTRALIZED/HYBRID, else None
    chain_length: int              # Length of the critical path (longest module chain)
    parallelism_width: int         # Max modules executable in parallel (widest layer)
    num_sources: int               # Modules with no input wires (entry points)


@dataclass(frozen=True)
class ErrorAmplificationBound:
    """Theorem 1: Error amplification bounds.

    Predicts worst-case error propagation through the wiring topology.
    Independent agents each fail independently (bound = n); a centralized
    hub with detection rate d catches errors before propagation
    (bound = n * (1 - d)).

    Biological Analogy:
    Immune error detection — without a thymus (centralized checker),
    each T-cell can independently produce autoimmune errors. With
    central tolerance checking, error rate drops by detection fraction.
    """

    n_agents: int               # Number of error-producing modules (workers / isolated agents)
    independent_bound: int      # Worst-case errors without central detection (= n)
    centralized_bound: float    # Worst-case errors with central detection (= n*(1-d))
    detection_rate: float       # Fraction of errors caught by hub (d)
    amplification_ratio: float  # Independent / centralized ratio (= 1/(1-d))


@dataclass(frozen=True)
class SequentialPenalty:
    """Theorem 2: Sequential communication overhead.

    Each handoff along the critical path adds communication cost.
    Overhead ratio = (handoffs * comm_cost_ratio) / chain_length,
    representing the fraction of total path cost spent on
    inter-module communication rather than useful work.

    Biological Analogy:
    Signal transduction cascades — each kinase→kinase handoff
    in a MAPK cascade has latency and fidelity cost. Longer
    cascades amplify signal but accumulate handoff overhead.
    """

    chain_length: int        # Number of modules on the critical path (k)
    num_handoffs: int        # Number of edges on the critical path (k - 1)
    comm_cost_ratio: float   # Per-handoff cost as fraction of module cost
    overhead_ratio: float    # Total communication overhead fraction


@dataclass(frozen=True)
class ParallelSpeedup:
    """Theorem 3: Parallel speedup from independent groups.

    Amdahl's-law-style bound: speedup = total_work / bottleneck_path.
    The bottleneck path is the sum of the most expensive module in
    each topological layer (since modules within a layer run in parallel,
    only the slowest matters).

    Biological Analogy:
    Parallel metabolic pathways — glycolysis and the pentose phosphate
    pathway run concurrently, but total throughput is limited by the
    slowest shared step (bottleneck enzyme).
    """

    num_subtasks: int            # Total number of modules in the diagram
    speedup: float               # total_cost / bottleneck_cost (>= 1.0)
    max_layer_cost: ResourceCost # Cost of the single most expensive layer
    total_cost: ResourceCost     # Sum of all module costs


@dataclass(frozen=True)
class ToolDensityAnalysis:
    """Theorem 4: Tool density scaling.

    Distributing capabilities across modules increases coordination
    cost: each module must plan across remote tools it cannot directly
    invoke. Planning cost scales linearly with the number of modules
    that hold capabilities.

    Biological Analogy:
    Organ specialization — a single-celled organism has all enzymes
    locally (no coordination cost). A multicellular organism distributes
    enzymes across organs, requiring hormonal signaling (planning cost)
    to coordinate metabolic activity.
    """

    total_tools: int           # Distinct Capability values across all modules
    num_modules: int           # Modules that have at least one capability
    tools_per_module: float    # Average tools per capability-bearing module
    remote_fraction: float     # Fraction of tools not locally available ((n-1)/n)
    planning_cost_ratio: float # Multi-agent planning cost vs single-agent (= n)


@dataclass(frozen=True)
class TopologyRecommendation:
    """Recommendation for optimal topology given task constraints.

    Standalone result from ``recommend_topology()`` — no diagram needed,
    just task properties (subtask count, independence, tool count,
    error tolerance).
    """

    recommended: TopologyClass  # Suggested topology type
    rationale: str              # Human-readable explanation of the recommendation


@dataclass(frozen=True)
class EpistemicAnalysis:
    """Complete epistemic analysis bundle from ``analyze()``.

    Combines all epistemic properties and theorem predictions
    computed from a single ``WiringDiagram`` in one pass.
    """

    profiles: dict[str, ObservationProfile]   # Per-module observation profiles
    partition: EpistemicPartition              # Observation equivalence classes
    classification: TopologyClassification    # Topology type and structural metrics
    error_bound: ErrorAmplificationBound      # Theorem 1: error amplification
    sequential: SequentialPenalty              # Theorem 2: sequential overhead
    speedup: ParallelSpeedup                  # Theorem 3: parallel speedup
    density: ToolDensityAnalysis              # Theorem 4: tool density


# ── Internal helpers ───────────────────────────────────────────────


def _transitive_closure(deps: dict[str, set[str]]) -> dict[str, frozenset[str]]:
    """Compute transitive predecessors for each node via iterative expansion."""
    closure: dict[str, set[str]] = {name: set(preds) for name, preds in deps.items()}
    changed = True
    while changed:
        changed = False
        for name in closure:
            expanded = set()
            for pred in closure[name]:
                expanded |= closure.get(pred, set())
            if not expanded <= closure[name]:
                closure[name] |= expanded
                changed = True
    return {name: frozenset(preds) for name, preds in closure.items()}


# ── Public API ─────────────────────────────────────────────────────


def observation_profiles(diagram: WiringDiagram) -> dict[str, ObservationProfile]:
    """Compute per-module epistemic observation profiles.

    For each module, determines which other modules it can observe
    (directly and transitively) and what filters exist on its input
    wires. This is the foundation for all other epistemic analysis.

    Returns a dict mapping module name -> ObservationProfile.
    """
    deps = dependency_graph(diagram)
    closure = _transitive_closure(deps)

    # Collect input wires per module
    wires_by_dst: dict[str, list[Wire]] = {name: [] for name in diagram.modules}
    for wire in diagram.wires:
        wires_by_dst[wire.dst_module].append(wire)

    profiles: dict[str, ObservationProfile] = {}
    for name in diagram.modules:
        input_wires = tuple(wires_by_dst[name])
        has_optic = any(w.optic is not None for w in input_wires)
        has_denature = any(w.denature is not None for w in input_wires)
        profiles[name] = ObservationProfile(
            module_name=name,
            direct_sources=frozenset(deps[name]),
            transitive_sources=closure[name],
            input_wires=input_wires,
            has_optic_filter=has_optic,
            has_denature_filter=has_denature,
        )
    return profiles


def epistemic_partition(diagram: WiringDiagram) -> EpistemicPartition:
    """Group modules by observation equivalence.

    Partition key = (sorted transitive sources, optic flag, denature flag).
    """
    profiles = observation_profiles(diagram)
    groups: dict[tuple, set[str]] = {}
    for name, prof in profiles.items():
        key = (prof.transitive_sources, prof.has_optic_filter, prof.has_denature_filter)
        groups.setdefault(key, set()).add(name)
    classes = tuple(frozenset(g) for g in groups.values())
    return EpistemicPartition(equivalence_classes=classes)


def classify_topology(diagram: WiringDiagram) -> TopologyClassification:
    """Classify the diagram's wiring topology into one of four classes.

    Classification rules (checked in priority order):
    1. No wires → INDEPENDENT
    2. Critical path spans all modules → SEQUENTIAL
    3. One hub with all others as sources (pure fan-in) → CENTRALIZED
    4. Everything else → HYBRID

    Also computes structural metrics: chain length, parallelism width,
    hub module identity, and number of source modules.
    """
    n = len(diagram.modules)
    deps = dependency_graph(diagram)
    closure = _transitive_closure(deps)
    groups = find_independent_groups(diagram)
    path, _ = critical_path(diagram)

    # Sources: modules with no inputs
    num_sources = sum(1 for preds in deps.values() if len(preds) == 0)
    parallelism_width = max((len(g) for g in groups), default=0)
    chain_length = len(path)

    # Classification rules
    if len(diagram.wires) == 0:
        return TopologyClassification(
            topology_class=TopologyClass.INDEPENDENT,
            hub_module=None,
            chain_length=chain_length,
            parallelism_width=parallelism_width,
            num_sources=num_sources,
        )

    if chain_length == n:
        return TopologyClassification(
            topology_class=TopologyClass.SEQUENTIAL,
            hub_module=None,
            chain_length=chain_length,
            parallelism_width=parallelism_width,
            num_sources=num_sources,
        )

    # Hub detection: module whose transitive sources > 50% of other modules
    hub_module = None
    threshold = (n - 1) * 0.5
    for name, trans in closure.items():
        if len(trans) > threshold:
            if hub_module is None or len(trans) > len(closure[hub_module]):
                hub_module = name

    # CENTRALIZED: hub exists and all non-hub modules are sources
    # (pure fan-in with no intermediate processing)
    if hub_module is not None and num_sources == n - 1:
        return TopologyClassification(
            topology_class=TopologyClass.CENTRALIZED,
            hub_module=hub_module,
            chain_length=chain_length,
            parallelism_width=parallelism_width,
            num_sources=num_sources,
        )

    # HYBRID: everything else (fan-out + fan-in, mixed patterns)
    return TopologyClassification(
        topology_class=TopologyClass.HYBRID,
        hub_module=hub_module,
        chain_length=chain_length,
        parallelism_width=parallelism_width,
        num_sources=num_sources,
    )


def error_amplification_bound(
    diagram: WiringDiagram, *, detection_rate: float = 0.75
) -> ErrorAmplificationBound:
    """Theorem 1: Error amplification bounds from topology.

    Computes worst-case error propagation for independent vs centralized
    architectures. ``detection_rate`` (0-1) is the fraction of errors
    a centralized hub catches before they propagate downstream.

    Independent bound = n (each worker fails independently).
    Centralized bound = n * (1 - d) (hub catches fraction d).

    ``n`` counts all non-source modules -- any module that has at least
    one input wire. These are the modules that process data and can
    propagate or originate errors. Source-only modules (no inputs) are
    excluded because they are external inputs, not error-producing stages.
    """
    in_degree = {name: 0 for name in diagram.modules}
    for wire in diagram.wires:
        in_degree[wire.dst_module] += 1

    # Count all non-source modules (modules with at least one input)
    n = sum(1 for name in diagram.modules if in_degree[name] > 0)

    if n == 0:
        return ErrorAmplificationBound(
            n_agents=0,
            independent_bound=0,
            centralized_bound=0,
            detection_rate=detection_rate,
            amplification_ratio=0.0,
        )

    ind_bound = n
    cent_bound = n * (1 - detection_rate)
    ratio = 1.0 / (1 - detection_rate) if detection_rate < 1.0 else float("inf")
    return ErrorAmplificationBound(
        n_agents=n,
        independent_bound=ind_bound,
        centralized_bound=cent_bound,
        detection_rate=detection_rate,
        amplification_ratio=ratio,
    )


def sequential_penalty(
    diagram: WiringDiagram, *, comm_cost_ratio: float = 0.4
) -> SequentialPenalty:
    """Theorem 2: Sequential communication overhead.

    Measures the overhead of inter-module handoffs along the critical path.
    ``comm_cost_ratio`` (0-1) is the per-handoff communication cost as a
    fraction of average module execution cost.

    Overhead = (handoffs * comm_cost_ratio) / chain_length.
    """
    path, _ = critical_path(diagram)
    k = len(path)
    h = max(k - 1, 0)  # Handoffs = edges on critical path
    overhead = (h * comm_cost_ratio / k) if k > 0 else 0.0
    return SequentialPenalty(
        chain_length=k,
        num_handoffs=h,
        comm_cost_ratio=comm_cost_ratio,
        overhead_ratio=overhead,
    )


def parallel_speedup(diagram: WiringDiagram) -> ParallelSpeedup:
    """Theorem 3: Parallel speedup from independent groups.

    Computes speedup = total_cost / bottleneck_cost, where the
    bottleneck is the sum of the most expensive module per
    topological layer. Modules within the same layer execute
    concurrently, so only the slowest determines layer cost.

    Returns speedup >= 1.0 (1.0 means no parallelism benefit).
    """
    groups = find_independent_groups(diagram)
    tc = total_cost(diagram)

    def _layer_cost(group: set[str]) -> ResourceCost:
        """Cost of the most expensive module in a layer."""
        best = ResourceCost()
        for name in group:
            spec = diagram.modules[name]
            mc = spec.cost if spec.cost is not None else ResourceCost()
            if mc.atp + mc.latency_ms > best.atp + best.latency_ms:
                best = mc
        return best

    layer_costs = [_layer_cost(g) for g in groups]
    # Critical path through layers = sum of max-cost-per-layer
    bottleneck = ResourceCost()
    for lc in layer_costs:
        bottleneck = bottleneck + lc

    max_layer = max(layer_costs, key=lambda c: c.atp + c.latency_ms) if layer_costs else ResourceCost()
    total_scalar = tc.atp + tc.latency_ms
    bottleneck_scalar = bottleneck.atp + bottleneck.latency_ms
    sp = total_scalar / bottleneck_scalar if bottleneck_scalar > 0 else 1.0

    return ParallelSpeedup(
        num_subtasks=len(diagram.modules),
        speedup=sp,
        max_layer_cost=max_layer,
        total_cost=tc,
    )


def tool_density(diagram: WiringDiagram) -> ToolDensityAnalysis:
    """Theorem 4: Tool density scaling.

    Analyzes how capabilities (tools) are distributed across modules.
    More modules with capabilities means higher coordination cost:
    each module must plan around tools it cannot directly invoke.

    Only counts modules that have at least one Capability annotation.
    """
    all_caps: set = set()
    modules_with_caps = 0
    for spec in diagram.modules.values():
        if spec.capabilities:
            all_caps |= spec.capabilities
            modules_with_caps += 1

    if modules_with_caps == 0:
        return ToolDensityAnalysis(
            total_tools=0,
            num_modules=0,
            tools_per_module=0.0,
            remote_fraction=0.0,
            planning_cost_ratio=0.0,
        )

    n = modules_with_caps
    t = len(all_caps)
    return ToolDensityAnalysis(
        total_tools=t,
        num_modules=n,
        tools_per_module=t / n,
        remote_fraction=(n - 1) / n if n > 1 else 0.0,
        planning_cost_ratio=float(n),
    )


def recommend_topology(
    *,
    num_subtasks: int,
    subtasks_independent: bool,
    num_tools: int,
    error_tolerance: float = 0.1,
) -> TopologyRecommendation:
    """Recommend optimal topology given task constraints.

    Pure decision function — no diagram needed. Uses task properties
    to suggest the best topology class:

    - Independent subtasks + few tools → INDEPENDENT (parallel)
    - Dependent subtasks + many tools or low error tolerance → CENTRALIZED
    - Dependent subtasks → SEQUENTIAL
    - Otherwise → HYBRID
    """
    if subtasks_independent and num_tools <= num_subtasks:
        return TopologyRecommendation(
            recommended=TopologyClass.INDEPENDENT,
            rationale=(
                f"{num_subtasks} independent subtasks with {num_tools} tools "
                "favor parallel execution"
            ),
        )

    if not subtasks_independent and (num_tools > num_subtasks * 2 or error_tolerance < 0.05):
        return TopologyRecommendation(
            recommended=TopologyClass.CENTRALIZED,
            rationale=(
                f"{num_tools} tools with error tolerance {error_tolerance} "
                "favor centralized coordination for error detection"
            ),
        )

    if not subtasks_independent:
        return TopologyRecommendation(
            recommended=TopologyClass.SEQUENTIAL,
            rationale=(
                f"{num_subtasks} dependent subtasks favor sequential pipeline"
            ),
        )

    return TopologyRecommendation(
        recommended=TopologyClass.HYBRID,
        rationale="Mixed independence and tool density suggest hybrid topology",
    )


def analyze(
    diagram: WiringDiagram,
    *,
    detection_rate: float = 0.75,
    comm_cost_ratio: float = 0.4,
) -> EpistemicAnalysis:
    """Run all epistemic analyses on a wiring diagram.

    Single entry point that computes observation profiles, partition,
    topology classification, and all four theorem predictions.
    Parameters ``detection_rate`` and ``comm_cost_ratio`` are forwarded
    to the respective theorem functions.
    """
    return EpistemicAnalysis(
        profiles=observation_profiles(diagram),
        partition=epistemic_partition(diagram),
        classification=classify_topology(diagram),
        error_bound=error_amplification_bound(diagram, detection_rate=detection_rate),
        sequential=sequential_penalty(diagram, comm_cost_ratio=comm_cost_ratio),
        speedup=parallel_speedup(diagram),
        density=tool_density(diagram),
    )
