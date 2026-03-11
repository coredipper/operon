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
- Article Section 6.2: Epistemic Topology of Multi-Cellular Agents
- Theorems 1-4: Error amplification, sequential penalty, parallel
  speedup, tool density scaling
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
    """Classification of wiring diagram topology."""

    INDEPENDENT = "independent"
    CENTRALIZED = "centralized"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"


# ── Data Types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObservationProfile:
    """Per-module epistemic observation profile."""

    module_name: str
    direct_sources: frozenset[str]
    transitive_sources: frozenset[str]
    input_wires: tuple[Wire, ...]
    has_optic_filter: bool
    has_denature_filter: bool

    @property
    def observation_width(self) -> int:
        return len(self.transitive_sources)


@dataclass(frozen=True)
class EpistemicPartition:
    """Groups modules by observation equivalence."""

    equivalence_classes: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class TopologyClassification:
    """Result of classifying a diagram's topology."""

    topology_class: TopologyClass
    hub_module: str | None
    chain_length: int
    parallelism_width: int
    num_sources: int


@dataclass(frozen=True)
class ErrorAmplificationBound:
    """Theorem 1: Error amplification bounds."""

    n_agents: int
    independent_bound: int
    centralized_bound: float
    detection_rate: float
    amplification_ratio: float


@dataclass(frozen=True)
class SequentialPenalty:
    """Theorem 2: Sequential communication overhead."""

    chain_length: int
    num_handoffs: int
    comm_cost_ratio: float
    overhead_ratio: float


@dataclass(frozen=True)
class ParallelSpeedup:
    """Theorem 3: Parallel speedup from independent groups."""

    num_subtasks: int
    speedup: float
    max_layer_cost: ResourceCost
    total_cost: ResourceCost


@dataclass(frozen=True)
class ToolDensityAnalysis:
    """Theorem 4: Tool density scaling."""

    total_tools: int
    num_modules: int
    tools_per_module: float
    remote_fraction: float
    planning_cost_ratio: float


@dataclass(frozen=True)
class TopologyRecommendation:
    """Recommendation for optimal topology given constraints."""

    recommended: TopologyClass
    rationale: str


@dataclass(frozen=True)
class EpistemicAnalysis:
    """Complete epistemic analysis of a wiring diagram."""

    profiles: dict[str, ObservationProfile]
    partition: EpistemicPartition
    classification: TopologyClassification
    error_bound: ErrorAmplificationBound
    sequential: SequentialPenalty
    speedup: ParallelSpeedup
    density: ToolDensityAnalysis


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
    """Compute per-module epistemic observation profiles."""
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
    """Classify the diagram's topology."""
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
    """Theorem 1: Error amplification bounds from topology."""
    deps = dependency_graph(diagram)
    # Non-source modules (workers that receive input)
    n = sum(1 for preds in deps.values() if len(preds) > 0)
    n = max(n, 1)  # Avoid zero division for single-module diagrams
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
    """Theorem 2: Sequential communication overhead."""
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
    """Theorem 3: Parallel speedup from independent groups."""
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
    """Theorem 4: Tool density scaling."""
    all_caps: set = set()
    modules_with_caps = 0
    for spec in diagram.modules.values():
        if spec.capabilities:
            all_caps |= spec.capabilities
            modules_with_caps += 1
    n = max(modules_with_caps, 1)
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
    """Recommend optimal topology given constraints (no diagram needed)."""
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
    """Complete epistemic analysis of a wiring diagram."""
    return EpistemicAnalysis(
        profiles=observation_profiles(diagram),
        partition=epistemic_partition(diagram),
        classification=classify_topology(diagram),
        error_bound=error_amplification_bound(diagram, detection_rate=detection_rate),
        sequential=sequential_penalty(diagram, comm_cost_ratio=comm_cost_ratio),
        speedup=parallel_speedup(diagram),
        density=tool_density(diagram),
    )
