"""
Diagram Optimizer: Rewriting Rules as Endofunctors
===================================================

Biological Analogy:
Metabolic pathway rewiring — cells optimize flux through enzyme
regulation, allosteric control, and pathway shortcuts. Each
optimization pass is an endofunctor on the category of wiring
diagrams that preserves input-output behavior (bisimulation).

This module transforms a WiringDiagram into an equivalent but
more efficient OptimizedDiagram via composable rewriting passes.

References:
- Article Section 7: Diagram Optimization via Categorical Rewriting
- Abbott & Zardini (2025): Category-theoretic diagram optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .analyzer import dependency_graph, find_dead_wires, find_independent_groups
from .wagent import ResourceCost, Wire, WiringDiagram


@dataclass
class OptimizedDiagram:
    """A WiringDiagram with optimization metadata.

    Wraps the original diagram with pre-computed parallel groups
    and an execution schedule for resource-aware execution.
    """

    diagram: WiringDiagram
    parallel_groups: list[set[str]] = field(default_factory=list)
    schedule: list[str] = field(default_factory=list)
    eliminated_wires: list[Wire] = field(default_factory=list)
    passes_applied: list[str] = field(default_factory=list)


@runtime_checkable
class OptimizationPass(Protocol):
    """Protocol for diagram rewriting passes.

    Each pass is an endofunctor on WiringDiagram that preserves
    input-output behavior while improving resource utilization.
    """

    @property
    def name(self) -> str: ...

    def apply(self, diagram: WiringDiagram) -> WiringDiagram: ...


@dataclass(frozen=True)
class EliminateDeadWires:
    """Remove wires where prism analysis proves they never transmit.

    Biological Analogy: Pruning vestigial metabolic pathways —
    enzymes that never encounter their substrate are downregulated.
    """

    @property
    def name(self) -> str:
        return "eliminate_dead_wires"

    def apply(self, diagram: WiringDiagram) -> WiringDiagram:
        dead = set(id(w) for w in find_dead_wires(diagram))
        if not dead:
            return diagram
        result = WiringDiagram(
            modules=dict(diagram.modules),
            wires=[w for w in diagram.wires if id(w) not in dead],
        )
        return result


@dataclass(frozen=True)
class ParallelGrouping:
    """Annotate independent module groups for concurrent execution.

    This pass does not modify the diagram structure; it computes
    parallel groups that the executor can use for concurrency.

    Biological Analogy: Identifying independent metabolic pathways
    that can operate simultaneously (e.g., glycolysis and fatty acid
    oxidation in different cellular compartments).
    """

    @property
    def name(self) -> str:
        return "parallel_grouping"

    def apply(self, diagram: WiringDiagram) -> WiringDiagram:
        # Structure-preserving: no modification needed
        return diagram


@dataclass(frozen=True)
class CostOrderSchedule:
    """Among equally-ready modules, prefer cheaper ones first.

    Biological Analogy: Enzyme kinetics optimization — cells
    preferentially activate low-cost pathways (e.g., glycolysis
    before oxidative phosphorylation) to minimize ATP investment
    before committing to expensive operations.
    """

    @property
    def name(self) -> str:
        return "cost_order_schedule"

    def apply(self, diagram: WiringDiagram) -> WiringDiagram:
        # Structure-preserving: scheduling is handled at execution time
        return diagram


def _build_schedule(diagram: WiringDiagram) -> list[str]:
    """Build a cost-ordered topological schedule.

    Within each topological layer, modules are sorted by ascending
    ATP cost so cheaper modules execute first.
    """
    deps = dependency_graph(diagram)
    remaining = {name: set(preds) for name, preds in deps.items()}
    schedule: list[str] = []
    resolved: set[str] = set()

    while remaining:
        ready = [name for name, preds in remaining.items() if preds <= resolved]
        if not ready:
            # Cycle — just append remaining
            schedule.extend(sorted(remaining.keys()))
            break

        # Sort by cost (ascending) so cheaper modules execute first
        def _cost_key(name: str) -> float:
            spec = diagram.modules[name]
            if spec.cost is not None:
                return spec.cost.atp + spec.cost.latency_ms
            return 0.0

        ready.sort(key=_cost_key)
        schedule.extend(ready)
        resolved |= set(ready)
        for name in ready:
            del remaining[name]

    return schedule


def optimize(
    diagram: WiringDiagram,
    passes: list[OptimizationPass] | None = None,
) -> OptimizedDiagram:
    """Apply optimization passes in sequence.

    If no passes are specified, applies all default passes:
    EliminateDeadWires, ParallelGrouping, CostOrderSchedule.
    """
    if passes is None:
        passes = [EliminateDeadWires(), ParallelGrouping(), CostOrderSchedule()]

    current = diagram
    eliminated: list[Wire] = []
    applied: list[str] = []

    for p in passes:
        before_wires = set(id(w) for w in current.wires)
        current = p.apply(current)
        after_wires = set(id(w) for w in current.wires)
        # Track eliminated wires
        for w in diagram.wires:
            if id(w) in before_wires and id(w) not in after_wires:
                eliminated.append(w)
        applied.append(p.name)

    parallel_groups = find_independent_groups(current)
    schedule = _build_schedule(current)

    return OptimizedDiagram(
        diagram=current,
        parallel_groups=parallel_groups,
        schedule=schedule,
        eliminated_wires=eliminated,
        passes_applied=applied,
    )
