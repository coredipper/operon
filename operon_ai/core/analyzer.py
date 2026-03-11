"""
Diagram Analyzer: Static Analysis of Wiring Diagrams
=====================================================

Biological Analogy:
Metabolic pathway analysis — cells don't blindly run pathways,
they identify bottlenecks (rate-limiting enzymes), dead-end
metabolites, and parallelizable branches. Flux Balance Analysis
is the biological precedent for diagram optimization.

This module provides static analysis of WiringDiagram structure
to identify optimization opportunities before execution.

References:
- Article Section 7: Diagram Optimization via Categorical Rewriting
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .optics import PrismOptic
from .wagent import ResourceCost, Wire, WiringDiagram


@dataclass
class Optimization:
    """A human-readable optimization opportunity."""

    kind: str  # "parallel", "dead_wire", "cost_hotspot"
    description: str
    affected_modules: list[str] = field(default_factory=list)
    affected_wires: list[Wire] = field(default_factory=list)
    estimated_saving: ResourceCost | None = None


def dependency_graph(diagram: WiringDiagram) -> dict[str, set[str]]:
    """Build adjacency list of module dependencies.

    Returns a dict mapping each module name to the set of modules
    it depends on (i.e., modules that must execute before it).
    """
    deps: dict[str, set[str]] = {name: set() for name in diagram.modules}
    for wire in diagram.wires:
        deps[wire.dst_module].add(wire.src_module)
    return deps


def _reverse_graph(deps: dict[str, set[str]]) -> dict[str, set[str]]:
    """Build reverse adjacency list (dependents of each module)."""
    rev: dict[str, set[str]] = {name: set() for name in deps}
    for module, predecessors in deps.items():
        for pred in predecessors:
            rev[pred].add(module)
    return rev


def find_independent_groups(diagram: WiringDiagram) -> list[set[str]]:
    """Find sets of modules with no mutual dependencies (parallelizable).

    Uses topological layering: modules in the same layer have all
    their dependencies satisfied by previous layers and can execute
    concurrently.
    """
    deps = dependency_graph(diagram)
    remaining = {name: set(preds) for name, preds in deps.items()}
    groups: list[set[str]] = []
    resolved: set[str] = set()

    while remaining:
        # Find all modules whose dependencies are fully resolved
        ready = {name for name, preds in remaining.items() if preds <= resolved}
        if not ready:
            # Cycle detected — put all remaining in one group
            groups.append(set(remaining.keys()))
            break
        groups.append(ready)
        resolved |= ready
        for name in ready:
            del remaining[name]

    return groups


def find_dead_wires(diagram: WiringDiagram) -> list[Wire]:
    """Find wires whose prism optics can never accept any output type from the source module.

    A wire is "dead" if it carries a PrismOptic whose accepted types
    have no intersection with the source module's output port DataType.
    """
    dead: list[Wire] = []
    for wire in diagram.wires:
        if not isinstance(wire.optic, PrismOptic):
            continue
        src_module = diagram.modules.get(wire.src_module)
        if src_module is None:
            continue
        src_port_type = src_module.outputs.get(wire.src_port)
        if src_port_type is None:
            continue
        # The source port has a fixed DataType; if the prism doesn't accept it, wire is dead
        if src_port_type.data_type not in wire.optic.accept:
            dead.append(wire)
    return dead


def critical_path(diagram: WiringDiagram) -> tuple[list[str], ResourceCost]:
    """Find the longest cost-weighted path through the DAG.

    Returns (path_as_module_names, total_cost_along_path).
    Cost = sum of module costs + wire costs along the path.
    Uses dynamic programming on topological order.
    """
    deps = dependency_graph(diagram)

    def _module_cost(name: str) -> ResourceCost:
        spec = diagram.modules[name]
        return spec.cost if spec.cost is not None else ResourceCost()

    # Build wire cost lookup: (src, dst) -> total wire cost between them
    wire_costs: dict[tuple[str, str], int] = {}
    for wire in diagram.wires:
        key = (wire.src_module, wire.dst_module)
        wire_costs[key] = wire_costs.get(key, 0) + wire.cost

    # Topological sort via Kahn's algorithm
    in_degree = {name: len(preds) for name, preds in deps.items()}
    queue = [name for name, deg in in_degree.items() if deg == 0]
    topo_order: list[str] = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for name, preds in deps.items():
            if node in preds:
                in_degree[name] -= 1
                if in_degree[name] == 0:
                    queue.append(name)

    # DP: longest path ending at each node
    dist: dict[str, ResourceCost] = {}
    prev: dict[str, str | None] = {}

    for name in topo_order:
        mc = _module_cost(name)
        best_cost = ResourceCost()
        best_prev: str | None = None

        for pred in deps[name]:
            wc = ResourceCost(atp=wire_costs.get((pred, name), 0))
            candidate = dist[pred] + wc
            if candidate.atp + candidate.latency_ms > best_cost.atp + best_cost.latency_ms:
                best_cost = candidate
                best_prev = pred

        dist[name] = best_cost + mc
        prev[name] = best_prev

    if not dist:
        return [], ResourceCost()

    # Find the node with maximum cost
    end_node = max(dist, key=lambda n: dist[n].atp + dist[n].latency_ms)

    # Reconstruct path
    path: list[str] = []
    node: str | None = end_node
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path, dist[end_node]


def total_cost(diagram: WiringDiagram) -> ResourceCost:
    """Sum of all module + wire costs in the diagram."""
    result = ResourceCost()
    for spec in diagram.modules.values():
        if spec.cost is not None:
            result = result + spec.cost
    for wire in diagram.wires:
        result = result + ResourceCost(atp=wire.cost)
    return result


def suggest_optimizations(diagram: WiringDiagram) -> list[Optimization]:
    """Return human-readable optimization opportunities."""
    suggestions: list[Optimization] = []

    # 1. Parallelizable groups
    groups = find_independent_groups(diagram)
    parallel_groups = [g for g in groups if len(g) > 1]
    for group in parallel_groups:
        suggestions.append(
            Optimization(
                kind="parallel",
                description=f"Modules {sorted(group)} can execute concurrently",
                affected_modules=sorted(group),
            )
        )

    # 2. Dead wires
    dead = find_dead_wires(diagram)
    for wire in dead:
        suggestions.append(
            Optimization(
                kind="dead_wire",
                description=(
                    f"Wire {wire.src_module}.{wire.src_port} -> "
                    f"{wire.dst_module}.{wire.dst_port} can never transmit "
                    f"(prism rejects source DataType)"
                ),
                affected_wires=[wire],
            )
        )

    # 3. Cost hotspots (modules > 50% of total cost)
    tc = total_cost(diagram)
    if tc.atp > 0:
        for name, spec in diagram.modules.items():
            if spec.cost is not None and spec.cost.atp > tc.atp * 0.5:
                suggestions.append(
                    Optimization(
                        kind="cost_hotspot",
                        description=(
                            f"Module '{name}' consumes {spec.cost.atp} ATP "
                            f"({spec.cost.atp * 100 // tc.atp}% of total)"
                        ),
                        affected_modules=[name],
                    )
                )

    return suggestions
