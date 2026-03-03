"""Tests for diagram analyzer — static analysis of WiringDiagram structure."""

import pytest

from operon_ai.core.analyzer import (
    Optimization,
    critical_path,
    dependency_graph,
    find_dead_wires,
    find_independent_groups,
    suggest_optimizations,
    total_cost,
)
from operon_ai.core.optics import PrismOptic
from operon_ai.core.types import DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram


def _pt(dt=DataType.JSON, il=IntegrityLabel.VALIDATED):
    return PortType(dt, il)


# ── dependency_graph ────────────────────────────────────────────────


def test_dependency_graph_linear():
    """A -> B -> C produces correct adjacency."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}))
    d.connect("A", "out", "B", "in")
    d.connect("B", "out", "C", "in")

    deps = dependency_graph(d)
    assert deps["A"] == set()
    assert deps["B"] == {"A"}
    assert deps["C"] == {"B"}


def test_dependency_graph_fan_out():
    """A fans out to B and C."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")

    deps = dependency_graph(d)
    assert deps["B"] == {"A"}
    assert deps["C"] == {"A"}


def test_dependency_graph_no_wires():
    """Isolated modules have no dependencies."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("X"))
    d.add_module(ModuleSpec("Y"))
    deps = dependency_graph(d)
    assert deps == {"X": set(), "Y": set()}


# ── find_independent_groups ─────────────────────────────────────────


def test_independent_groups_linear():
    """Linear chain: each module is its own group."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}))
    d.connect("A", "out", "B", "in")
    d.connect("B", "out", "C", "in")

    groups = find_independent_groups(d)
    assert len(groups) == 3
    assert groups[0] == {"A"}
    assert groups[1] == {"B"}
    assert groups[2] == {"C"}


def test_independent_groups_parallel():
    """A -> {B, C} -> D: B and C form a parallel group."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}, outputs={"out": _pt()}))
    d.add_module(ModuleSpec("D", inputs={"i1": _pt(), "i2": _pt()}))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")
    d.connect("B", "out", "D", "i1")
    d.connect("C", "out", "D", "i2")

    groups = find_independent_groups(d)
    assert groups[0] == {"A"}
    assert {"B", "C"} in groups
    assert groups[-1] == {"D"}


def test_independent_groups_isolated():
    """Isolated modules are all in the first group."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("X"))
    d.add_module(ModuleSpec("Y"))
    d.add_module(ModuleSpec("Z"))

    groups = find_independent_groups(d)
    assert len(groups) == 1
    assert groups[0] == {"X", "Y", "Z"}


# ── find_dead_wires ─────────────────────────────────────────────────


def test_dead_wire_prism_mismatch():
    """A prism that doesn't accept the source port's DataType is dead."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt(DataType.ERROR)}))
    # Prism accepts ERROR but source outputs JSON -> dead wire
    prism = PrismOptic(accept=frozenset({DataType.ERROR}))
    d.connect("A", "out", "B", "in", optic=prism)

    dead = find_dead_wires(d)
    assert len(dead) == 1
    assert dead[0].src_module == "A"
    assert dead[0].dst_module == "B"


def test_live_wire_prism_match():
    """A prism that accepts the source port's DataType is NOT dead."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt(DataType.JSON)}))
    prism = PrismOptic(accept=frozenset({DataType.JSON}))
    d.connect("A", "out", "B", "in", optic=prism)

    dead = find_dead_wires(d)
    assert dead == []


def test_no_optic_wire_not_dead():
    """Wires without optics are never dead."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.connect("A", "out", "B", "in")

    dead = find_dead_wires(d)
    assert dead == []


# ── critical_path ───────────────────────────────────────────────────


def test_critical_path_linear():
    """Linear chain: critical path is the entire chain."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}, cost=ResourceCost(atp=10)))
    d.add_module(
        ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=20))
    )
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}, cost=ResourceCost(atp=5)))
    d.connect("A", "out", "B", "in")
    d.connect("B", "out", "C", "in")

    path, cost = critical_path(d)
    assert path == ["A", "B", "C"]
    assert cost.atp == 35


def test_critical_path_parallel_chooses_heavier():
    """With parallel branches, critical path follows the heavier one."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}, cost=ResourceCost(atp=1)))
    d.add_module(
        ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=100))
    )
    d.add_module(
        ModuleSpec("C", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=5))
    )
    d.add_module(ModuleSpec("D", inputs={"i1": _pt(), "i2": _pt()}, cost=ResourceCost(atp=1)))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")
    d.connect("B", "out", "D", "i1")
    d.connect("C", "out", "D", "i2")

    path, cost = critical_path(d)
    assert "B" in path
    assert cost.atp >= 102  # A(1) + B(100) + D(1)


def test_critical_path_with_wire_costs():
    """Wire costs are included in critical path calculation."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}, cost=ResourceCost(atp=5)))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, cost=ResourceCost(atp=5)))
    d.wires = []  # Clear for manual wire with cost
    from operon_ai.core.wagent import Wire

    d.wires.append(Wire("A", "out", "B", "in", cost=10))

    path, cost = critical_path(d)
    assert path == ["A", "B"]
    assert cost.atp == 20  # A(5) + wire(10) + B(5)


# ── total_cost ──────────────────────────────────────────────────────


def test_total_cost():
    """Total cost sums all module and wire costs."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}, cost=ResourceCost(atp=10, latency_ms=5.0)))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, cost=ResourceCost(atp=20, latency_ms=10.0)))
    from operon_ai.core.wagent import Wire

    d.wires = [Wire("A", "out", "B", "in", cost=3)]

    tc = total_cost(d)
    assert tc.atp == 33
    assert tc.latency_ms == 15.0


def test_total_cost_no_annotations():
    """Modules without cost annotations contribute zero."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.connect("A", "out", "B", "in")

    tc = total_cost(d)
    assert tc.atp == 0


# ── suggest_optimizations ──────────────────────────────────────────


def test_suggest_parallelism():
    """Parallel modules produce a 'parallel' suggestion."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")

    suggestions = suggest_optimizations(d)
    parallel = [s for s in suggestions if s.kind == "parallel"]
    assert len(parallel) >= 1
    assert set(parallel[0].affected_modules) == {"B", "C"}


def test_suggest_dead_wire():
    """Dead wires produce a 'dead_wire' suggestion."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt(DataType.ERROR)}))
    prism = PrismOptic(accept=frozenset({DataType.ERROR}))
    d.connect("A", "out", "B", "in", optic=prism)

    suggestions = suggest_optimizations(d)
    dead = [s for s in suggestions if s.kind == "dead_wire"]
    assert len(dead) == 1


def test_suggest_cost_hotspot():
    """A module consuming >50% of total cost is a hotspot."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("Cheap", outputs={"out": _pt()}, cost=ResourceCost(atp=5)))
    d.add_module(
        ModuleSpec("Expensive", inputs={"in": _pt()}, cost=ResourceCost(atp=100))
    )
    d.connect("Cheap", "out", "Expensive", "in")

    suggestions = suggest_optimizations(d)
    hotspots = [s for s in suggestions if s.kind == "cost_hotspot"]
    assert len(hotspots) == 1
    assert hotspots[0].affected_modules == ["Expensive"]
