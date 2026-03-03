"""Tests for diagram optimizer — rewriting passes and OptimizedDiagram."""

import pytest

from operon_ai.core.optics import PrismOptic
from operon_ai.core.optimizer import (
    CostOrderSchedule,
    EliminateDeadWires,
    OptimizationPass,
    OptimizedDiagram,
    ParallelGrouping,
    optimize,
)
from operon_ai.core.types import DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram


def _pt(dt=DataType.JSON, il=IntegrityLabel.VALIDATED):
    return PortType(dt, il)


# ── EliminateDeadWires ──────────────────────────────────────────────


def test_eliminate_dead_wires_removes_dead():
    """Dead prism wires are removed from the diagram."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt(DataType.ERROR)}))

    # Live wire: JSON prism on JSON source
    live_prism = PrismOptic(accept=frozenset({DataType.JSON}))
    d.connect("A", "out", "B", "in", optic=live_prism)

    # Dead wire: ERROR prism on JSON source
    dead_prism = PrismOptic(accept=frozenset({DataType.ERROR}))
    d.connect("A", "out", "C", "in", optic=dead_prism)

    assert len(d.wires) == 2

    eliminator = EliminateDeadWires()
    result = eliminator.apply(d)

    assert len(result.wires) == 1
    assert result.wires[0].dst_module == "B"


def test_eliminate_dead_wires_no_change():
    """When no dead wires exist, diagram is unchanged."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.connect("A", "out", "B", "in")

    eliminator = EliminateDeadWires()
    result = eliminator.apply(d)
    assert len(result.wires) == len(d.wires)


def test_eliminate_preserves_behavior():
    """Elimination preserves all non-dead wires."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}))
    d.connect("A", "out", "B", "in")
    d.connect("B", "out", "C", "in")

    eliminator = EliminateDeadWires()
    result = eliminator.apply(d)
    assert len(result.wires) == 2


# ── ParallelGrouping ───────────────────────────────────────────────


def test_parallel_grouping_preserves_structure():
    """ParallelGrouping does not modify the diagram structure."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.connect("A", "out", "B", "in")

    grouping = ParallelGrouping()
    result = grouping.apply(d)
    assert len(result.modules) == len(d.modules)
    assert len(result.wires) == len(d.wires)


def test_parallel_grouping_protocol():
    """ParallelGrouping satisfies OptimizationPass protocol."""
    assert isinstance(ParallelGrouping(), OptimizationPass)


# ── CostOrderSchedule ─────────────────────────────────────────────


def test_cost_order_schedule_preserves_structure():
    """CostOrderSchedule does not modify the diagram."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", cost=ResourceCost(atp=50)))
    d.add_module(ModuleSpec("B", cost=ResourceCost(atp=10)))

    scheduler = CostOrderSchedule()
    result = scheduler.apply(d)
    assert set(result.modules.keys()) == {"A", "B"}


def test_cost_order_schedule_protocol():
    """CostOrderSchedule satisfies OptimizationPass protocol."""
    assert isinstance(CostOrderSchedule(), OptimizationPass)


# ── optimize pipeline ──────────────────────────────────────────────


def test_optimize_default_passes():
    """Full optimize pipeline with default passes."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}, cost=ResourceCost(atp=5)))
    d.add_module(
        ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=10))
    )
    d.add_module(
        ModuleSpec("C", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=20))
    )
    d.add_module(ModuleSpec("D", inputs={"i1": _pt(), "i2": _pt()}, cost=ResourceCost(atp=3)))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")
    d.connect("B", "out", "D", "i1")
    d.connect("C", "out", "D", "i2")

    result = optimize(d)

    assert isinstance(result, OptimizedDiagram)
    assert len(result.passes_applied) == 3
    assert "eliminate_dead_wires" in result.passes_applied
    assert "parallel_grouping" in result.passes_applied
    assert "cost_order_schedule" in result.passes_applied


def test_optimize_parallel_groups():
    """Parallel groups are detected in optimized diagram."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")

    result = optimize(d)
    assert any({"B", "C"} == g for g in result.parallel_groups)


def test_optimize_schedule_cost_ordered():
    """Within a parallel group, cheaper modules come first in schedule."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("Expensive", cost=ResourceCost(atp=100)))
    d.add_module(ModuleSpec("Cheap", cost=ResourceCost(atp=1)))

    result = optimize(d)
    # Both are in the same group (no dependencies), schedule should have Cheap first
    cheap_idx = result.schedule.index("Cheap")
    expensive_idx = result.schedule.index("Expensive")
    assert cheap_idx < expensive_idx


def test_optimize_with_dead_wire_elimination():
    """Dead wires are eliminated and tracked."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt(DataType.ERROR)}))

    live_prism = PrismOptic(accept=frozenset({DataType.JSON}))
    d.connect("A", "out", "B", "in", optic=live_prism)
    dead_prism = PrismOptic(accept=frozenset({DataType.ERROR}))
    d.connect("A", "out", "C", "in", optic=dead_prism)

    result = optimize(d)
    assert len(result.diagram.wires) == 1
    assert len(result.eliminated_wires) == 1


def test_optimize_custom_passes():
    """Custom pass list is applied in order."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A"))

    result = optimize(d, passes=[ParallelGrouping()])
    assert result.passes_applied == ["parallel_grouping"]
