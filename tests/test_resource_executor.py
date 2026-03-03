"""Tests for resource-aware executor — ATP-gated wiring diagram execution."""

import pytest

from operon_ai.core.optimizer import optimize
from operon_ai.core.types import DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram
from operon_ai.core.wiring_runtime import (
    ResourceAwareExecutor,
    ResourceAwareReport,
    TypedValue,
)
from operon_ai.state.metabolism import ATP_Store, MetabolicState


def _pt(dt=DataType.JSON, il=IntegrityLabel.VALIDATED):
    return PortType(dt, il)


def _identity_handler(inputs):
    """Pass-through handler: copies first input to 'out'."""
    first_val = next(iter(inputs.values()))
    return {"out": first_val.value}


# ── ATP deduction ───────────────────────────────────────────────────


def test_atp_deduction_during_execution():
    """Module execution deducts ATP from the store."""
    d = WiringDiagram()
    d.add_module(
        ModuleSpec("A", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=10))
    )
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, cost=ResourceCost(atp=20)))
    d.connect("A", "out", "B", "in")

    store = ATP_Store(budget=100, silent=True)
    od = optimize(d)
    executor = ResourceAwareExecutor(od, store)
    executor.register_module("A", _identity_handler)

    report = executor.execute(
        external_inputs={"A": {"in": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, "data")}},
    )

    assert isinstance(report, ResourceAwareReport)
    assert report.total_cost.atp == 30
    assert store.get_balance() <= 70  # At least 30 deducted


def test_wire_cost_deducted():
    """Wire transmission costs are deducted from ATP."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", inputs={"in": _pt()}, outputs={"out": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    from operon_ai.core.wagent import Wire

    d.wires = [Wire("A", "out", "B", "in", cost=15)]

    store = ATP_Store(budget=100, silent=True)
    od = optimize(d)
    executor = ResourceAwareExecutor(od, store)
    executor.register_module("A", _identity_handler)

    executor.execute(
        external_inputs={"A": {"in": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, "data")}},
    )

    assert store.get_balance() <= 85


# ── Module skipping under STARVING ──────────────────────────────────


def test_skip_nonessential_when_starving():
    """Non-essential modules are skipped when store is STARVING."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("Essential", cost=ResourceCost(atp=1), essential=True))
    d.add_module(ModuleSpec("Optional", cost=ResourceCost(atp=1), essential=False))

    store = ATP_Store(budget=100, silent=True)
    # Drain to STARVING
    store.consume(95, "drain")

    od = optimize(d)
    executor = ResourceAwareExecutor(od, store)

    report = executor.execute()

    assert "Optional" in report.skipped_modules
    assert "Essential" not in report.skipped_modules


# ── Cost-ordered execution under CONSERVING ─────────────────────────


def test_cost_ordered_under_conserving():
    """Under CONSERVING, expensive non-essential modules may be skipped."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("Cheap", cost=ResourceCost(atp=1), essential=False))
    d.add_module(ModuleSpec("Expensive", cost=ResourceCost(atp=50), essential=False))

    store = ATP_Store(budget=100, silent=True)
    # Drain to CONSERVING (below 30%)
    store.consume(75, "drain")

    od = optimize(d)
    executor = ResourceAwareExecutor(od, store)

    report = executor.execute()

    # Expensive should be skipped (cost > balance//2), Cheap should execute
    assert "Expensive" in report.skipped_modules


# ── Parallel execution of independent groups ────────────────────────


def test_parallel_execution():
    """Independent modules execute (report includes both)."""
    d = WiringDiagram()
    d.add_module(
        ModuleSpec("A", inputs={"in": _pt()}, outputs={"o1": _pt(), "o2": _pt()}, cost=ResourceCost(atp=1))
    )
    d.add_module(
        ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=1))
    )
    d.add_module(
        ModuleSpec("C", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=1))
    )
    d.add_module(ModuleSpec("D", inputs={"i1": _pt(), "i2": _pt()}, cost=ResourceCost(atp=1)))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")
    d.connect("B", "out", "D", "i1")
    d.connect("C", "out", "D", "i2")

    store = ATP_Store(budget=1000, silent=True)
    # Ensure NORMAL/FEASTING state for parallel execution
    od = optimize(d)
    executor = ResourceAwareExecutor(od, store, max_workers=2)
    executor.register_module("A", lambda inputs: {
        "o1": "from_a1",
        "o2": "from_a2",
    })
    executor.register_module("B", _identity_handler)
    executor.register_module("C", _identity_handler)

    report = executor.execute(
        external_inputs={"A": {"in": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, "start")}},
        enforce_static_checks=False,
    )

    # All modules should have executed
    assert "A" in report.execution_order
    assert "B" in report.execution_order
    assert "C" in report.execution_order
    assert "D" in report.execution_order
    assert report.total_cost.atp == 4
    assert report.skipped_modules == []


# ── Edge cases ──────────────────────────────────────────────────────


def test_no_cost_annotation():
    """Modules without cost annotations execute with zero cost."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A"))

    store = ATP_Store(budget=100, silent=True)
    od = optimize(d)
    executor = ResourceAwareExecutor(od, store)

    report = executor.execute()
    assert report.total_cost.atp == 0
    assert store.get_balance() == 100


def test_insufficient_atp_skips_module():
    """When ATP is insufficient, module is skipped."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("Hungry", cost=ResourceCost(atp=200)))

    store = ATP_Store(budget=50, silent=True)
    od = optimize(d)
    executor = ResourceAwareExecutor(od, store)

    report = executor.execute()
    assert "Hungry" in report.skipped_modules
