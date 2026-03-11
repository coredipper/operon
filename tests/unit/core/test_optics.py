"""Tests for optic-based wiring (Paper §3.4)."""

import pytest

from operon_ai.core.optics import (
    ComposedOptic,
    LensOptic,
    Optic,
    OpticError,
    PrismOptic,
    TraversalOptic,
)
from operon_ai.core.types import DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, Wire, WiringDiagram, WiringError
from operon_ai.core.wiring_runtime import DiagramExecutor, TypedValue


# ── TestLensOptic ────────────────────────────────────────────────────────


class TestLensOptic:
    def test_protocol(self):
        assert isinstance(LensOptic(), Optic)

    def test_always_transmits(self):
        lens = LensOptic()
        for dt in DataType:
            for il in IntegrityLabel:
                assert lens.can_transmit(dt, il) is True

    def test_passthrough(self):
        lens = LensOptic()
        assert lens.transmit("hello", DataType.JSON, IntegrityLabel.VALIDATED) == "hello"

    def test_frozen(self):
        lens = LensOptic()
        with pytest.raises(AttributeError):
            lens.foo = "bar"  # type: ignore[attr-defined]


# ── TestPrismOptic ───────────────────────────────────────────────────────


class TestPrismOptic:
    def test_protocol(self):
        prism = PrismOptic(accept=frozenset({DataType.JSON}))
        assert isinstance(prism, Optic)

    def test_accepts_matching(self):
        prism = PrismOptic(accept=frozenset({DataType.JSON}))
        assert prism.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED) is True

    def test_rejects_non_matching(self):
        prism = PrismOptic(accept=frozenset({DataType.JSON}))
        assert prism.can_transmit(DataType.ERROR, IntegrityLabel.VALIDATED) is False

    def test_transmit_raises_on_mismatch(self):
        prism = PrismOptic(accept=frozenset({DataType.JSON}))
        with pytest.raises(OpticError):
            prism.transmit("err", DataType.ERROR, IntegrityLabel.VALIDATED)

    def test_passthrough_on_match(self):
        prism = PrismOptic(accept=frozenset({DataType.JSON}))
        assert prism.transmit({"key": 1}, DataType.JSON, IntegrityLabel.VALIDATED) == {"key": 1}

    def test_multiple_accepted_types(self):
        prism = PrismOptic(accept=frozenset({DataType.JSON, DataType.TEXT}))
        assert prism.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED) is True
        assert prism.can_transmit(DataType.TEXT, IntegrityLabel.VALIDATED) is True
        assert prism.can_transmit(DataType.ERROR, IntegrityLabel.VALIDATED) is False


# ── TestTraversalOptic ───────────────────────────────────────────────────


class TestTraversalOptic:
    def test_protocol(self):
        assert isinstance(TraversalOptic(), Optic)

    def test_no_transform_passthrough(self):
        t = TraversalOptic()
        assert t.transmit([1, 2, 3], DataType.JSON, IntegrityLabel.VALIDATED) == [1, 2, 3]

    def test_transform_list(self):
        t = TraversalOptic(transform=lambda x: x * 2)
        assert t.transmit([1, 2, 3], DataType.JSON, IntegrityLabel.VALIDATED) == [2, 4, 6]

    def test_transform_single_value(self):
        t = TraversalOptic(transform=lambda x: x.upper())
        assert t.transmit("hello", DataType.TEXT, IntegrityLabel.VALIDATED) == "HELLO"


# ── TestComposedOptic ────────────────────────────────────────────────────


class TestComposedOptic:
    def test_protocol(self):
        assert isinstance(ComposedOptic(optics=()), Optic)

    def test_compose_prism_and_traversal(self):
        composed = ComposedOptic(optics=(
            PrismOptic(accept=frozenset({DataType.JSON})),
            TraversalOptic(transform=lambda x: x * 10),
        ))
        assert composed.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED) is True
        assert composed.can_transmit(DataType.ERROR, IntegrityLabel.VALIDATED) is False
        assert composed.transmit([1, 2], DataType.JSON, IntegrityLabel.VALIDATED) == [10, 20]

    def test_empty_passthrough(self):
        composed = ComposedOptic(optics=())
        assert composed.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED) is True
        assert composed.transmit("x", DataType.JSON, IntegrityLabel.VALIDATED) == "x"


# ── TestWireOpticIntegration ─────────────────────────────────────────────


class TestWireOpticIntegration:
    def test_wire_stores_optic(self):
        optic = LensOptic()
        wire = Wire("a", "out", "b", "in", optic=optic)
        assert wire.optic is optic

    def test_wire_defaults_none(self):
        wire = Wire("a", "out", "b", "in")
        assert wire.optic is None

    def test_connect_with_optic(self):
        diagram = WiringDiagram()
        pt = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        diagram.add_module(ModuleSpec(name="a", outputs={"out": pt}))
        diagram.add_module(ModuleSpec(name="b", inputs={"in": pt}))
        prism = PrismOptic(accept=frozenset({DataType.JSON}))
        diagram.connect("a", "out", "b", "in", optic=prism)
        assert diagram.wires[0].optic is prism

    def test_backward_compat(self):
        diagram = WiringDiagram()
        pt = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        diagram.add_module(ModuleSpec(name="a", outputs={"out": pt}))
        diagram.add_module(ModuleSpec(name="b", inputs={"in": pt}))
        diagram.connect("a", "out", "b", "in")
        assert diagram.wires[0].optic is None


# ── TestExecutorOptic ────────────────────────────────────────────────────


class TestExecutorOptic:
    def _make_router_diagram(self):
        """A -> B (prism JSON) and A -> C (prism ERROR), fan-out."""
        pt_json = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        pt_err = PortType(DataType.ERROR, IntegrityLabel.VALIDATED)

        diagram = WiringDiagram()
        diagram.add_module(ModuleSpec(name="A", inputs={"in": pt_json}, outputs={"out": pt_json}))
        diagram.add_module(ModuleSpec(name="B", inputs={"in": pt_json}))
        diagram.add_module(ModuleSpec(name="C", inputs={"in": pt_err}))

        diagram.connect("A", "out", "B", "in", optic=PrismOptic(accept=frozenset({DataType.JSON})))
        diagram.connect("A", "out", "C", "in", optic=PrismOptic(accept=frozenset({DataType.ERROR})))
        return diagram

    def test_prism_routing_json(self):
        diagram = self._make_router_diagram()
        executor = DiagramExecutor(diagram)
        executor.register_module("A", lambda inputs: {
            "out": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, {"result": 42})
        })
        report = executor.execute(
            external_inputs={"A": {"in": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, "start")}},
            enforce_static_checks=False,
        )
        # JSON prism accepts → B gets data
        assert "B" in report.modules
        assert report.modules["B"].inputs["in"].value == {"result": 42}
        # ERROR prism rejects → C gets nothing
        assert report.modules["C"].inputs == {}

    def test_prism_routing_error(self):
        diagram = self._make_router_diagram()
        executor = DiagramExecutor(diagram)
        executor.register_module("A", lambda inputs: {
            "out": TypedValue(DataType.ERROR, IntegrityLabel.VALIDATED, "fail!")
        })
        report = executor.execute(
            external_inputs={"A": {"in": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, "start")}},
            enforce_static_checks=False,
        )
        # JSON prism rejects → B gets nothing
        assert report.modules["B"].inputs == {}
        # ERROR prism accepts → C gets data
        assert "C" in report.modules
        assert report.modules["C"].inputs["in"].value == "fail!"

    def test_traversal_maps_over_list(self):
        pt = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        diagram = WiringDiagram()
        diagram.add_module(ModuleSpec(name="A", inputs={"in": pt}, outputs={"out": pt}))
        diagram.add_module(ModuleSpec(name="B", inputs={"in": pt}))
        diagram.connect("A", "out", "B", "in", optic=TraversalOptic(transform=lambda x: x * 2))

        executor = DiagramExecutor(diagram)
        executor.register_module("A", lambda inputs: {"out": [1, 2, 3]})
        report = executor.execute(
            external_inputs={"A": {"in": "trigger"}},
        )
        assert report.modules["B"].inputs["in"].value == [2, 4, 6]

    def test_optic_and_denature_coexist(self):
        from operon_ai.core.denature import NormalizeFilter

        pt = PortType(DataType.TEXT, IntegrityLabel.VALIDATED)
        diagram = WiringDiagram()
        diagram.add_module(ModuleSpec(name="A", inputs={"in": pt}, outputs={"out": pt}))
        diagram.add_module(ModuleSpec(name="B", inputs={"in": pt}))
        diagram.connect(
            "A", "out", "B", "in",
            denature=NormalizeFilter(),
            optic=TraversalOptic(transform=str.upper),
        )

        executor = DiagramExecutor(diagram)
        executor.register_module("A", lambda inputs: {"out": "  hello  world  "})
        report = executor.execute(
            external_inputs={"A": {"in": "trigger"}},
        )
        # NormalizeFilter lowercases + NFKC normalizes, then traversal uppercases
        result = report.modules["B"].inputs["in"].value
        assert result == "  HELLO  WORLD  "

    def test_no_optic_backward_compat(self):
        pt = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        diagram = WiringDiagram()
        diagram.add_module(ModuleSpec(name="A", inputs={"in": pt}, outputs={"out": pt}))
        diagram.add_module(ModuleSpec(name="B", inputs={"in": pt}))
        diagram.connect("A", "out", "B", "in")

        executor = DiagramExecutor(diagram)
        executor.register_module("A", lambda inputs: {"out": {"key": "val"}})
        report = executor.execute(
            external_inputs={"A": {"in": "trigger"}},
        )
        assert report.modules["B"].inputs["in"].value == {"key": "val"}

    def test_fan_in_without_optics_still_rejected(self):
        """Multiple non-optic wires to same port must still raise."""
        pt = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        diagram = WiringDiagram()
        diagram.add_module(ModuleSpec(name="A", outputs={"out": pt}))
        diagram.add_module(ModuleSpec(name="B", outputs={"out": pt}))
        diagram.add_module(ModuleSpec(name="C", inputs={"in": pt}))
        diagram.connect("A", "out", "C", "in")
        diagram.connect("B", "out", "C", "in")

        executor = DiagramExecutor(diagram)
        executor.register_module("A", lambda inputs: {"out": 1})
        executor.register_module("B", lambda inputs: {"out": 2})
        with pytest.raises(WiringError, match="Multiple sources"):
            executor.execute()
