"""
Comprehensive tests for WAgent typed wiring diagrams.

Tests cover:
- Basic diagram functionality
- Wire connections and validation
- Capability tracking
- Integrity label flows
- Complex multi-module scenarios
"""

import pytest

from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, Wire, WiringDiagram, WiringError


# ============================================================================
# TestWiringDiagramBasics - Basic functionality
# ============================================================================


class TestWiringDiagramBasics:
    """Test basic WiringDiagram functionality."""

    def test_create_empty_diagram(self):
        """Should create an empty wiring diagram."""
        diagram = WiringDiagram()
        assert len(diagram.modules) == 0
        assert len(diagram.wires) == 0

    def test_add_module(self):
        """Should add a module to the diagram."""
        diagram = WiringDiagram()
        module = ModuleSpec(
            name="test_module",
            inputs={"in": PortType(DataType.TEXT)},
            outputs={"out": PortType(DataType.TEXT)},
        )
        diagram.add_module(module)
        assert "test_module" in diagram.modules
        assert diagram.modules["test_module"] == module

    def test_add_multiple_modules(self):
        """Should add multiple modules to the diagram."""
        diagram = WiringDiagram()
        module1 = ModuleSpec(name="module1")
        module2 = ModuleSpec(name="module2")
        module3 = ModuleSpec(name="module3")

        diagram.add_module(module1)
        diagram.add_module(module2)
        diagram.add_module(module3)

        assert len(diagram.modules) == 3
        assert "module1" in diagram.modules
        assert "module2" in diagram.modules
        assert "module3" in diagram.modules

    def test_duplicate_module_raises_error(self):
        """Should raise WiringError when adding duplicate module."""
        diagram = WiringDiagram()
        module = ModuleSpec(name="duplicate")
        diagram.add_module(module)

        with pytest.raises(WiringError, match="Module already exists: duplicate"):
            diagram.add_module(module)

    def test_module_with_no_ports(self):
        """Should allow module with no input/output ports."""
        diagram = WiringDiagram()
        module = ModuleSpec(name="isolated")
        diagram.add_module(module)
        assert "isolated" in diagram.modules
        assert len(module.inputs) == 0
        assert len(module.outputs) == 0


# ============================================================================
# TestWiringConnections - Wire connection tests
# ============================================================================


class TestWiringConnections:
    """Test wire connections and validation."""

    def test_valid_connection_same_type_and_integrity(self):
        """Should connect ports with matching type and integrity."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )

        diagram.connect("source", "out", "sink", "in")
        assert len(diagram.wires) == 1
        assert diagram.wires[0] == Wire("source", "out", "sink", "in")

    def test_valid_connection_integrity_downgrade(self):
        """Should allow connection from higher to lower integrity."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            )
        )

        diagram.connect("source", "out", "sink", "in")
        assert len(diagram.wires) == 1

    def test_type_mismatch_raises_error(self):
        """Should raise WiringError on data type mismatch."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink",
                inputs={"in": PortType(DataType.JSON)},
            )
        )

        with pytest.raises(WiringError, match="Type mismatch: text -> json"):
            diagram.connect("source", "out", "sink", "in")

    def test_integrity_upgrade_raises_error(self):
        """Should raise WiringError when trying to upgrade integrity."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )

        with pytest.raises(WiringError, match="Integrity violation"):
            diagram.connect("source", "out", "sink", "in")

    def test_nonexistent_source_module_raises_error(self):
        """Should raise WiringError for nonexistent source module."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="sink",
                inputs={"in": PortType(DataType.TEXT)},
            )
        )

        with pytest.raises(WiringError, match="Unknown output port"):
            diagram.connect("nonexistent", "out", "sink", "in")

    def test_nonexistent_destination_module_raises_error(self):
        """Should raise WiringError for nonexistent destination module."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT)},
            )
        )

        with pytest.raises(WiringError, match="Unknown input port"):
            diagram.connect("source", "out", "nonexistent", "in")

    def test_nonexistent_source_port_raises_error(self):
        """Should raise WiringError for nonexistent source port."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink",
                inputs={"in": PortType(DataType.TEXT)},
            )
        )

        with pytest.raises(WiringError, match="Unknown output port: source.wrong"):
            diagram.connect("source", "wrong", "sink", "in")

    def test_nonexistent_destination_port_raises_error(self):
        """Should raise WiringError for nonexistent destination port."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink",
                inputs={"in": PortType(DataType.TEXT)},
            )
        )

        with pytest.raises(WiringError, match="Unknown input port: sink.wrong"):
            diagram.connect("source", "out", "sink", "wrong")

    def test_multiple_wires_from_same_output(self):
        """Should allow multiple wires from the same output (fanout)."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="source",
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink1",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sink2",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            )
        )

        diagram.connect("source", "out", "sink1", "in")
        diagram.connect("source", "out", "sink2", "in")
        assert len(diagram.wires) == 2


# ============================================================================
# TestCapabilities - Capability tracking
# ============================================================================


class TestCapabilities:
    """Test capability tracking and aggregation."""

    def test_module_capabilities_tracked(self):
        """Should track capabilities assigned to modules."""
        module = ModuleSpec(
            name="fs_writer",
            capabilities={Capability.WRITE_FS},
        )
        assert Capability.WRITE_FS in module.capabilities

    def test_empty_capabilities(self):
        """Should handle modules with no capabilities."""
        module = ModuleSpec(name="pure")
        assert len(module.capabilities) == 0

    def test_required_capabilities_empty_diagram(self):
        """Should return empty set for diagram with no modules."""
        diagram = WiringDiagram()
        assert diagram.required_capabilities() == set()

    def test_required_capabilities_single_module(self):
        """Should aggregate capabilities from single module."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="writer",
                capabilities={Capability.WRITE_FS, Capability.READ_FS},
            )
        )
        required = diagram.required_capabilities()
        assert required == {Capability.WRITE_FS, Capability.READ_FS}

    def test_required_capabilities_multiple_modules(self):
        """Should aggregate capabilities from multiple modules."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="reader",
                capabilities={Capability.READ_FS},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="writer",
                capabilities={Capability.WRITE_FS},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="network",
                capabilities={Capability.NET},
            )
        )
        required = diagram.required_capabilities()
        assert required == {Capability.READ_FS, Capability.WRITE_FS, Capability.NET}

    def test_required_capabilities_overlapping(self):
        """Should deduplicate overlapping capabilities."""
        diagram = WiringDiagram()
        diagram.add_module(
            ModuleSpec(
                name="module1",
                capabilities={Capability.READ_FS, Capability.WRITE_FS},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="module2",
                capabilities={Capability.READ_FS},
            )
        )
        required = diagram.required_capabilities()
        assert required == {Capability.READ_FS, Capability.WRITE_FS}


# ============================================================================
# TestIntegrityLabels - Integrity label tests
# ============================================================================


class TestIntegrityLabels:
    """Test integrity label flow rules."""

    def test_validated_to_validated_allowed(self):
        """Should allow VALIDATED -> VALIDATED connection."""
        src = PortType(DataType.TEXT, IntegrityLabel.VALIDATED)
        dst = PortType(DataType.TEXT, IntegrityLabel.VALIDATED)
        assert src.can_flow_to(dst) is True
        src.require_flow_to(dst)  # Should not raise

    def test_trusted_to_validated_allowed(self):
        """Should allow TRUSTED -> VALIDATED connection."""
        src = PortType(DataType.TEXT, IntegrityLabel.TRUSTED)
        dst = PortType(DataType.TEXT, IntegrityLabel.VALIDATED)
        assert src.can_flow_to(dst) is True
        src.require_flow_to(dst)  # Should not raise

    def test_trusted_to_untrusted_allowed(self):
        """Should allow TRUSTED -> UNTRUSTED connection."""
        src = PortType(DataType.TEXT, IntegrityLabel.TRUSTED)
        dst = PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)
        assert src.can_flow_to(dst) is True
        src.require_flow_to(dst)  # Should not raise

    def test_untrusted_to_untrusted_allowed(self):
        """Should allow UNTRUSTED -> UNTRUSTED connection."""
        src = PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)
        dst = PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)
        assert src.can_flow_to(dst) is True
        src.require_flow_to(dst)  # Should not raise

    def test_untrusted_to_validated_blocked(self):
        """Should block UNTRUSTED -> VALIDATED connection."""
        src = PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)
        dst = PortType(DataType.TEXT, IntegrityLabel.VALIDATED)
        assert src.can_flow_to(dst) is False
        with pytest.raises(WiringError, match="Integrity violation"):
            src.require_flow_to(dst)

    def test_untrusted_to_trusted_blocked(self):
        """Should block UNTRUSTED -> TRUSTED connection."""
        src = PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)
        dst = PortType(DataType.TEXT, IntegrityLabel.TRUSTED)
        assert src.can_flow_to(dst) is False
        with pytest.raises(WiringError, match="Integrity violation"):
            src.require_flow_to(dst)

    def test_validated_to_trusted_blocked(self):
        """Should block VALIDATED -> TRUSTED connection."""
        src = PortType(DataType.TEXT, IntegrityLabel.VALIDATED)
        dst = PortType(DataType.TEXT, IntegrityLabel.TRUSTED)
        assert src.can_flow_to(dst) is False
        with pytest.raises(WiringError, match="Integrity violation"):
            src.require_flow_to(dst)


# ============================================================================
# TestComplexDiagrams - Complex scenarios
# ============================================================================


class TestComplexDiagrams:
    """Test complex multi-module wiring scenarios."""

    def test_multi_module_chain(self):
        """Should wire a chain of multiple modules."""
        diagram = WiringDiagram()

        # Create a chain: input -> processor1 -> processor2 -> output
        diagram.add_module(
            ModuleSpec(
                name="input",
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="processor1",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="processor2",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="output",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )

        diagram.connect("input", "out", "processor1", "in")
        diagram.connect("processor1", "out", "processor2", "in")
        diagram.connect("processor2", "out", "output", "in")

        assert len(diagram.wires) == 3
        assert len(diagram.modules) == 4

    def test_fanout_pattern(self):
        """Should support fanout pattern (one source, multiple sinks)."""
        diagram = WiringDiagram()

        diagram.add_module(
            ModuleSpec(
                name="broadcaster",
                outputs={"out": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="listener1",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="listener2",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="listener3",
                inputs={"in": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            )
        )

        diagram.connect("broadcaster", "out", "listener1", "in")
        diagram.connect("broadcaster", "out", "listener2", "in")
        diagram.connect("broadcaster", "out", "listener3", "in")

        assert len(diagram.wires) == 3

    def test_multiple_data_types(self):
        """Should handle modules with different data types."""
        diagram = WiringDiagram()

        diagram.add_module(
            ModuleSpec(
                name="multi_output",
                outputs={
                    "text": PortType(DataType.TEXT),
                    "json": PortType(DataType.JSON),
                    "image": PortType(DataType.IMAGE),
                },
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="text_sink",
                inputs={"in": PortType(DataType.TEXT)},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="json_sink",
                inputs={"in": PortType(DataType.JSON)},
            )
        )

        diagram.connect("multi_output", "text", "text_sink", "in")
        diagram.connect("multi_output", "json", "json_sink", "in")

        assert len(diagram.wires) == 2

    def test_complex_capabilities_aggregation(self):
        """Should aggregate capabilities in complex diagram."""
        diagram = WiringDiagram()

        diagram.add_module(
            ModuleSpec(
                name="reader",
                outputs={"data": PortType(DataType.TEXT)},
                capabilities={Capability.READ_FS},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="processor",
                inputs={"in": PortType(DataType.TEXT)},
                outputs={"out": PortType(DataType.JSON)},
                capabilities={Capability.EXEC_CODE},
            )
        )
        diagram.add_module(
            ModuleSpec(
                name="sender",
                inputs={"data": PortType(DataType.JSON)},
                capabilities={Capability.NET, Capability.EMAIL_SEND},
            )
        )

        diagram.connect("reader", "data", "processor", "in")
        diagram.connect("processor", "out", "sender", "data")

        required = diagram.required_capabilities()
        assert required == {
            Capability.READ_FS,
            Capability.EXEC_CODE,
            Capability.NET,
            Capability.EMAIL_SEND,
        }

    def test_port_type_with_all_data_types(self):
        """Should handle all defined data types."""
        all_types = [
            DataType.TEXT,
            DataType.JSON,
            DataType.IMAGE,
            DataType.TOOL_CALL,
            DataType.ERROR,
            DataType.STOP,
            DataType.APPROVAL,
        ]

        for data_type in all_types:
            port = PortType(data_type, IntegrityLabel.TRUSTED)
            assert port.data_type == data_type
            assert port.integrity == IntegrityLabel.TRUSTED

    def test_module_with_multiple_inputs_and_outputs(self):
        """Should handle modules with multiple ports."""
        diagram = WiringDiagram()

        diagram.add_module(
            ModuleSpec(
                name="multi_io",
                inputs={
                    "text_in": PortType(DataType.TEXT),
                    "json_in": PortType(DataType.JSON),
                },
                outputs={
                    "text_out": PortType(DataType.TEXT),
                    "json_out": PortType(DataType.JSON),
                    "error_out": PortType(DataType.ERROR),
                },
            )
        )

        assert "multi_io" in diagram.modules
        assert len(diagram.modules["multi_io"].inputs) == 2
        assert len(diagram.modules["multi_io"].outputs) == 3
