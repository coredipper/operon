"""Runtime executor for wiring diagrams."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import DataType, IntegrityLabel
from .wagent import PortType, WiringDiagram, WiringError


@dataclass(frozen=True)
class TypedValue:
    """Runtime value tagged with a data type and integrity label."""

    data_type: DataType
    integrity: IntegrityLabel
    value: Any


@dataclass
class ModuleExecution:
    """Execution record for a single module."""

    inputs: dict[str, TypedValue] = field(default_factory=dict)
    outputs: dict[str, TypedValue] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """Execution record for an entire wiring diagram."""

    modules: dict[str, ModuleExecution] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)


ModuleHandler = Callable[[dict[str, TypedValue]], dict[str, Any]]


def _coerce_output(value: Any, port: PortType) -> TypedValue:
    if isinstance(value, TypedValue):
        if value.data_type != port.data_type:
            raise WiringError(
                f"Output type mismatch: {value.data_type.value} -> {port.data_type.value}"
            )
        if value.integrity != port.integrity:
            raise WiringError(
                "Output integrity mismatch: "
                f"{value.integrity.name} != {port.integrity.name}"
            )
        return value
    return TypedValue(port.data_type, port.integrity, value)


def _coerce_input(value: Any, port: PortType) -> TypedValue:
    if isinstance(value, TypedValue):
        if value.data_type != port.data_type:
            raise WiringError(
                f"Input type mismatch: {value.data_type.value} -> {port.data_type.value}"
            )
        if value.integrity < port.integrity:
            raise WiringError(
                "Input integrity violation: "
                f"{value.integrity.name} cannot flow into {port.integrity.name}"
            )
        return value
    return TypedValue(port.data_type, port.integrity, value)


class DiagramExecutor:
    """Execute a wiring diagram with registered module handlers."""

    def __init__(self, diagram: WiringDiagram):
        self.diagram = diagram
        self._handlers: dict[str, ModuleHandler] = {}

    def register_module(self, name: str, handler: ModuleHandler) -> None:
        if name not in self.diagram.modules:
            raise WiringError(f"Unknown module: {name}")
        self._handlers[name] = handler

    def execute(
        self,
        external_inputs: dict[str, dict[str, Any]] | None = None,
        enforce_static_checks: bool = True,
    ) -> ExecutionReport:
        external_inputs = external_inputs or {}
        module_inputs: dict[str, dict[str, TypedValue]] = {
            name: {} for name in self.diagram.modules
        }

        for module_name, inputs in external_inputs.items():
            if module_name not in self.diagram.modules:
                raise WiringError(f"Unknown module in external inputs: {module_name}")
            spec = self.diagram.modules[module_name]
            for port_name, value in inputs.items():
                if port_name not in spec.inputs:
                    raise WiringError(f"Unknown input port: {module_name}.{port_name}")
                module_inputs[module_name][port_name] = _coerce_input(
                    value, spec.inputs[port_name]
                )

        outgoing: dict[str, list] = {name: [] for name in self.diagram.modules}
        incoming_by_port: dict[tuple[str, str], list] = {}

        for wire in self.diagram.wires:
            outgoing[wire.src_module].append(wire)
            incoming_by_port.setdefault((wire.dst_module, wire.dst_port), []).append(wire)

        for (module_name, port_name), wires in incoming_by_port.items():
            if len(wires) > 1:
                # Multiple prism-bearing wires are allowed (only one matches
                # at runtime).  Mixed optic/no-optic fan-in is still invalid.
                if not all(w.optic is not None for w in wires):
                    raise WiringError(
                        f"Multiple sources for input port: {module_name}.{port_name}"
                    )

        for module_name, spec in self.diagram.modules.items():
            if spec.outputs and module_name not in self._handlers:
                raise WiringError(f"No handler registered for module: {module_name}")
            for port_name in spec.inputs:
                if (module_name, port_name) not in incoming_by_port:
                    if port_name not in module_inputs[module_name]:
                        raise WiringError(
                            f"Missing input source for {module_name}.{port_name}"
                        )

        # Identify output ports whose wires ALL carry optics.
        # These ports may produce TypedValues with DataTypes that don't
        # match the port spec — the optic handles correctness at runtime.
        _optic_out_ports: set[tuple[str, str]] = set()
        for mod_name in self.diagram.modules:
            ports_with_optic: dict[str, bool] = {}
            for wire in outgoing[mod_name]:
                prev = ports_with_optic.get(wire.src_port, True)
                ports_with_optic[wire.src_port] = prev and (wire.optic is not None)
            for port, all_optic in ports_with_optic.items():
                if all_optic:
                    _optic_out_ports.add((mod_name, port))

        report = ExecutionReport()
        executed: set[str] = set()

        while len(executed) < len(self.diagram.modules):
            progressed = False
            for module_name, spec in self.diagram.modules.items():
                if module_name in executed:
                    continue

                ready = all(
                    port in module_inputs[module_name] for port in spec.inputs
                )
                if not ready:
                    # Check if all missing inputs come from optic wires whose
                    # sources already executed (= prism rejected the data).
                    # If so, skip this module gracefully.
                    all_prism_rejected = True
                    for port in spec.inputs:
                        if port in module_inputs[module_name]:
                            continue
                        port_wires = incoming_by_port.get((module_name, port), [])
                        if not port_wires or not all(
                            w.optic is not None and w.src_module in executed
                            for w in port_wires
                        ):
                            all_prism_rejected = False
                            break
                    if all_prism_rejected:
                        # Module skipped: all inputs were prism-rejected
                        report.modules[module_name] = ModuleExecution()
                        report.execution_order.append(module_name)
                        executed.add(module_name)
                        progressed = True
                    continue

                handler = self._handlers.get(module_name)
                inputs = module_inputs[module_name]
                outputs: dict[str, TypedValue] = {}

                if handler is not None:
                    raw_outputs = handler(inputs) or {}
                    if set(raw_outputs.keys()) != set(spec.outputs.keys()):
                        raise WiringError(
                            f"Output ports mismatch for {module_name}: "
                            f"expected {sorted(spec.outputs.keys())}, "
                            f"got {sorted(raw_outputs.keys())}"
                        )
                    for port_name, port_type in spec.outputs.items():
                        raw_val = raw_outputs[port_name]
                        if (module_name, port_name) in _optic_out_ports and isinstance(raw_val, TypedValue):
                            # Optic-bearing port: accept TypedValue as-is;
                            # the optic handles type correctness at runtime.
                            outputs[port_name] = raw_val
                        else:
                            outputs[port_name] = _coerce_output(raw_val, port_type)

                report.modules[module_name] = ModuleExecution(
                    inputs=dict(inputs),
                    outputs=dict(outputs),
                )
                report.execution_order.append(module_name)
                executed.add(module_name)
                progressed = True

                for wire in outgoing[module_name]:
                    value = outputs.get(wire.src_port)
                    if value is None:
                        raise WiringError(
                            f"Missing output {module_name}.{wire.src_port} for wire"
                        )
                    # Denaturation: apply wire-level filter (Paper §5.3)
                    if wire.denature is not None:
                        denatured_str = wire.denature.denature(str(value.value))
                        value = TypedValue(
                            data_type=value.data_type,
                            integrity=value.integrity,
                            value=denatured_str,
                        )
                    # Optic: apply wire-level optic routing/transform (Paper §3.3)
                    if wire.optic is not None:
                        if not wire.optic.can_transmit(value.data_type, value.integrity):
                            continue  # Prism rejection: skip this wire
                        transformed = wire.optic.transmit(
                            value.value, value.data_type, value.integrity
                        )
                        value = TypedValue(
                            data_type=value.data_type,
                            integrity=value.integrity,
                            value=transformed,
                        )
                    dst_spec = self.diagram.modules[wire.dst_module].inputs[wire.dst_port]
                    if enforce_static_checks:
                        if value.data_type != dst_spec.data_type:
                            raise WiringError(
                                f"Type mismatch: {value.data_type.value} -> {dst_spec.data_type.value}"
                            )
                        if value.integrity < dst_spec.integrity:
                            raise WiringError(
                                "Integrity violation: "
                                f"{value.integrity.name} cannot flow into {dst_spec.integrity.name}"
                            )
                    if wire.dst_port in module_inputs[wire.dst_module]:
                        raise WiringError(
                            f"Multiple values for input {wire.dst_module}.{wire.dst_port}"
                        )
                    module_inputs[wire.dst_module][wire.dst_port] = value

            if not progressed:
                pending = [m for m in self.diagram.modules if m not in executed]
                missing = {
                    m: [
                        port for port in self.diagram.modules[m].inputs
                        if port not in module_inputs[m]
                    ]
                    for m in pending
                }
                raise WiringError(f"Cannot resolve wiring; missing inputs: {missing}")

        return report
