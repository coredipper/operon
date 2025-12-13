"""
WAgent: Typed Wiring Diagrams for Agents
=======================================

This module provides a minimal, runtime-checkable representation of the
paper's `WAgent` idea: modules with typed ports, integrity labels, and
capability/effect annotations.

It is intentionally lightweight: it checks wiring constraints, but it does
not execute the diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Capability, DataType, IntegrityLabel


class WiringError(ValueError):
    """Raised when a wiring diagram is ill-typed or invalid."""


@dataclass(frozen=True)
class PortType:
    """A decorated port type: (data type, integrity label)."""

    data_type: DataType
    integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED

    def can_flow_to(self, other: "PortType") -> bool:
        """Return True if this port can legally connect to `other`."""
        return self.data_type == other.data_type and self.integrity >= other.integrity

    def require_flow_to(self, other: "PortType"):
        """Raise WiringError if this port cannot connect to `other`."""
        if self.data_type != other.data_type:
            raise WiringError(
                f"Type mismatch: {self.data_type.value} -> {other.data_type.value}"
            )
        if self.integrity < other.integrity:
            raise WiringError(
                "Integrity violation: "
                f"{self.integrity.name} cannot flow into {other.integrity.name}"
            )


@dataclass(frozen=True)
class ModuleSpec:
    """A module with named input/output ports and capability annotations."""

    name: str
    inputs: dict[str, PortType] = field(default_factory=dict)
    outputs: dict[str, PortType] = field(default_factory=dict)
    capabilities: set[Capability] = field(default_factory=set)


@dataclass(frozen=True)
class Wire:
    """A connection between two module ports."""

    src_module: str
    src_port: str
    dst_module: str
    dst_port: str


@dataclass
class WiringDiagram:
    """A collection of modules connected by well-typed wires."""

    modules: dict[str, ModuleSpec] = field(default_factory=dict)
    wires: list[Wire] = field(default_factory=list)

    def add_module(self, module: ModuleSpec):
        if module.name in self.modules:
            raise WiringError(f"Module already exists: {module.name}")
        self.modules[module.name] = module

    def connect(self, src_module: str, src_port: str, dst_module: str, dst_port: str):
        try:
            src = self.modules[src_module].outputs[src_port]
        except KeyError as e:
            raise WiringError(f"Unknown output port: {src_module}.{src_port}") from e

        try:
            dst = self.modules[dst_module].inputs[dst_port]
        except KeyError as e:
            raise WiringError(f"Unknown input port: {dst_module}.{dst_port}") from e

        src.require_flow_to(dst)
        self.wires.append(Wire(src_module, src_port, dst_module, dst_port))

    def required_capabilities(self) -> set[Capability]:
        """Union of capabilities across all modules (effect aggregation)."""
        required: set[Capability] = set()
        for module in self.modules.values():
            required |= module.capabilities
        return required

