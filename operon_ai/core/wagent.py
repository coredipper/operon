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
from typing import Any

from .types import Capability, DataType, IntegrityLabel


class WiringError(ValueError):
    """Raised when a wiring diagram is ill-typed or invalid."""


@dataclass(frozen=True)
class ResourceCost:
    """Cost annotation for modules and wires.

    Models resource consumption as enrichment of the wiring category
    over (R_+, +, 0) — costs compose additively along paths.

    Biological Analogy:
    ATP cost of enzyme catalysis, latency of signal transduction,
    memory footprint of protein complexes.
    """

    atp: int = 0
    latency_ms: float = 0.0
    memory_mb: float = 0.0

    def __add__(self, other: "ResourceCost") -> "ResourceCost":
        return ResourceCost(
            atp=self.atp + other.atp,
            latency_ms=self.latency_ms + other.latency_ms,
            memory_mb=self.memory_mb + other.memory_mb,
        )


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
    cost: ResourceCost | None = None
    essential: bool = True


@dataclass(frozen=True)
class Wire:
    """A connection between two module ports.

    The optional ``denature`` field accepts a DenatureFilter that
    transforms data in transit, disrupting prompt-injection cascades
    (Paper §5.3 — anti-prion defense).
    """

    src_module: str
    src_port: str
    dst_module: str
    dst_port: str
    denature: Any | None = None  # Optional DenatureFilter
    optic: Any | None = None  # Optional Optic (Paper §3.4)
    cost: int = 0  # Transmission cost in ATP


@dataclass
class WiringDiagram:
    """A collection of modules connected by well-typed wires."""

    modules: dict[str, ModuleSpec] = field(default_factory=dict)
    wires: list[Wire] = field(default_factory=list)

    def add_module(self, module: ModuleSpec):
        if module.name in self.modules:
            raise WiringError(f"Module already exists: {module.name}")
        self.modules[module.name] = module

    def connect(
        self,
        src_module: str,
        src_port: str,
        dst_module: str,
        dst_port: str,
        denature: Any | None = None,
        optic: Any | None = None,
    ):
        try:
            src = self.modules[src_module].outputs[src_port]
        except KeyError as e:
            raise WiringError(f"Unknown output port: {src_module}.{src_port}") from e

        try:
            dst = self.modules[dst_module].inputs[dst_port]
        except KeyError as e:
            raise WiringError(f"Unknown input port: {dst_module}.{dst_port}") from e

        # When an optic is attached, it handles type filtering at runtime
        # (e.g. a prism intentionally connects mismatched DataTypes).
        if optic is None:
            src.require_flow_to(dst)
        self.wires.append(
            Wire(src_module, src_port, dst_module, dst_port, denature=denature, optic=optic)
        )

    def required_capabilities(self) -> set[Capability]:
        """Union of capabilities across all modules (effect aggregation)."""
        required: set[Capability] = set()
        for module in self.modules.values():
            required |= module.capabilities
        return required
