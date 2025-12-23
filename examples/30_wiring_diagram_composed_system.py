#!/usr/bin/env python3
"""
Example 30: Wiring Diagram - Composed System
============================================

Demonstrates composing two wiring diagrams into a larger system:
- Ingress pipeline: user -> membrane -> sanitizer
- Execution pipeline: planner -> tool_builder + policy -> sink
- Composition with namespacing and cross-diagram connections

This does not execute anything; it only validates wiring constraints.
"""

from operon_ai import (
    Capability,
    DataType,
    IntegrityLabel,
    ModuleSpec,
    PortType,
    Wire,
    WiringDiagram,
    WiringError,
)


def build_ingress_diagram() -> WiringDiagram:
    diagram = WiringDiagram()

    diagram.add_module(
        ModuleSpec(
            name="user",
            outputs={"request": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="membrane",
            inputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"screened": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="sanitizer",
            inputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"clean": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.connect("user", "request", "membrane", "raw")
    diagram.connect("membrane", "screened", "sanitizer", "raw")

    return diagram


def build_execution_diagram() -> WiringDiagram:
    diagram = WiringDiagram()

    diagram.add_module(
        ModuleSpec(
            name="planner",
            inputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="tool_builder",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
            capabilities={Capability.WRITE_FS},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="policy",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="sink",
            inputs={
                "action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED),
                "approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED),
            },
        )
    )

    diagram.connect("planner", "plan", "tool_builder", "plan")
    diagram.connect("planner", "plan", "policy", "plan")
    diagram.connect("tool_builder", "action", "sink", "action")
    diagram.connect("policy", "approval", "sink", "approval")

    return diagram


def add_diagram(target: WiringDiagram, source: WiringDiagram, prefix: str) -> None:
    for module in source.modules.values():
        target.add_module(
            ModuleSpec(
                name=f"{prefix}{module.name}",
                inputs=module.inputs,
                outputs=module.outputs,
                capabilities=set(module.capabilities),
            )
        )

    for wire in source.wires:
        target.wires.append(
            Wire(
                src_module=f"{prefix}{wire.src_module}",
                src_port=wire.src_port,
                dst_module=f"{prefix}{wire.dst_module}",
                dst_port=wire.dst_port,
            )
        )


def main() -> None:
    ingress = build_ingress_diagram()
    execution = build_execution_diagram()

    composed = WiringDiagram()
    add_diagram(composed, ingress, "ingress.")
    add_diagram(composed, execution, "exec.")

    # Demonstrate a failed cross-diagram connection (untrusted -> validated).
    try:
        composed.connect("ingress.user", "request", "exec.planner", "prompt")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")

    # Correct wiring using the validated sanitizer output.
    composed.connect("ingress.sanitizer", "clean", "exec.planner", "prompt")

    print("✅ Wiring accepted")
    required = sorted(composed.required_capabilities(), key=lambda c: c.value)
    print("Required capabilities:", [cap.value for cap in required])


if __name__ == "__main__":
    main()
