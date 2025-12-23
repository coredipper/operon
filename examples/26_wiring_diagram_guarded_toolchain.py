#!/usr/bin/env python3
"""
Example 26: Wiring Diagram - Guarded Toolchain
==============================================

Shows rigorous WAgent wiring constraints:
- Integrity upgrades via explicit validator/attestor modules
- Type adapters for JSON <-> TEXT
- Approval-gated tool execution
- Capability aggregation across modules

This does not execute anything; it only validates wiring constraints.

Mermaid diagram:
    examples/wiring_diagrams/example26_guarded_toolchain.md
"""

from operon_ai import (
    Capability,
    DataType,
    IntegrityLabel,
    ModuleSpec,
    PortType,
    WiringDiagram,
    WiringError,
)


def main() -> None:
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
            inputs={"in": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="validator",
            inputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"clean": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="parser",
            inputs={"text": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="planner",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"refined": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="serializer",
            inputs={"data": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"text": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="policy",
            inputs={"summary": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED)},
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
            name="attestor",
            inputs={"clean": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"trusted": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="operator_console",
            inputs={"message": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
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

    diagram.connect("user", "request", "membrane", "in")
    diagram.connect("membrane", "raw", "validator", "raw")
    diagram.connect("validator", "clean", "parser", "text")
    diagram.connect("parser", "plan", "planner", "plan")

    # Demonstrate a type mismatch (JSON -> TEXT) and fix via serializer.
    try:
        diagram.connect("planner", "refined", "policy", "summary")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")
    diagram.connect("planner", "refined", "serializer", "data")
    diagram.connect("serializer", "text", "policy", "summary")

    # Demonstrate an integrity violation (VALIDATED -> TRUSTED).
    try:
        diagram.connect("validator", "clean", "operator_console", "message")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")
    diagram.connect("validator", "clean", "attestor", "clean")
    diagram.connect("attestor", "trusted", "operator_console", "message")

    # Guarded tool execution requires approval + action.
    diagram.connect("planner", "refined", "tool_builder", "plan")
    diagram.connect("tool_builder", "action", "sink", "action")
    diagram.connect("policy", "approval", "sink", "approval")

    print("✅ Wiring accepted")
    required = sorted(diagram.required_capabilities(), key=lambda c: c.value)
    print("Required capabilities:", [cap.value for cap in required])


if __name__ == "__main__":
    main()
