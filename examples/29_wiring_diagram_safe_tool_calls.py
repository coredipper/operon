#!/usr/bin/env python3
"""
Example 29: Wiring Diagram - Safe Tool Calls
============================================

Demonstrates a wiring diagram for approval-gated tool usage:
- Untrusted input must be validated before planning
- Plan JSON feeds tool builder and approval policy
- Sink requires validated tool call + trusted approval
- Capability aggregation for effectful tools

This does not execute anything; it only validates wiring constraints.
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
            name="validator",
            inputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"clean": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )

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
            capabilities={Capability.NET},
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

    diagram.connect("user", "request", "validator", "raw")

    # Demonstrate a type mismatch by wiring TEXT directly into a JSON plan input.
    try:
        diagram.connect("validator", "clean", "tool_builder", "plan")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")

    diagram.connect("validator", "clean", "planner", "prompt")
    diagram.connect("planner", "plan", "tool_builder", "plan")
    diagram.connect("planner", "plan", "policy", "plan")

    diagram.connect("tool_builder", "action", "sink", "action")
    diagram.connect("policy", "approval", "sink", "approval")

    print("✅ Wiring accepted")
    required = sorted(diagram.required_capabilities(), key=lambda c: c.value)
    print("Required capabilities:", [cap.value for cap in required])


if __name__ == "__main__":
    main()
