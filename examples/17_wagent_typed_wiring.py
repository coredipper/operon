"""
Example 17: WAgent Typed Wiring (Integrity Labels + Capabilities)
=================================================================

Demonstrates a lightweight, runtime-checkable version of the paper's WAgent idea.
This does not execute anything; it only validates wiring constraints.

Key concepts:
- Ports have a (DataType, IntegrityLabel)
- Wires must match types and preserve integrity (UNTRUSTED ≤ VALIDATED ≤ TRUSTED)
- Modules carry capability/effect tags (least-privilege bookkeeping)
- Runtime verification of type safety and information flow

Prerequisites:
- None (introductory example)

Usage:
    python examples/17_wagent_typed_wiring.py

Related resources:
    examples/wiring_diagrams/example17_typed_wiring.md
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


def main():
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
            outputs={"out": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="chaperone",
            inputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"validated": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="executor",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
            capabilities={Capability.WRITE_FS},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="verifier",
            inputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
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

    # Wire the diagram
    diagram.connect("user", "request", "membrane", "in")
    diagram.connect("membrane", "out", "chaperone", "raw")
    diagram.connect("chaperone", "validated", "executor", "plan")
    diagram.connect("executor", "action", "verifier", "action")
    diagram.connect("executor", "action", "sink", "action")
    diagram.connect("verifier", "approval", "sink", "approval")

    print("✅ Wiring accepted")
    print(f"Required capabilities: {[c.value for c in sorted(diagram.required_capabilities(), key=lambda c: c.value)]}")

    # Demonstrate a wiring error (untrusted -> trusted)
    try:
        diagram.connect("user", "request", "sink", "approval")
    except WiringError as e:
        print(f"🧯 Expected wiring error: {e}")


if __name__ == "__main__":
    main()
