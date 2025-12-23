#!/usr/bin/env python3
"""
Example 28: Wiring Diagram - Quorum Consensus Gate
===================================================

Demonstrates a wiring diagram for multi-agent consensus:
- Inputs validated before voting
- Multiple voter outputs aggregated into a trusted approval token
- Approval-gated tool execution
- Capability aggregation across voters/executors

This does not execute anything; it only validates wiring constraints.

Mermaid diagram:
    examples/wiring_diagrams/example28_quorum_consensus.md
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
            name="sanitizer",
            inputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"clean": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="voter_a",
            inputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"vote": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="voter_b",
            inputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"vote": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            capabilities={Capability.NET},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="voter_c",
            inputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"vote": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="quorum",
            inputs={
                "vote_a": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "vote_b": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "vote_c": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
            },
            outputs={"approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="tool_builder",
            inputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
            capabilities={Capability.WRITE_FS},
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

    diagram.connect("user", "request", "sanitizer", "raw")

    # Demonstrate integrity violation when skipping validation.
    try:
        diagram.connect("user", "request", "tool_builder", "prompt")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")

    diagram.connect("sanitizer", "clean", "voter_a", "prompt")
    diagram.connect("sanitizer", "clean", "voter_b", "prompt")
    diagram.connect("sanitizer", "clean", "voter_c", "prompt")

    diagram.connect("voter_a", "vote", "quorum", "vote_a")
    diagram.connect("voter_b", "vote", "quorum", "vote_b")
    diagram.connect("voter_c", "vote", "quorum", "vote_c")

    diagram.connect("sanitizer", "clean", "tool_builder", "prompt")
    diagram.connect("tool_builder", "action", "sink", "action")
    diagram.connect("quorum", "approval", "sink", "approval")

    print("✅ Wiring accepted")
    required = sorted(diagram.required_capabilities(), key=lambda c: c.value)
    print("Required capabilities:", [cap.value for cap in required])


if __name__ == "__main__":
    main()
