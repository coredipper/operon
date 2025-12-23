#!/usr/bin/env python3
"""
Example 33: Wiring Diagram Execution - Failure Cases
====================================================

Demonstrates runtime failures in the wiring executor:
- Missing approval input
- Output type mismatch

Mermaid diagram:
    examples/wiring_diagrams/example33_execution_failures.md
"""

from __future__ import annotations

import json

from operon_ai import (
    Capability,
    DataType,
    DiagramExecutor,
    IntegrityLabel,
    ModuleSpec,
    PortType,
    TypedValue,
    Wire,
    WiringDiagram,
    WiringError,
)


def build_diagram() -> WiringDiagram:
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
    diagram.connect("validator", "clean", "planner", "prompt")
    diagram.connect("planner", "plan", "tool_builder", "plan")
    diagram.connect("planner", "plan", "policy", "plan")
    diagram.connect("tool_builder", "action", "sink", "action")
    diagram.connect("policy", "approval", "sink", "approval")

    return diagram


def register_common(executor: DiagramExecutor) -> None:
    executor.register_module("user", lambda _inputs: {
        "request": "Generate deployment report",
    })
    executor.register_module("validator", lambda inputs: {
        "clean": inputs["raw"].value.strip(),
    })
    executor.register_module("planner", lambda inputs: {
        "plan": {
            "task": inputs["prompt"].value,
            "steps": ["collect", "summarize"],
        }
    })


def run_missing_approval() -> None:
    print("\nCase 1: Missing approval handler")
    executor = DiagramExecutor(build_diagram())
    register_common(executor)

    executor.register_module("tool_builder", lambda inputs: {
        "action": {
            "tool": "fetch_report",
            "args": inputs["plan"].value,
        }
    })

    try:
        executor.execute(enforce_static_checks=False)
    except WiringError as exc:
        print("Expected error:", exc)


def run_output_mismatch() -> None:
    print("\nCase 2: Output type mismatch")
    executor = DiagramExecutor(build_diagram())
    register_common(executor)

    executor.register_module("tool_builder", lambda inputs: {
        "action": {
            "tool": "fetch_report",
            "args": inputs["plan"].value,
        }
    })

    executor.register_module("policy", lambda inputs: {
        "approval": TypedValue(
            DataType.TEXT,
            IntegrityLabel.TRUSTED,
            json.dumps({"plan": inputs["plan"].value}),
        ),
    })

    executor.register_module("sink", lambda _inputs: {})

    try:
        executor.execute()
    except WiringError as exc:
        print("Expected error:", exc)


def run_multiple_sources() -> None:
    print("\nCase 3: Multiple sources for one input")
    diagram = build_diagram()

    diagram.add_module(
        ModuleSpec(
            name="shadow_planner",
            outputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.connect("shadow_planner", "plan", "policy", "plan")

    executor = DiagramExecutor(diagram)
    register_common(executor)

    executor.register_module("tool_builder", lambda inputs: {
        "action": {
            "tool": "fetch_report",
            "args": inputs["plan"].value,
        }
    })
    executor.register_module("policy", lambda inputs: {
        "approval": TypedValue(
            DataType.APPROVAL,
            IntegrityLabel.TRUSTED,
            {"policy": "multi-source", "plan": inputs["plan"].value},
        ),
    })
    executor.register_module("sink", lambda _inputs: {})

    try:
        executor.execute()
    except WiringError as exc:
        print("Expected error:", exc)


def run_missing_input() -> None:
    print("\nCase 4: Missing input source")
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
            capabilities={Capability.NET},
        )
    )
    diagram.connect("planner", "plan", "tool_builder", "plan")

    executor = DiagramExecutor(diagram)
    executor.register_module("planner", lambda _inputs: {
        "plan": {"task": "no upstream prompt"},
    })
    executor.register_module("tool_builder", lambda inputs: {
        "action": {
            "tool": "fetch_report",
            "args": inputs["plan"].value,
        }
    })

    try:
        executor.execute()
    except WiringError as exc:
        print("Expected error:", exc)


def run_integrity_violation() -> None:
    print("\nCase 5: Integrity violation at runtime")
    diagram = WiringDiagram()

    diagram.add_module(
        ModuleSpec(
            name="producer",
            outputs={"output": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="trusted_sink",
            inputs={"input": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
        )
    )

    diagram.wires.append(
        Wire(
            src_module="producer",
            src_port="output",
            dst_module="trusted_sink",
            dst_port="input",
        )
    )

    executor = DiagramExecutor(diagram)
    executor.register_module("producer", lambda _inputs: {
        "output": TypedValue(
            DataType.TEXT,
            IntegrityLabel.VALIDATED,
            "validated text",
        )
    })
    executor.register_module("trusted_sink", lambda _inputs: {})

    try:
        executor.execute()
    except WiringError as exc:
        print("Expected error:", exc)


def main() -> None:
    run_missing_approval()
    run_output_mismatch()
    run_multiple_sources()
    run_missing_input()
    run_integrity_violation()


if __name__ == "__main__":
    main()
