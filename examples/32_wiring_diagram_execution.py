#!/usr/bin/env python3
"""
Example 32: Wiring Diagram Execution
====================================

Shows a minimal runtime executor that runs a typed wiring diagram
with concrete module handlers.

Mermaid diagram:
    examples/wiring_diagrams/example32_execution.md
"""

from __future__ import annotations

import hashlib
import json

from operon_ai import (
    ApprovalToken,
    Capability,
    DataType,
    DiagramExecutor,
    IntegrityLabel,
    ModuleSpec,
    PortType,
    WiringDiagram,
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

    diagram.connect("user", "request", "validator", "raw")
    diagram.connect("validator", "clean", "planner", "prompt")
    diagram.connect("planner", "plan", "tool_builder", "plan")
    diagram.connect("planner", "plan", "policy", "plan")
    diagram.connect("tool_builder", "action", "sink", "action")
    diagram.connect("policy", "approval", "sink", "approval")

    executor = DiagramExecutor(diagram)

    executor.register_module("user", lambda _inputs: {
        "request": "Rotate service logs and archive to /var/logs",
    })

    executor.register_module("validator", lambda inputs: {
        "clean": inputs["raw"].value.strip(),
    })

    def plan_handler(inputs):
        return {
            "plan": {
                "task": inputs["prompt"].value,
                "steps": ["precheck", "execute", "verify"],
            }
        }

    executor.register_module("planner", plan_handler)

    executor.register_module("tool_builder", lambda inputs: {
        "action": {
            "tool": "runbook",
            "args": inputs["plan"].value,
        }
    })

    def policy_handler(inputs):
        payload = json.dumps(inputs["plan"].value, sort_keys=True)
        request_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return {
            "approval": ApprovalToken(
                request_hash=request_hash,
                issuer="policy",
                reason="validated plan",
            )
        }

    executor.register_module("policy", policy_handler)

    def sink_handler(inputs):
        action = inputs["action"].value
        approval = inputs["approval"].value
        print("Sink received action:", action)
        print("Approval token:", approval.request_hash, approval.issuer)
        return {}

    executor.register_module("sink", sink_handler)

    report = executor.execute()
    print("Execution order:", " -> ".join(report.execution_order))


if __name__ == "__main__":
    main()
