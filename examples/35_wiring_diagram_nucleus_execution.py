#!/usr/bin/env python3
"""
Example 35: Wiring Diagram Execution - Nucleus LLM Integration
==============================================================

Executes a wiring diagram that routes a request through the nucleus
LLM integration and into guarded tool execution.

Mermaid diagram:
    examples/wiring_diagrams/example35_nucleus_execution.md
"""

from __future__ import annotations

import hashlib
import json

from operon_ai import (
    ApprovalToken,
    Capability,
    DataType,
    DiagramExecutor,
    GeminiProvider,
    IntegrityLabel,
    ModuleSpec,
    Nucleus,
    PortType,
    WiringDiagram,
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
            name="membrane",
            inputs={"in": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"raw": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
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
            name="context_retriever",
            inputs={"query": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={"context": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            capabilities={Capability.READ_FS},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="genome_policy",
            outputs={"policy": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="tool_registry",
            outputs={"schemas": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="prompt_assembler",
            inputs={
                "query": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "context": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "policy": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
                "tools": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="nucleus_llm",
            inputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
            outputs={
                "draft_plan": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED),
                "draft_reply": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED),
            },
            capabilities={Capability.NET},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="plan_validator",
            inputs={"draft": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
            outputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="response_sanitizer",
            inputs={"draft": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"reply": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="policy_gate",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="tool_builder",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
            capabilities={Capability.NET, Capability.WRITE_FS},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="executor",
            inputs={
                "action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED),
                "approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED),
            },
            outputs={"result": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
            capabilities={Capability.NET, Capability.WRITE_FS},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="response_merger",
            inputs={
                "reply": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "result": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"response": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="outbox",
            inputs={"response": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.connect("user", "request", "membrane", "in")
    diagram.connect("membrane", "raw", "sanitizer", "raw")
    diagram.connect("sanitizer", "clean", "context_retriever", "query")
    diagram.connect("sanitizer", "clean", "prompt_assembler", "query")
    diagram.connect("context_retriever", "context", "prompt_assembler", "context")
    diagram.connect("genome_policy", "policy", "prompt_assembler", "policy")
    diagram.connect("tool_registry", "schemas", "prompt_assembler", "tools")
    diagram.connect("prompt_assembler", "prompt", "nucleus_llm", "prompt")
    diagram.connect("nucleus_llm", "draft_plan", "plan_validator", "draft")
    diagram.connect("nucleus_llm", "draft_reply", "response_sanitizer", "draft")
    diagram.connect("plan_validator", "plan", "policy_gate", "plan")
    diagram.connect("plan_validator", "plan", "tool_builder", "plan")
    diagram.connect("tool_builder", "action", "executor", "action")
    diagram.connect("policy_gate", "approval", "executor", "approval")
    diagram.connect("executor", "result", "response_merger", "result")
    diagram.connect("response_sanitizer", "reply", "response_merger", "reply")
    diagram.connect("response_merger", "response", "outbox", "response")

    return diagram


def _extract_request(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Request:"):
            return line.split("Request:", 1)[1].strip()
    return "unknown request"


def main() -> None:
    diagram = build_diagram()
    executor = DiagramExecutor(diagram)

    nucleus = Nucleus(provider=GeminiProvider())

    executor.register_module("user", lambda _inputs: {
        "request": "Generate a deployment summary for the last release.",
    })

    executor.register_module("membrane", lambda inputs: {
        "raw": inputs["in"].value,
    })

    executor.register_module("sanitizer", lambda inputs: {
        "clean": inputs["raw"].value.strip(),
    })

    executor.register_module("context_retriever", lambda inputs: {
        "context": {
            "recent_release": "v2.4.1",
            "notes": ["auth patch", "queue tuning"],
            "query": inputs["query"].value,
        }
    })

    executor.register_module("genome_policy", lambda _inputs: {
        "policy": {
            "approval": "two-key",
            "safety": "no destructive changes",
        }
    })

    executor.register_module("tool_registry", lambda _inputs: {
        "schemas": [
            {"name": "runbook", "desc": "Execute a runbook step."},
            {"name": "release_report", "desc": "Fetch release metadata."},
        ]
    })

    def prompt_assembler(inputs):
        tool_names = ", ".join(tool["name"] for tool in inputs["tools"].value)
        prompt = (
            "You are the nucleus coordinating a deployment summary.\n"
            f"Request: {inputs['query'].value}\n"
            f"Context: {json.dumps(inputs['context'].value, sort_keys=True)}\n"
            f"Policy: {json.dumps(inputs['policy'].value, sort_keys=True)}\n"
            f"Tools: {tool_names}\n"
            "Generate a plan and a short reply."
        )
        return {"prompt": prompt}

    executor.register_module("prompt_assembler", prompt_assembler)

    def nucleus_handler(inputs):
        prompt = inputs["prompt"].value
        response = nucleus.transcribe(prompt)
        request = _extract_request(prompt)
        draft_plan = {
            "task": request,
            "steps": ["collect", "summarize", "verify"],
            "llm_notes": response.content,
        }
        return {
            "draft_plan": draft_plan,
            "draft_reply": response.content,
        }

    executor.register_module("nucleus_llm", nucleus_handler)

    executor.register_module("plan_validator", lambda inputs: {
        "plan": {
            **inputs["draft"].value,
            "validated": True,
        }
    })

    executor.register_module("response_sanitizer", lambda inputs: {
        "reply": inputs["draft"].value.strip(),
    })

    executor.register_module("tool_builder", lambda inputs: {
        "action": {
            "tool": "release_report",
            "args": inputs["plan"].value,
        }
    })

    def policy_handler(inputs):
        payload = json.dumps(inputs["plan"].value, sort_keys=True)
        request_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return {
            "approval": ApprovalToken(
                request_hash=request_hash,
                issuer="policy_gate",
                reason="validated plan",
            )
        }

    executor.register_module("policy_gate", policy_handler)

    def executor_handler(inputs):
        action = inputs["action"].value
        approval = inputs["approval"].value
        result = {
            "status": "completed",
            "tool": action["tool"],
            "request_hash": approval.request_hash,
        }
        return {"result": result}

    executor.register_module("executor", executor_handler)

    def response_merger(inputs):
        reply = inputs["reply"].value
        result = json.dumps(inputs["result"].value, sort_keys=True)
        response = f"{reply}\n\nResult: {result}"
        return {"response": response}

    executor.register_module("response_merger", response_merger)

    def outbox_handler(inputs):
        print("Final response:")
        print(inputs["response"].value)
        return {}

    executor.register_module("outbox", outbox_handler)

    report = executor.execute()
    print("Execution order:", " -> ".join(report.execution_order))


if __name__ == "__main__":
    main()
