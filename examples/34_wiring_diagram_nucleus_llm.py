#!/usr/bin/env python3
"""
Example 34: Wiring Diagram - Nucleus LLM Integration
====================================================

Demonstrates a nucleus-centric wiring diagram with:
- Ingress validation and context retrieval
- Prompt assembly + LLM transcription in the nucleus
- Plan validation and approval gating
- Tool execution with feedback to memory and the response channel

This does not execute anything; it only validates wiring constraints.

Mermaid diagram:
    examples/wiring_diagrams/example34_nucleus_llm.md
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
            name="memory_writer",
            inputs={"result": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
            outputs={"update": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
            capabilities={Capability.WRITE_FS},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="episodic_store",
            inputs={"update": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
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

    # LLM output must be validated before it can drive tools.
    try:
        diagram.connect("nucleus_llm", "draft_plan", "tool_builder", "plan")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")

    diagram.connect("nucleus_llm", "draft_plan", "plan_validator", "draft")
    diagram.connect("plan_validator", "plan", "policy_gate", "plan")
    diagram.connect("plan_validator", "plan", "tool_builder", "plan")

    # Draft replies must be sanitized before user-facing output.
    try:
        diagram.connect("nucleus_llm", "draft_reply", "response_merger", "reply")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")

    diagram.connect("nucleus_llm", "draft_reply", "response_sanitizer", "draft")
    diagram.connect("response_sanitizer", "reply", "response_merger", "reply")

    diagram.connect("tool_builder", "action", "executor", "action")
    diagram.connect("policy_gate", "approval", "executor", "approval")

    diagram.connect("executor", "result", "memory_writer", "result")
    diagram.connect("memory_writer", "update", "episodic_store", "update")
    diagram.connect("executor", "result", "response_merger", "result")
    diagram.connect("response_merger", "response", "outbox", "response")

    print("✅ Wiring accepted")
    required = sorted(diagram.required_capabilities(), key=lambda c: c.value)
    print("Required capabilities:", [cap.value for cap in required])


if __name__ == "__main__":
    main()
