#!/usr/bin/env python3
"""
Example 36: Wiring Diagram Execution - Multi-Gemini Resource Allocation
=======================================================================

Allocates a shared token budget across multiple Gemini-powered agents,
then aggregates their drafts into a validated plan with approval gating.

Requires:
    GEMINI_API_KEY=... and `pip install google-genai`

Mermaid diagram:
    examples/wiring_diagrams/example36_multi_gemini_allocation.md
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
    ProviderConfig,
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
            name="resource_monitor",
            outputs={"resources": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="budget_allocator",
            inputs={
                "request": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "resources": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={
                "fast_budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
                "deep_budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
                "safety_budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="prompt_fast",
            inputs={
                "request": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="prompt_deep",
            inputs={
                "request": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="prompt_safety",
            inputs={
                "request": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="nucleus_fast",
            inputs={
                "prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"draft": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
            capabilities={Capability.NET},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="nucleus_deep",
            inputs={
                "prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"draft": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
            capabilities={Capability.NET},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="nucleus_safety",
            inputs={
                "prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED),
            },
            outputs={"draft": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
            capabilities={Capability.NET},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="plan_aggregator",
            inputs={
                "request": PortType(DataType.TEXT, IntegrityLabel.VALIDATED),
                "fast": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED),
                "deep": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED),
                "safety": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED),
            },
            outputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
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
            name="response_builder",
            inputs={
                "plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED),
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
    diagram.connect("sanitizer", "clean", "budget_allocator", "request")
    diagram.connect("resource_monitor", "resources", "budget_allocator", "resources")

    diagram.connect("sanitizer", "clean", "prompt_fast", "request")
    diagram.connect("budget_allocator", "fast_budget", "prompt_fast", "budget")
    diagram.connect("sanitizer", "clean", "prompt_deep", "request")
    diagram.connect("budget_allocator", "deep_budget", "prompt_deep", "budget")
    diagram.connect("sanitizer", "clean", "prompt_safety", "request")
    diagram.connect("budget_allocator", "safety_budget", "prompt_safety", "budget")

    diagram.connect("prompt_fast", "prompt", "nucleus_fast", "prompt")
    diagram.connect("budget_allocator", "fast_budget", "nucleus_fast", "budget")
    diagram.connect("prompt_deep", "prompt", "nucleus_deep", "prompt")
    diagram.connect("budget_allocator", "deep_budget", "nucleus_deep", "budget")
    diagram.connect("prompt_safety", "prompt", "nucleus_safety", "prompt")
    diagram.connect("budget_allocator", "safety_budget", "nucleus_safety", "budget")

    diagram.connect("nucleus_fast", "draft", "plan_aggregator", "fast")
    diagram.connect("nucleus_deep", "draft", "plan_aggregator", "deep")
    diagram.connect("nucleus_safety", "draft", "plan_aggregator", "safety")
    diagram.connect("sanitizer", "clean", "plan_aggregator", "request")

    diagram.connect("plan_aggregator", "plan", "policy_gate", "plan")
    diagram.connect("plan_aggregator", "plan", "response_builder", "plan")
    diagram.connect("policy_gate", "approval", "response_builder", "approval")
    diagram.connect("response_builder", "response", "outbox", "response")

    return diagram


def _allocate_budgets(total_tokens: int, weights: dict[str, float]) -> dict[str, int]:
    budgets = {name: int(total_tokens * weight) for name, weight in weights.items()}
    remainder = total_tokens - sum(budgets.values())
    if remainder > 0:
        largest = max(budgets, key=budgets.get)
        budgets[largest] += remainder
    return budgets


def _build_prompt(role: str, request: str, budget: dict) -> str:
    return (
        f"Role: {role}\n"
        f"Request: {request}\n"
        f"Budget: {budget.get('max_tokens')} tokens\n"
        "Return a concise plan or risk assessment in JSON-like bullets."
    )


def _summarize(content: str, limit: int = 140) -> str:
    trimmed = " ".join(content.split())
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: limit - 3] + "..."


def main() -> None:
    diagram = build_diagram()
    executor = DiagramExecutor(diagram)

    nucleus_fast = Nucleus(provider=GeminiProvider())
    nucleus_deep = Nucleus(provider=GeminiProvider())
    nucleus_safety = Nucleus(provider=GeminiProvider())

    executor.register_module("user", lambda _inputs: {
        "request": "Draft a deployment summary and highlight any risky changes.",
    })

    executor.register_module("membrane", lambda inputs: {
        "raw": inputs["in"].value,
    })

    executor.register_module("sanitizer", lambda inputs: {
        "clean": inputs["raw"].value.strip(),
    })

    executor.register_module("resource_monitor", lambda _inputs: {
        "resources": {
            "token_budget": 2200,
            "priority": "accuracy",
        }
    })

    def budget_allocator(inputs):
        request = inputs["request"].value
        resources = inputs["resources"].value
        total = int(resources.get("token_budget", 1800))
        length = len(request.split())
        weights = {
            "fast": 0.25 if length > 8 else 0.35,
            "deep": 0.55 if length > 8 else 0.45,
            "safety": 0.20,
        }
        budgets = _allocate_budgets(total, weights)
        return {
            "fast_budget": {"max_tokens": budgets["fast"], "temperature": 0.3},
            "deep_budget": {"max_tokens": budgets["deep"], "temperature": 0.2},
            "safety_budget": {"max_tokens": budgets["safety"], "temperature": 0.1},
        }

    executor.register_module("budget_allocator", budget_allocator)

    executor.register_module("prompt_fast", lambda inputs: {
        "prompt": _build_prompt("fast-drafter", inputs["request"].value, inputs["budget"].value),
    })
    executor.register_module("prompt_deep", lambda inputs: {
        "prompt": _build_prompt("deep-planner", inputs["request"].value, inputs["budget"].value),
    })
    executor.register_module("prompt_safety", lambda inputs: {
        "prompt": _build_prompt("safety-auditor", inputs["request"].value, inputs["budget"].value),
    })

    def make_nucleus_handler(role: str, nucleus: Nucleus):
        def handler(inputs):
            budget = inputs["budget"].value
            config = ProviderConfig(
                max_tokens=int(budget.get("max_tokens", 512)),
                temperature=float(budget.get("temperature", 0.3)),
            )
            response = nucleus.transcribe(inputs["prompt"].value, config=config)
            return {
                "draft": {
                    "role": role,
                    "budget": budget,
                    "content": response.content,
                    "tokens_used": response.tokens_used,
                    "model": response.model,
                }
            }

        return handler

    executor.register_module("nucleus_fast", make_nucleus_handler("fast", nucleus_fast))
    executor.register_module("nucleus_deep", make_nucleus_handler("deep", nucleus_deep))
    executor.register_module("nucleus_safety", make_nucleus_handler("safety", nucleus_safety))

    def plan_aggregator(inputs):
        drafts = {
            "fast": inputs["fast"].value,
            "deep": inputs["deep"].value,
            "safety": inputs["safety"].value,
        }
        summary = {key: _summarize(value.get("content", "")) for key, value in drafts.items()}
        budgets = {key: value.get("budget", {}) for key, value in drafts.items()}
        plan = {
            "request": inputs["request"].value,
            "draft_summary": summary,
            "budgets": budgets,
            "decision": "merge fast + deep, apply safety notes",
        }
        return {"plan": plan}

    executor.register_module("plan_aggregator", plan_aggregator)

    def policy_handler(inputs):
        payload = json.dumps(inputs["plan"].value, sort_keys=True)
        request_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return {
            "approval": ApprovalToken(
                request_hash=request_hash,
                issuer="policy_gate",
                reason="resource allocation validated",
            )
        }

    executor.register_module("policy_gate", policy_handler)

    def response_builder(inputs):
        plan = inputs["plan"].value
        approval = inputs["approval"].value
        response = (
            f"Plan for: {plan['request']}\n"
            f"Budgets: {json.dumps(plan['budgets'], sort_keys=True)}\n"
            f"Drafts: {json.dumps(plan['draft_summary'], sort_keys=True)}\n"
            f"Approval: {approval.request_hash}"
        )
        return {"response": response}

    executor.register_module("response_builder", response_builder)

    report = executor.execute()
    print("Execution order:", " -> ".join(report.execution_order))
    print("Final response:")
    print(report.modules["outbox"].inputs["response"].value)


if __name__ == "__main__":
    main()
