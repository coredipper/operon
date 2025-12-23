#!/usr/bin/env python3
"""
Example 27: Wiring Diagram - Resource Allocator
===============================================

Demonstrates a resource allocation wiring diagram with:
- Three resource sensors feeding a shared budget
- Integrity-gated validation before aggregation
- Allocation fanout to multiple objectives
- Approval-gated execution sinks
- Capability aggregation across effectful modules

This does not execute anything; it only validates wiring constraints.

Mermaid diagram:
    examples/wiring_diagrams/example27_resource_allocator.md
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
            name="nutrient_sensor",
            outputs={"nutrients": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="machinery_sensor",
            outputs={"machinery": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="energy_sensor",
            outputs={"energy": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="nutrient_validator",
            inputs={"raw": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
            outputs={"checked": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="machinery_validator",
            inputs={"raw": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
            outputs={"checked": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="energy_validator",
            inputs={"raw": PortType(DataType.JSON, IntegrityLabel.UNTRUSTED)},
            outputs={"checked": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="budget_aggregator",
            inputs={
                "nutrients": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "machinery": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "energy": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
            },
            outputs={"budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="allocator",
            inputs={"budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
            outputs={
                "growth": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "maintenance": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
                "specialization": PortType(DataType.JSON, IntegrityLabel.VALIDATED),
            },
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="policy",
            inputs={"budget": PortType(DataType.JSON, IntegrityLabel.TRUSTED)},
            outputs={"approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED)},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="growth_executor",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
            capabilities={Capability.WRITE_FS},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="maintenance_executor",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
            capabilities={Capability.READ_FS},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="specialization_executor",
            inputs={"plan": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED)},
            capabilities={Capability.NET},
        )
    )

    diagram.add_module(
        ModuleSpec(
            name="growth_sink",
            inputs={
                "action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED),
                "approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED),
            },
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="maintenance_sink",
            inputs={
                "action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED),
                "approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED),
            },
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="specialization_sink",
            inputs={
                "action": PortType(DataType.TOOL_CALL, IntegrityLabel.VALIDATED),
                "approval": PortType(DataType.APPROVAL, IntegrityLabel.TRUSTED),
            },
        )
    )

    # Integrity mismatch (UNTRUSTED -> VALIDATED) to show validation requirement.
    try:
        diagram.connect("nutrient_sensor", "nutrients", "budget_aggregator", "nutrients")
    except WiringError as exc:
        print(f"Expected wiring error: {exc}")

    diagram.connect("nutrient_sensor", "nutrients", "nutrient_validator", "raw")
    diagram.connect("machinery_sensor", "machinery", "machinery_validator", "raw")
    diagram.connect("energy_sensor", "energy", "energy_validator", "raw")

    diagram.connect("nutrient_validator", "checked", "budget_aggregator", "nutrients")
    diagram.connect("machinery_validator", "checked", "budget_aggregator", "machinery")
    diagram.connect("energy_validator", "checked", "budget_aggregator", "energy")

    diagram.connect("budget_aggregator", "budget", "allocator", "budget")
    diagram.connect("budget_aggregator", "budget", "policy", "budget")

    diagram.connect("allocator", "growth", "growth_executor", "plan")
    diagram.connect("allocator", "maintenance", "maintenance_executor", "plan")
    diagram.connect("allocator", "specialization", "specialization_executor", "plan")

    diagram.connect("growth_executor", "action", "growth_sink", "action")
    diagram.connect("maintenance_executor", "action", "maintenance_sink", "action")
    diagram.connect("specialization_executor", "action", "specialization_sink", "action")

    diagram.connect("policy", "approval", "growth_sink", "approval")
    diagram.connect("policy", "approval", "maintenance_sink", "approval")
    diagram.connect("policy", "approval", "specialization_sink", "approval")

    print("✅ Wiring accepted")
    required = sorted(diagram.required_capabilities(), key=lambda c: c.value)
    print("Required capabilities:", [cap.value for cap in required])


if __name__ == "__main__":
    main()
