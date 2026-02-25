"""
Example 63: Optic-Based Wiring — Lens, Prism, Traversal
=========================================================

Demonstrates optic-based wire routing and transformation,
implementing Paper §3.3.

Biological Analogy:
- Lens  = constitutive expression — always active, data passes through
- Prism = receptor specificity — only responds to matching ligand type
- Traversal = polymerase processivity — walks a sequence, transforms each element

Key points:
1. LensOptic is explicit pass-through (equivalent to no optic)
2. PrismOptic routes by DataType — fan-out from one port to prism-filtered wires
3. TraversalOptic maps a transform over list elements on the wire
4. ComposedOptic chains optics sequentially
5. Fully backward compatible — wires without optics work unchanged
6. Optics coexist with DenatureFilters on the same wire

References:
- Article Section 3.3: Optic-Based Wiring
"""

from operon_ai.core.optics import (
    ComposedOptic,
    LensOptic,
    Optic,
    PrismOptic,
    TraversalOptic,
)
from operon_ai.core.types import DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, WiringDiagram
from operon_ai.core.wiring_runtime import DiagramExecutor, TypedValue


def main():
    try:
        print("=" * 60)
        print("Optic-Based Wiring")
        print("=" * 60)

        # ── 1. LensOptic pass-through ──────────────────────────────
        print("\n--- 1. LensOptic (Pass-Through) ---")
        lens = LensOptic()
        print(f"  Protocol: {isinstance(lens, Optic)}")
        print(f"  can_transmit(JSON): {lens.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)}")
        print(f"  transmit('hello'): {lens.transmit('hello', DataType.JSON, IntegrityLabel.VALIDATED)}")

        # ── 2. PrismOptic conditional routing ──────────────────────
        print("\n--- 2. PrismOptic (Conditional Routing) ---")
        json_prism = PrismOptic(accept=frozenset({DataType.JSON}))
        error_prism = PrismOptic(accept=frozenset({DataType.ERROR}))
        print(f"  JSON prism accepts JSON:  {json_prism.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)}")
        print(f"  JSON prism accepts ERROR: {json_prism.can_transmit(DataType.ERROR, IntegrityLabel.VALIDATED)}")
        print(f"  ERROR prism accepts JSON: {error_prism.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)}")
        print(f"  ERROR prism accepts ERROR: {error_prism.can_transmit(DataType.ERROR, IntegrityLabel.VALIDATED)}")

        # Fan-out routing: A -> B (JSON only), A -> C (ERROR only)
        print("\n  Fan-out routing example:")
        pt_json = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        pt_err = PortType(DataType.ERROR, IntegrityLabel.VALIDATED)

        diagram = WiringDiagram()
        diagram.add_module(ModuleSpec(name="Router", inputs={"in": pt_json}, outputs={"out": pt_json}))
        diagram.add_module(ModuleSpec(name="JSONHandler", inputs={"in": pt_json}))
        diagram.add_module(ModuleSpec(name="ErrorHandler", inputs={"in": pt_err}))
        diagram.connect("Router", "out", "JSONHandler", "in", optic=json_prism)
        diagram.connect("Router", "out", "ErrorHandler", "in", optic=error_prism)

        executor = DiagramExecutor(diagram)
        executor.register_module("Router", lambda inputs: {
            "out": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, {"status": "ok"})
        })
        report = executor.execute(
            external_inputs={"Router": {"in": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, "start")}},
            enforce_static_checks=False,
        )
        print(f"  JSON output -> JSONHandler received: {report.modules['JSONHandler'].inputs.get('in', 'nothing')}")
        print(f"  JSON output -> ErrorHandler received: {report.modules['ErrorHandler'].inputs.get('in', 'nothing')}")

        # ── 3. TraversalOptic collection processing ────────────────
        print("\n--- 3. TraversalOptic (Collection Processing) ---")
        doubler = TraversalOptic(transform=lambda x: x * 2)
        print(f"  transmit([1,2,3]): {doubler.transmit([1, 2, 3], DataType.JSON, IntegrityLabel.VALIDATED)}")
        print(f"  transmit(5):       {doubler.transmit(5, DataType.JSON, IntegrityLabel.VALIDATED)}")

        upper = TraversalOptic(transform=str.upper)
        print(f"  transmit(['a','b']): {upper.transmit(['a', 'b'], DataType.JSON, IntegrityLabel.VALIDATED)}")

        # ── 4. ComposedOptic ───────────────────────────────────────
        print("\n--- 4. ComposedOptic (Sequential Composition) ---")
        composed = ComposedOptic(optics=(
            PrismOptic(accept=frozenset({DataType.JSON})),
            TraversalOptic(transform=lambda x: x * 10),
        ))
        print(f"  name: {composed.name}")
        print(f"  can_transmit(JSON):  {composed.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)}")
        print(f"  can_transmit(ERROR): {composed.can_transmit(DataType.ERROR, IntegrityLabel.VALIDATED)}")
        print(f"  transmit([1,2,3]):   {composed.transmit([1, 2, 3], DataType.JSON, IntegrityLabel.VALIDATED)}")

        # ── 5. Backward compatibility ──────────────────────────────
        print("\n--- 5. Backward Compatibility ---")
        pt = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        d2 = WiringDiagram()
        d2.add_module(ModuleSpec(name="A", inputs={"in": pt}, outputs={"out": pt}))
        d2.add_module(ModuleSpec(name="B", inputs={"in": pt}))
        d2.connect("A", "out", "B", "in")  # No optic
        ex2 = DiagramExecutor(d2)
        ex2.register_module("A", lambda inputs: {"out": {"data": 42}})
        r2 = ex2.execute(external_inputs={"A": {"in": "trigger"}})
        print(f"  No optic, wire works as before: {r2.modules['B'].inputs['in'].value}")

        # ── 6. Optic + DenatureFilter coexistence ──────────────────
        print("\n--- 6. Optic + DenatureFilter Coexistence ---")
        from operon_ai.core.denature import SummarizeFilter

        pt_text = PortType(DataType.TEXT, IntegrityLabel.VALIDATED)
        d3 = WiringDiagram()
        d3.add_module(ModuleSpec(name="Source", inputs={"in": pt_text}, outputs={"out": pt_text}))
        d3.add_module(ModuleSpec(name="Sink", inputs={"in": pt_text}))
        d3.connect(
            "Source", "out", "Sink", "in",
            denature=SummarizeFilter(max_length=20),
            optic=TraversalOptic(transform=str.upper),
        )
        ex3 = DiagramExecutor(d3)
        ex3.register_module("Source", lambda inputs: {"out": "This is a long input that should be summarized first"})
        r3 = ex3.execute(external_inputs={"Source": {"in": "trigger"}})
        result = r3.modules["Sink"].inputs["in"].value
        print(f"  Denature then optic: '{result}'")
        print(f"  (SummarizeFilter truncated, then TraversalOptic uppercased)")

        print("\n" + "=" * 60)
        print("Optic-based wiring demo complete.")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
