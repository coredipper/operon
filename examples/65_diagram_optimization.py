"""
Example 65: Diagram Optimization via Categorical Rewriting
===========================================================

Demonstrates cost-annotated wiring diagrams, static analysis,
optimization passes, and resource-aware execution.

Biological Analogy:
Metabolic pathway optimization — cells don't blindly run all
pathways. They analyze flux, identify bottlenecks, prune dead-end
reactions, and parallelize independent branches. Under ATP
depletion, non-essential pathways are downregulated.

Key points:
1. ResourceCost annotates modules and wires with ATP/latency/memory
2. Analyzer identifies parallelism, dead wires, critical paths, hotspots
3. Optimizer applies rewriting passes (dead wire elimination, parallel grouping)
4. ResourceAwareExecutor respects metabolic state during execution
5. BudgetOptic caps cumulative wire cost

References:
- Article Section 7: Diagram Optimization via Categorical Rewriting
"""

from operon_ai.core.analyzer import (
    critical_path,
    find_dead_wires,
    find_independent_groups,
    suggest_optimizations,
    total_cost,
)
from operon_ai.core.optics import BudgetOptic, PrismOptic
from operon_ai.core.optimizer import optimize
from operon_ai.core.types import DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram
from operon_ai.core.wiring_runtime import (
    DiagramExecutor,
    ResourceAwareExecutor,
    TypedValue,
)
from operon_ai.state.metabolism import ATP_Store


def main():
    try:
        print("=" * 60)
        print("Diagram Optimization via Categorical Rewriting")
        print("=" * 60)

        pt = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
        pt_err = PortType(DataType.ERROR, IntegrityLabel.VALIDATED)

        # ── 1. Build a cost-annotated diagram ─────────────────────────
        print("\n--- 1. Cost-Annotated Wiring Diagram ---")

        diagram = WiringDiagram()
        diagram.add_module(ModuleSpec(
            "Parser",
            inputs={"raw": pt},
            outputs={"parsed": pt},
            cost=ResourceCost(atp=5, latency_ms=10.0),
        ))
        diagram.add_module(ModuleSpec(
            "Validator",
            inputs={"data": pt},
            outputs={"valid": pt},
            cost=ResourceCost(atp=10, latency_ms=20.0),
            essential=True,
        ))
        diagram.add_module(ModuleSpec(
            "Enricher",
            inputs={"data": pt},
            outputs={"enriched": pt},
            cost=ResourceCost(atp=30, latency_ms=50.0),
            essential=False,
        ))
        diagram.add_module(ModuleSpec(
            "Logger",
            inputs={"data": pt},
            outputs={"logged": pt},
            cost=ResourceCost(atp=2, latency_ms=5.0),
            essential=False,
        ))
        diagram.add_module(ModuleSpec(
            "Output",
            inputs={"i1": pt, "i2": pt, "i3": pt},
            cost=ResourceCost(atp=3, latency_ms=2.0),
        ))
        # Dead wire: error prism on a JSON source (will be eliminated)
        diagram.add_module(ModuleSpec(
            "ErrorSink",
            inputs={"err": pt_err},
            cost=ResourceCost(atp=1),
        ))

        diagram.connect("Parser", "parsed", "Validator", "data")
        diagram.connect("Parser", "parsed", "Enricher", "data")
        diagram.connect("Parser", "parsed", "Logger", "data")
        diagram.connect("Validator", "valid", "Output", "i1")
        diagram.connect("Enricher", "enriched", "Output", "i2")
        diagram.connect("Logger", "logged", "Output", "i3")

        # Dead wire: ERROR prism on JSON source
        dead_prism = PrismOptic(accept=frozenset({DataType.ERROR}))
        diagram.connect("Parser", "parsed", "ErrorSink", "err", optic=dead_prism)

        print(f"  Modules: {list(diagram.modules.keys())}")
        print(f"  Wires: {len(diagram.wires)}")

        # ── 2. Static Analysis ───────────────────────────────────────
        print("\n--- 2. Static Analysis ---")

        groups = find_independent_groups(diagram)
        print(f"  Parallel groups: {[sorted(g) for g in groups]}")

        dead = find_dead_wires(diagram)
        print(f"  Dead wires: {len(dead)}")
        for w in dead:
            print(f"    {w.src_module}.{w.src_port} -> {w.dst_module}.{w.dst_port}")

        path, path_cost = critical_path(diagram)
        print(f"  Critical path: {' -> '.join(path)}")
        print(f"  Critical path cost: {path_cost.atp} ATP, {path_cost.latency_ms}ms")

        tc = total_cost(diagram)
        print(f"  Total cost: {tc.atp} ATP, {tc.latency_ms}ms latency, {tc.memory_mb}MB memory")

        suggestions = suggest_optimizations(diagram)
        print(f"\n  Optimization suggestions ({len(suggestions)}):")
        for s in suggestions:
            print(f"    [{s.kind}] {s.description}")

        # ── 3. Apply Optimizer Passes ─────────────────────────────────
        print("\n--- 3. Optimizer Passes ---")

        optimized = optimize(diagram)
        print(f"  Passes applied: {optimized.passes_applied}")
        print(f"  Eliminated wires: {len(optimized.eliminated_wires)}")
        print(f"  Parallel groups: {[sorted(g) for g in optimized.parallel_groups]}")
        print(f"  Execution schedule: {optimized.schedule}")

        # ── 4. Naive Execution (DiagramExecutor) ──────────────────────
        print("\n--- 4. Naive Execution (DiagramExecutor) ---")

        def make_handler(output_port):
            def handler(inputs):
                first = next(iter(inputs.values()))
                return {output_port: first.value}
            return handler

        naive = DiagramExecutor(diagram)
        naive.register_module("Parser", make_handler("parsed"))
        naive.register_module("Validator", make_handler("valid"))
        naive.register_module("Enricher", make_handler("enriched"))
        naive.register_module("Logger", make_handler("logged"))

        naive_report = naive.execute(
            external_inputs={
                "Parser": {"raw": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, {"msg": "hello"})},
            },
            enforce_static_checks=False,
        )
        print(f"  Execution order: {naive_report.execution_order}")

        # ── 5. Resource-Aware Execution (Normal state) ────────────────
        print("\n--- 5. Resource-Aware Execution (NORMAL state) ---")

        store = ATP_Store(budget=200, silent=True)
        aware = ResourceAwareExecutor(optimized, store, max_workers=2)
        aware.register_module("Parser", make_handler("parsed"))
        aware.register_module("Validator", make_handler("valid"))
        aware.register_module("Enricher", make_handler("enriched"))
        aware.register_module("Logger", make_handler("logged"))

        aware_report = aware.execute(
            external_inputs={
                "Parser": {"raw": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, {"msg": "hello"})},
            },
            enforce_static_checks=False,
        )
        print(f"  Execution order: {aware_report.execution_order}")
        print(f"  Total cost: {aware_report.total_cost.atp} ATP")
        print(f"  Skipped modules: {aware_report.skipped_modules}")
        print(f"  Remaining ATP: {store.get_balance()}")

        # ── 6. Resource-Aware Execution (STARVING state) ──────────────
        print("\n--- 6. Resource-Aware Execution (STARVING state) ---")

        store2 = ATP_Store(budget=100, silent=True)
        store2.consume(92, "drain_to_starving")

        aware2 = ResourceAwareExecutor(optimized, store2, max_workers=2)
        aware2.register_module("Parser", make_handler("parsed"))
        aware2.register_module("Validator", make_handler("valid"))
        aware2.register_module("Enricher", make_handler("enriched"))
        aware2.register_module("Logger", make_handler("logged"))

        aware_report2 = aware2.execute(
            external_inputs={
                "Parser": {"raw": TypedValue(DataType.JSON, IntegrityLabel.VALIDATED, {"msg": "hello"})},
            },
            enforce_static_checks=False,
        )
        print(f"  Metabolic state: {store2.get_state().value}")
        print(f"  Execution order: {aware_report2.execution_order}")
        print(f"  Skipped modules: {aware_report2.skipped_modules}")
        print(f"  (Non-essential modules skipped under metabolic stress)")

        # ── 7. BudgetOptic ────────────────────────────────────────────
        print("\n--- 7. BudgetOptic ---")

        budget = BudgetOptic(max_cost=10)
        print(f"  Initial: {budget.name}")
        print(f"  can_transmit: {budget.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)}")
        budget.add_cost(8)
        print(f"  After 8 cost: {budget.name}, remaining={budget.remaining}")
        budget.add_cost(5)
        print(f"  After 13 cost: {budget.name}, remaining={budget.remaining}")
        print(f"  can_transmit: {budget.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)}")

        print("\n" + "=" * 60)
        print("Diagram optimization demo complete.")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
