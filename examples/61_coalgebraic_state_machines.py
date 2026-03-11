"""
Example 61: Coalgebraic State Machines — Composable Observation & Evolution
============================================================================

Demonstrates the coalgebraic interface for state machines, implementing
Paper §3.5.

Biological Analogy:
- Cell state = internal biochemistry (never directly accessible)
- Readout = surface markers / secreted proteins (observable output)
- Update = signal transduction (state evolution on stimulus)
- Bisimulation = two cells that respond identically to all stimuli

Key points:
1. FunctionalCoalgebra wraps two plain functions into a coalgebra
2. StateMachine tracks current state and records transitions
3. ParallelCoalgebra composes two machines running on the same input
4. SequentialCoalgebra pipes one machine's output into another's input
5. check_bisimulation tests observational equivalence

References:
- Article Section 3.5: Epigenetics and State - The Coalgebra
"""

from operon_ai.core.coalgebra import (
    FunctionalCoalgebra,
    ParallelCoalgebra,
    SequentialCoalgebra,
    StateMachine,
    check_bisimulation,
    counter_coalgebra,
)


def main():
    try:
        print("=" * 60)
        print("Coalgebraic State Machines")
        print("=" * 60)

        # ── 1. Counter coalgebra ────────────────────────────────────
        print("\n--- 1. Counter Coalgebra ---")
        counter = counter_coalgebra()
        print(f"readout(0)  = {counter.readout(0)}")
        print(f"update(0,5) = {counter.update(0, 5)}")
        print(f"readout(5)  = {counter.readout(5)}")

        # ── 2. StateMachine with trace ──────────────────────────────
        print("\n--- 2. StateMachine with Trace ---")
        sm = StateMachine(state=0, coalgebra=counter)
        outputs = sm.run([1, 2, 3, 4, 5])
        print(f"Inputs:  [1, 2, 3, 4, 5]")
        print(f"Outputs: {outputs}")
        print(f"Final state: {sm.state}")
        print(f"Trace length: {len(sm.trace)}")
        for rec in sm.trace:
            print(f"  step {rec.step}: {rec.state_before} --({rec.input})--> {rec.state_after}  [out={rec.output}]")

        # ── 3. Parallel composition ─────────────────────────────────
        print("\n--- 3. Parallel Composition ---")
        adder = counter_coalgebra()
        multiplier = FunctionalCoalgebra(
            readout_fn=lambda s: s,
            update_fn=lambda s, i: s * i,
        )
        parallel = ParallelCoalgebra(first=adder, second=multiplier)
        psm = StateMachine(state=(0, 1), coalgebra=parallel)
        for inp in [2, 3, 4]:
            out = psm.step(inp)
            print(f"  input={inp}  output={out}  state={psm.state}")

        # ── 4. Sequential composition ───────────────────────────────
        print("\n--- 4. Sequential Composition ---")
        seq = SequentialCoalgebra(first=adder, second=multiplier)
        ssm = StateMachine(state=(0, 1), coalgebra=seq)
        for inp in [1, 2, 3]:
            out = ssm.step(inp)
            print(f"  input={inp}  output={out}  state={ssm.state}")

        # ── 5. Bisimulation check ───────────────────────────────────
        print("\n--- 5. Bisimulation Check ---")
        a = StateMachine(state=0, coalgebra=counter_coalgebra())
        b = StateMachine(state=0, coalgebra=counter_coalgebra())
        result = check_bisimulation(a, b, [1, 2, 3, 4, 5])
        print(f"Same coalgebra, same initial state:")
        print(f"  equivalent={result.equivalent}, explored={result.states_explored}")
        print(f"  {result.message}")

        c = StateMachine(state=0, coalgebra=counter_coalgebra())
        d = StateMachine(state=10, coalgebra=counter_coalgebra())
        result2 = check_bisimulation(c, d, [1, 2, 3])
        print(f"\nSame coalgebra, different initial state:")
        print(f"  equivalent={result2.equivalent}, witness={result2.witness}")
        print(f"  {result2.message}")

        # ── 6. Conceptual mapping to HistoneStore ───────────────────
        print("\n--- 6. Conceptual Mapping ---")
        print("HistoneStore as coalgebra:")
        print("  State S  = dict of epigenetic markers")
        print("  Input I  = (key, MarkerType, MarkerStrength)")
        print("  Output O = RetrievalResult")
        print("  readout  = retrieve(key)")
        print("  update   = store(key, marker_type, strength)")
        print()
        print("ATP_Store as coalgebra:")
        print("  State S  = energy balances {ATP: x, GTP: y, ...}")
        print("  Input I  = EnergyTransaction(type, amount)")
        print("  Output O = MetabolicReport")
        print("  readout  = report()")
        print("  update   = consume() / deposit()")

        print("\n" + "=" * 60)
        print("Coalgebraic state machines demo complete.")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
