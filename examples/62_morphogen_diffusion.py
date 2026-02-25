"""
Example 62: Morphogen Diffusion — Spatially Varying Gradients
==============================================================

Demonstrates graph-based morphogen diffusion, implementing Paper §6.4.

Biological Analogy:
In embryonic development, morphogens (e.g., Bicoid in Drosophila) are
secreted from localized sources and diffuse through tissue.  Cells near
the source see high concentration; cells far away see low concentration.
This gradient drives spatially patterned gene expression without any
central controller.

Since Operon agents lack physical coordinates, we use graph adjacency
from wiring topology as the spatial model.

Key points:
1. DiffusionField manages a graph of nodes and edges
2. MorphogenSource emits at a specific node
3. Concentrations diffuse along edges (emit → diffuse → decay → clamp)
4. get_local_gradient() bridges to the existing MorphogenGradient API
5. Tissue integration provides per-cell gradients

References:
- Article Section 6.4: Morphogen Diffusion
"""

from operon_ai.coordination.diffusion import (
    DiffusionField,
    DiffusionParams,
    MorphogenSource,
)
from operon_ai.coordination.morphogen import MorphogenType


def main():
    try:
        print("=" * 60)
        print("Morphogen Diffusion — Spatially Varying Gradients")
        print("=" * 60)

        # ── 1. Linear chain with source ─────────────────────────────
        print("\n--- 1. Linear Chain (A - B - C - D - E) ---")
        field = DiffusionField()
        nodes = ["A", "B", "C", "D", "E"]
        for n in nodes:
            field.add_node(n)
        for i in range(len(nodes) - 1):
            field.add_edge(nodes[i], nodes[i + 1])

        field.add_source(MorphogenSource("A", MorphogenType.COMPLEXITY, 0.5))
        print("Source at A, emission_rate=0.5")
        print(f"Before diffusion: {field.snapshot()}")

        # ── 2. Gradient formation over steps ────────────────────────
        print("\n--- 2. Gradient Formation ---")
        for step in [5, 10, 20, 50]:
            f2 = DiffusionField()
            for n in nodes:
                f2.add_node(n)
            for i in range(len(nodes) - 1):
                f2.add_edge(nodes[i], nodes[i + 1])
            f2.add_source(MorphogenSource("A", MorphogenType.COMPLEXITY, 0.5))
            f2.run(step)
            concs = [f2.get_concentration(n, MorphogenType.COMPLEXITY) for n in nodes]
            bar = "  ".join(f"{n}={c:.3f}" for n, c in zip(nodes, concs))
            print(f"  step {step:3d}: {bar}")

        # ── 3. Star topology ───────────────────────────────────────
        print("\n--- 3. Star Topology (center + 4 arms) ---")
        star = DiffusionField()
        star.add_node("center")
        for arm in ["N", "S", "E", "W"]:
            star.add_node(arm)
            star.add_edge("center", arm)
        star.add_source(MorphogenSource("center", MorphogenType.URGENCY, 0.3))
        star.run(20)
        print(f"  center: {star.get_concentration('center', MorphogenType.URGENCY):.3f}")
        for arm in ["N", "S", "E", "W"]:
            print(f"  {arm}: {star.get_concentration(arm, MorphogenType.URGENCY):.3f}")

        # ── 4. Competing sources ───────────────────────────────────
        print("\n--- 4. Competing Sources ---")
        compete = DiffusionField(params=DiffusionParams(diffusion_rate=0.15, decay_rate=0.05))
        for n in ["Left", "Mid", "Right"]:
            compete.add_node(n)
        compete.add_edge("Left", "Mid")
        compete.add_edge("Mid", "Right")
        compete.add_source(MorphogenSource("Left", MorphogenType.COMPLEXITY, 0.4))
        compete.add_source(MorphogenSource("Right", MorphogenType.CONFIDENCE, 0.4))
        compete.run(30)
        snap = compete.snapshot()
        print(f"  Left:  {snap.get('Left', {})}")
        print(f"  Mid:   {snap.get('Mid', {})}")
        print(f"  Right: {snap.get('Right', {})}")
        print("  (Complexity high at Left, Confidence high at Right, both present at Mid)")

        # ── 5. Local gradients (MorphogenGradient bridge) ──────────
        print("\n--- 5. Local Gradient API ---")
        field.run(50)
        for n in nodes:
            g = field.get_local_gradient(n)
            level = g.get_level(MorphogenType.COMPLEXITY)
            val = g.get(MorphogenType.COMPLEXITY)
            print(f"  {n}: complexity={val:.3f} ({level})")

        # ── 6. from_adjacency convenience ──────────────────────────
        print("\n--- 6. from_adjacency ---")
        ring = DiffusionField.from_adjacency({
            "X": ["Y", "Z"],
            "Y": ["X", "Z"],
            "Z": ["X", "Y"],
        })
        ring.add_source(MorphogenSource("X", MorphogenType.ERROR_RATE, 0.3))
        ring.run(10)
        print(f"  Ring: {ring.snapshot()}")

        print("\n" + "=" * 60)
        print("Morphogen diffusion demo complete.")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
