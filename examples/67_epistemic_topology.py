"""
Example 67: Epistemic Topology of Multi-Cellular Agents
========================================================

Derives Kripke-style observation profiles from wiring diagram topology,
then applies four theorems to predict error amplification, coordination
overhead, parallelism bounds, and tool density scaling.

Biological Analogy:
Cells in a tissue don't all see the same signals — each cell's "knowledge"
of the tissue state depends on which receptors it expresses and which
morphogen gradients reach it (observation profiles). Cells with identical
receptor profiles form functional equivalence classes (epistemic partition).
Tissue architecture — epithelial sheets, digestive tracts, nervous system
hubs — determines how errors propagate (Theorem 1), how much overhead
signal transduction adds (Theorem 2), how much parallelism is possible
(Theorem 3), and how enzyme distribution drives coordination cost
(Theorem 4).

Key points:
1. ObservationProfile — per-module epistemic reach (direct + transitive)
2. EpistemicPartition — equivalence classes by shared observation
3. TopologyClassification — INDEPENDENT / SEQUENTIAL / CENTRALIZED / HYBRID
4. Theorem 1: Error amplification bounds (independent vs centralized)
5. Theorem 2: Sequential communication overhead
6. Theorems 3 & 4: Parallel speedup and tool density scaling

References:
- Article Section 6.5.4: Epistemic Topology of Wiring Diagrams
- Theorems 1-4: Error amplification, sequential penalty, parallel
  speedup, tool density scaling
"""

import sys

from operon_ai.core.denature import SummarizeFilter
from operon_ai.core.epistemic import (
    analyze,
    classify_topology,
    epistemic_partition,
    error_amplification_bound,
    observation_profiles,
    parallel_speedup,
    recommend_topology,
    sequential_penalty,
    tool_density,
)
from operon_ai.core.optics import PrismOptic
from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram


def _pt(dt=DataType.JSON, il=IntegrityLabel.VALIDATED):
    """Shorthand for PortType construction."""
    return PortType(dt, il)


# ── Diagram Builders ────────────────────────────────────────────────


def build_independent():
    """Three isolated workers — no wires, fully parallel."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Summarizer",
        outputs={"summary": _pt()},
        cost=ResourceCost(atp=15, latency_ms=30.0),
        capabilities={Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Translator",
        outputs={"translated": _pt()},
        cost=ResourceCost(atp=20, latency_ms=40.0),
        capabilities={Capability.NET},
    ))
    d.add_module(ModuleSpec(
        "Formatter",
        outputs={"formatted": _pt()},
        cost=ResourceCost(atp=10, latency_ms=15.0),
    ))
    return d


def build_sequential():
    """A → B → C → D linear pipeline."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Ingestor",
        outputs={"raw": _pt()},
        cost=ResourceCost(atp=5, latency_ms=10.0),
    ))
    d.add_module(ModuleSpec(
        "Parser",
        inputs={"data": _pt()},
        outputs={"parsed": _pt()},
        cost=ResourceCost(atp=10, latency_ms=20.0),
    ))
    d.add_module(ModuleSpec(
        "Validator",
        inputs={"data": _pt()},
        outputs={"valid": _pt()},
        cost=ResourceCost(atp=15, latency_ms=25.0),
        capabilities={Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Writer",
        inputs={"data": _pt()},
        cost=ResourceCost(atp=8, latency_ms=12.0),
        capabilities={Capability.WRITE_FS},
    ))
    d.connect("Ingestor", "raw", "Parser", "data")
    d.connect("Parser", "parsed", "Validator", "data")
    d.connect("Validator", "valid", "Writer", "data")
    return d


def build_centralized():
    """{W1, W2, W3} → Aggregator fan-in hub."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Worker1",
        outputs={"out": _pt()},
        cost=ResourceCost(atp=12, latency_ms=20.0),
        capabilities={Capability.READ_FS},
    ))
    d.add_module(ModuleSpec(
        "Worker2",
        outputs={"out": _pt()},
        cost=ResourceCost(atp=12, latency_ms=20.0),
        capabilities={Capability.NET},
    ))
    d.add_module(ModuleSpec(
        "Worker3",
        outputs={"out": _pt()},
        cost=ResourceCost(atp=12, latency_ms=20.0),
        capabilities={Capability.EXEC_CODE},
    ))
    d.add_module(ModuleSpec(
        "Aggregator",
        inputs={"i1": _pt(), "i2": _pt(), "i3": _pt()},
        cost=ResourceCost(atp=8, latency_ms=10.0),
    ))
    d.connect("Worker1", "out", "Aggregator", "i1")
    d.connect("Worker2", "out", "Aggregator", "i2")
    d.connect("Worker3", "out", "Aggregator", "i3")
    return d


def build_diamond():
    """Dispatcher → {Analyzer, Enricher} → Synthesizer (hybrid diamond).

    Enricher wire has a SummarizeFilter (denature), Analyzer wire has
    a PrismOptic — demonstrating mixed epistemic filtering.
    """
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "Dispatcher",
        outputs={"o1": _pt(), "o2": _pt()},
        cost=ResourceCost(atp=5, latency_ms=8.0),
    ))
    d.add_module(ModuleSpec(
        "Analyzer",
        inputs={"in": _pt(DataType.JSON, IntegrityLabel.VALIDATED)},
        outputs={"out": _pt()},
        cost=ResourceCost(atp=25, latency_ms=50.0),
        capabilities={Capability.READ_FS, Capability.EXEC_CODE},
    ))
    d.add_module(ModuleSpec(
        "Enricher",
        inputs={"in": _pt()},
        outputs={"out": _pt()},
        cost=ResourceCost(atp=18, latency_ms=35.0),
        capabilities={Capability.NET},
    ))
    d.add_module(ModuleSpec(
        "Synthesizer",
        inputs={"i1": _pt(), "i2": _pt()},
        cost=ResourceCost(atp=10, latency_ms=15.0),
    ))

    # Optic on Analyzer wire — prism routing by data type
    prism = PrismOptic(accept=frozenset({DataType.JSON, DataType.ERROR}))
    d.connect("Dispatcher", "o1", "Analyzer", "in", optic=prism)

    # Denature on Enricher wire — anti-injection summarize filter
    denature = SummarizeFilter(max_length=200, prefix="[enriched]")
    d.connect("Dispatcher", "o2", "Enricher", "in", denature=denature)

    d.connect("Analyzer", "out", "Synthesizer", "i1")
    d.connect("Enricher", "out", "Synthesizer", "i2")
    return d


def main():
    try:
        print("=" * 65)
        print("Epistemic Topology of Multi-Cellular Agents")
        print("=" * 65)

        # ── 1. Build Four Canonical Topologies ─────────────────────────
        print("\n--- 1. Four Canonical Topologies ---")

        independent = build_independent()
        sequential = build_sequential()
        centralized = build_centralized()
        diamond = build_diamond()

        diagrams = {
            "Independent": independent,
            "Sequential": sequential,
            "Centralized": centralized,
            "Diamond (Hybrid)": diamond,
        }

        for name, d in diagrams.items():
            print(f"  {name}: {len(d.modules)} modules, {len(d.wires)} wires")

        # ── 2. Observation Profiles ────────────────────────────────────
        print("\n--- 2. Observation Profiles (Diamond) ---")

        profiles = observation_profiles(diamond)
        for name, prof in sorted(profiles.items()):
            direct = sorted(prof.direct_sources) if prof.direct_sources else "none"
            trans = sorted(prof.transitive_sources) if prof.transitive_sources else "none"
            filters = []
            if prof.has_optic_filter:
                filters.append("optic")
            if prof.has_denature_filter:
                filters.append("denature")
            filter_str = ", ".join(filters) if filters else "none"
            print(f"  {name}:")
            print(f"    Direct sources:     {direct}")
            print(f"    Transitive sources: {trans}")
            print(f"    Observation width:  {prof.observation_width}")
            print(f"    Filters:            {filter_str}")

        # ── 3. Epistemic Partition ─────────────────────────────────────
        print("\n--- 3. Epistemic Partition (Diamond) ---")

        partition = epistemic_partition(diamond)
        for i, eq_class in enumerate(partition.equivalence_classes, 1):
            print(f"  Class {i}: {sorted(eq_class)}")
        print(f"  Total classes: {len(partition.equivalence_classes)}")

        # ── 4. Topology Classification ─────────────────────────────────
        print("\n--- 4. Topology Classification ---")

        print(f"  {'Diagram':<20} {'Class':<14} {'Hub':<12} {'Chain':<7} {'Width':<7} {'Sources'}")
        print(f"  {'-'*19} {'-'*13} {'-'*11} {'-'*6} {'-'*6} {'-'*7}")
        for name, d in diagrams.items():
            cls = classify_topology(d)
            hub = cls.hub_module or "-"
            print(
                f"  {name:<20} {cls.topology_class.value:<14} {hub:<12} "
                f"{cls.chain_length:<7} {cls.parallelism_width:<7} {cls.num_sources}"
            )

        # ── 5. Theorem 1: Error Amplification ─────────────────────────
        print("\n--- 5. Theorem 1: Error Amplification Bounds ---")

        print(f"  {'Diagram':<16} {'d':<6} {'Indep.':<9} {'Central.':<11} {'Ratio'}")
        print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*10} {'-'*8}")
        for rate in [0.5, 0.75, 0.9]:
            for label, d in [("Centralized", centralized), ("Independent", independent)]:
                bound = error_amplification_bound(d, detection_rate=rate)
                print(
                    f"  {label:<16} {rate:<6.2f} {bound.independent_bound:<9} "
                    f"{bound.centralized_bound:<11.2f} {bound.amplification_ratio:.2f}"
                )

        # ── 6. Theorem 2: Sequential Penalty ──────────────────────────
        print("\n--- 6. Theorem 2: Sequential Communication Overhead ---")

        print(f"  {'Comm Cost':<12} {'Chain':<8} {'Handoffs':<10} {'Overhead'}")
        print(f"  {'-'*11} {'-'*7} {'-'*9} {'-'*10}")
        for ratio in [0.1, 0.4, 0.8]:
            pen = sequential_penalty(sequential, comm_cost_ratio=ratio)
            print(
                f"  {ratio:<12.1f} {pen.chain_length:<8} "
                f"{pen.num_handoffs:<10} {pen.overhead_ratio:.4f}"
            )

        # ── 7. Theorems 3 & 4: Speedup + Tool Density ─────────────────
        print("\n--- 7. Theorem 3: Parallel Speedup ---")

        print(f"  {'Diagram':<20} {'Modules':<9} {'Speedup':<10} {'Total ATP':<11} {'Bottleneck ATP'}")
        print(f"  {'-'*19} {'-'*8} {'-'*9} {'-'*10} {'-'*14}")
        for name, d in diagrams.items():
            sp = parallel_speedup(d)
            print(
                f"  {name:<20} {sp.num_subtasks:<9} {sp.speedup:<10.2f} "
                f"{sp.total_cost.atp:<11} {sp.max_layer_cost.atp}"
            )

        print("\n--- 7b. Theorem 4: Tool Density ---")

        td = tool_density(diamond)
        print(f"  Diagram: Diamond (Hybrid)")
        print(f"  Total tools:          {td.total_tools}")
        print(f"  Modules with caps:    {td.num_modules}")
        print(f"  Tools per module:     {td.tools_per_module:.2f}")
        print(f"  Remote fraction:      {td.remote_fraction:.2f}")
        print(f"  Planning cost ratio:  {td.planning_cost_ratio:.1f}")

        # ── 8. Topology Recommendation ─────────────────────────────────
        print("\n--- 8. Topology Recommendation ---")

        scenarios = [
            {"num_subtasks": 8, "subtasks_independent": True, "num_tools": 3,
             "error_tolerance": 0.1},
            {"num_subtasks": 5, "subtasks_independent": False, "num_tools": 3,
             "error_tolerance": 0.1},
            {"num_subtasks": 4, "subtasks_independent": False, "num_tools": 20,
             "error_tolerance": 0.01},
            {"num_subtasks": 10, "subtasks_independent": True, "num_tools": 15,
             "error_tolerance": 0.05},
        ]
        for i, s in enumerate(scenarios, 1):
            rec = recommend_topology(**s)
            print(f"  Scenario {i}: {s}")
            print(f"    → {rec.recommended.value}: {rec.rationale}")

        print("\n" + "=" * 65)
        print("Epistemic topology demo complete.")
        print("=" * 65)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_smoke_test():
    """Smoke test: build diamond, run analyze(), verify types and classifications."""
    from operon_ai.core.epistemic import (
        EpistemicAnalysis,
        TopologyClass,
        TopologyClassification,
    )

    # Full analysis on diamond
    diamond = build_diamond()
    result = analyze(diamond)
    assert isinstance(result, EpistemicAnalysis), "analyze() must return EpistemicAnalysis"
    assert set(result.profiles.keys()) == {"Dispatcher", "Analyzer", "Enricher", "Synthesizer"}
    assert result.classification.topology_class == TopologyClass.HYBRID
    assert result.speedup.speedup > 1.0

    # Classification of all four topologies
    expected = {
        "Independent": TopologyClass.INDEPENDENT,
        "Sequential": TopologyClass.SEQUENTIAL,
        "Centralized": TopologyClass.CENTRALIZED,
        "Diamond": TopologyClass.HYBRID,
    }
    builders = {
        "Independent": build_independent,
        "Sequential": build_sequential,
        "Centralized": build_centralized,
        "Diamond": build_diamond,
    }
    for name, builder in builders.items():
        cls = classify_topology(builder())
        assert isinstance(cls, TopologyClassification)
        assert cls.topology_class == expected[name], (
            f"{name}: expected {expected[name]}, got {cls.topology_class}"
        )

    print("Smoke test passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
