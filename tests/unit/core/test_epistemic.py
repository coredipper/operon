"""Tests for epistemic analyzer — Kripke-style knowledge operators from wiring topology."""

import pytest

from operon_ai.core.epistemic import (
    TopologyClass,
    ObservationProfile,
    EpistemicPartition,
    TopologyClassification,
    ErrorAmplificationBound,
    SequentialPenalty,
    ParallelSpeedup,
    ToolDensityAnalysis,
    TopologyRecommendation,
    EpistemicAnalysis,
    observation_profiles,
    epistemic_partition,
    classify_topology,
    error_amplification_bound,
    sequential_penalty,
    parallel_speedup,
    tool_density,
    recommend_topology,
    analyze,
)
from operon_ai.core.optics import PrismOptic
from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram


def _pt(dt=DataType.JSON, il=IntegrityLabel.VALIDATED):
    return PortType(dt, il)


# ── Reusable diagram builders ─────────────────────────────────────


def _linear_chain():
    """A -> B -> C (sequential)."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}, cost=ResourceCost(atp=10)))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=20)))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}, cost=ResourceCost(atp=5)))
    d.connect("A", "out", "B", "in")
    d.connect("B", "out", "C", "in")
    return d


def _diamond():
    """A -> {B, C} -> D (hub-spoke with parallelism)."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}, cost=ResourceCost(atp=10)))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=20)))
    d.add_module(ModuleSpec("C", inputs={"in": _pt()}, outputs={"out": _pt()}, cost=ResourceCost(atp=15)))
    d.add_module(ModuleSpec("D", inputs={"i1": _pt(), "i2": _pt()}, cost=ResourceCost(atp=5)))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in")
    d.connect("B", "out", "D", "i1")
    d.connect("C", "out", "D", "i2")
    return d


def _independent():
    """A, B, C (no wires)."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", cost=ResourceCost(atp=10)))
    d.add_module(ModuleSpec("B", cost=ResourceCost(atp=10)))
    d.add_module(ModuleSpec("C", cost=ResourceCost(atp=10)))
    return d


def _fan_in():
    """{A, B, C} -> D (centralized)."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt()}, cost=ResourceCost(atp=10)))
    d.add_module(ModuleSpec("B", outputs={"out": _pt()}, cost=ResourceCost(atp=10)))
    d.add_module(ModuleSpec("C", outputs={"out": _pt()}, cost=ResourceCost(atp=10)))
    d.add_module(ModuleSpec("D", inputs={"i1": _pt(), "i2": _pt(), "i3": _pt()}, cost=ResourceCost(atp=10)))
    d.connect("A", "out", "D", "i1")
    d.connect("B", "out", "D", "i2")
    d.connect("C", "out", "D", "i3")
    return d


def _single_module():
    """A (degenerate)."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", cost=ResourceCost(atp=10)))
    return d


# ── observation_profiles ───────────────────────────────────────────


def test_observation_isolated_module():
    """Isolated module has empty sources."""
    d = _independent()
    profiles = observation_profiles(d)
    assert profiles["A"].direct_sources == frozenset()
    assert profiles["A"].transitive_sources == frozenset()
    assert profiles["A"].observation_width == 0


def test_observation_direct_source():
    """A -> B means B directly observes A."""
    d = _linear_chain()
    profiles = observation_profiles(d)
    assert profiles["B"].direct_sources == frozenset({"A"})


def test_observation_transitive():
    """A -> B -> C means C transitively observes {A, B}."""
    d = _linear_chain()
    profiles = observation_profiles(d)
    assert profiles["C"].transitive_sources == frozenset({"A", "B"})
    assert profiles["C"].observation_width == 2


def test_observation_fan_in():
    """{A, B, C} -> D means D directly observes all three."""
    d = _fan_in()
    profiles = observation_profiles(d)
    assert profiles["D"].direct_sources == frozenset({"A", "B", "C"})
    assert profiles["D"].transitive_sources == frozenset({"A", "B", "C"})


def test_observation_optic_filter():
    """Wire with optic sets has_optic_filter = True."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"out": _pt(DataType.JSON)}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt(DataType.ERROR)}))
    prism = PrismOptic(accept=frozenset({DataType.JSON, DataType.ERROR}))
    d.connect("A", "out", "B", "in", optic=prism)
    profiles = observation_profiles(d)
    assert profiles["B"].has_optic_filter is True
    assert profiles["B"].has_denature_filter is False


def test_observation_all_modules_get_profile():
    """Every module in the diagram gets a profile."""
    d = _diamond()
    profiles = observation_profiles(d)
    assert set(profiles.keys()) == {"A", "B", "C", "D"}


# ── epistemic_partition ────────────────────────────────────────────


def test_partition_identical_observers_same_class():
    """B and C in diamond see same source (A) → same equivalence class."""
    d = _diamond()
    part = epistemic_partition(d)
    # B and C both observe exactly {A}
    for eq_class in part.equivalence_classes:
        if "B" in eq_class:
            assert "C" in eq_class
            break
    else:
        raise AssertionError("B not found in any equivalence class")


def test_partition_different_sources_different_classes():
    """Modules with different transitive sources are in different classes."""
    d = _linear_chain()
    part = epistemic_partition(d)
    # A sees {}, B sees {A}, C sees {A,B} → all different
    assert len(part.equivalence_classes) == 3


def test_partition_all_isolated_one_class():
    """All isolated modules have identical (empty) observation → one class."""
    d = _independent()
    part = epistemic_partition(d)
    assert len(part.equivalence_classes) == 1
    assert part.equivalence_classes[0] == frozenset({"A", "B", "C"})


def test_partition_optic_difference_splits_class():
    """Same sources but optic on one wire → different equivalence classes."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", outputs={"o1": _pt(), "o2": _pt()}))
    d.add_module(ModuleSpec("B", inputs={"in": _pt()}))
    d.add_module(ModuleSpec("C", inputs={"in": _pt(DataType.ERROR)}))
    prism = PrismOptic(accept=frozenset({DataType.JSON, DataType.ERROR}))
    d.connect("A", "o1", "B", "in")
    d.connect("A", "o2", "C", "in", optic=prism)
    part = epistemic_partition(d)
    # B and C both see {A} but C has optic → different classes
    for eq_class in part.equivalence_classes:
        if "B" in eq_class:
            assert "C" not in eq_class


# ── classify_topology ──────────────────────────────────────────────


def test_classify_no_wires_independent():
    """No wires → INDEPENDENT."""
    d = _independent()
    cls = classify_topology(d)
    assert cls.topology_class == TopologyClass.INDEPENDENT


def test_classify_linear_sequential():
    """A -> B -> C → SEQUENTIAL."""
    d = _linear_chain()
    cls = classify_topology(d)
    assert cls.topology_class == TopologyClass.SEQUENTIAL
    assert cls.chain_length == 3


def test_classify_fan_in_centralized():
    """{A,B,C} -> D → CENTRALIZED, hub = D."""
    d = _fan_in()
    cls = classify_topology(d)
    assert cls.topology_class == TopologyClass.CENTRALIZED
    assert cls.hub_module == "D"


def test_classify_diamond_hybrid():
    """A -> {B,C} -> D → HYBRID."""
    d = _diamond()
    cls = classify_topology(d)
    assert cls.topology_class == TopologyClass.HYBRID


def test_classify_hub_correctly_identified():
    """Hub is the module with the most transitive sources."""
    d = _fan_in()
    cls = classify_topology(d)
    assert cls.hub_module == "D"
    assert cls.num_sources == 3  # A, B, C have no inputs


# ── error_amplification_bound ──────────────────────────────────────


def test_error_amplification_independent():
    """Independent bound = n (number of non-source workers)."""
    d = _fan_in()  # 3 sources + 1 hub
    bound = error_amplification_bound(d)
    # D is the only non-source module
    assert bound.n_agents == 1
    assert bound.independent_bound == 1


def test_error_amplification_centralized():
    """Centralized bound = n * (1 - detection_rate)."""
    d = _fan_in()
    bound = error_amplification_bound(d, detection_rate=0.75)
    assert bound.centralized_bound == bound.n_agents * (1 - 0.75)
    assert bound.detection_rate == 0.75


def test_error_amplification_custom_rate():
    """Custom detection rate changes the centralized bound."""
    d = _linear_chain()
    bound = error_amplification_bound(d, detection_rate=0.5)
    assert bound.detection_rate == 0.5
    assert bound.amplification_ratio == 1 / (1 - 0.5)


# ── sequential_penalty ─────────────────────────────────────────────


def test_sequential_penalty_linear():
    """Linear chain: overhead = handoffs * comm_cost / chain_length."""
    d = _linear_chain()  # A -> B -> C, chain=3, handoffs=2
    pen = sequential_penalty(d, comm_cost_ratio=0.4)
    assert pen.chain_length == 3
    assert pen.num_handoffs == 2
    assert pen.overhead_ratio == pytest.approx(2 * 0.4 / 3)


def test_sequential_penalty_single_module():
    """Single module: zero overhead."""
    d = _single_module()
    pen = sequential_penalty(d)
    assert pen.num_handoffs == 0
    assert pen.overhead_ratio == 0.0


def test_sequential_penalty_custom_ratio():
    """Custom comm_cost_ratio changes overhead."""
    d = _linear_chain()
    pen = sequential_penalty(d, comm_cost_ratio=0.8)
    assert pen.comm_cost_ratio == 0.8
    assert pen.overhead_ratio == pytest.approx(2 * 0.8 / 3)


# ── parallel_speedup ──────────────────────────────────────────────


def test_parallel_speedup_all_independent():
    """All independent modules: speedup ~ n."""
    d = _independent()  # 3 independent, each cost=10
    sp = parallel_speedup(d)
    assert sp.num_subtasks == 3
    # Total=30, max layer=10 → speedup=3.0
    assert sp.speedup == pytest.approx(3.0)


def test_parallel_speedup_sequential():
    """Sequential chain: speedup = 1 (no parallelism benefit)."""
    d = _linear_chain()  # Each layer has 1 module
    sp = parallel_speedup(d)
    # Total = 35, bottleneck (sum of max per layer) = 35 → speedup = 1.0
    assert sp.speedup == pytest.approx(1.0)


def test_parallel_speedup_diamond():
    """Diamond: partial parallelism."""
    d = _diamond()  # A(10) -> {B(20), C(15)} -> D(5)
    sp = parallel_speedup(d)
    # Layers: [A:10], [B:20, C:15], [D:5] → max layer cost = 20
    # Total = 50, speedup = 50/35 (sum of max per layer)
    assert sp.speedup > 1.0


# ── tool_density ───────────────────────────────────────────────────


def test_tool_density_single_module():
    """Single module with tools: no remote overhead."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "A",
        capabilities={Capability.READ_FS, Capability.NET},
        cost=ResourceCost(atp=10),
    ))
    td = tool_density(d)
    assert td.total_tools == 2
    assert td.num_modules == 1
    assert td.remote_fraction == 0.0
    assert td.planning_cost_ratio == 1.0


def test_tool_density_distributed():
    """Distributed tools across modules: correct remote fraction."""
    d = WiringDiagram()
    d.add_module(ModuleSpec(
        "A",
        outputs={"out": _pt()},
        capabilities={Capability.READ_FS},
        cost=ResourceCost(atp=10),
    ))
    d.add_module(ModuleSpec(
        "B",
        inputs={"in": _pt()},
        capabilities={Capability.NET},
        cost=ResourceCost(atp=10),
    ))
    d.connect("A", "out", "B", "in")
    td = tool_density(d)
    assert td.total_tools == 2
    assert td.num_modules == 2
    assert td.remote_fraction == 0.5  # (2-1)/2
    assert td.planning_cost_ratio == 2.0


def test_tool_density_planning_cost():
    """Planning cost ratio = number of modules with capabilities."""
    d = WiringDiagram()
    d.add_module(ModuleSpec("A", capabilities={Capability.READ_FS}))
    d.add_module(ModuleSpec("B", capabilities={Capability.NET}))
    d.add_module(ModuleSpec("C", capabilities={Capability.EXEC_CODE}))
    td = tool_density(d)
    assert td.planning_cost_ratio == 3.0


# ── recommend_topology ─────────────────────────────────────────────


def test_recommend_independent_subtasks_parallel():
    """Independent subtasks with few tools → INDEPENDENT (parallel)."""
    rec = recommend_topology(
        num_subtasks=5, subtasks_independent=True, num_tools=3,
    )
    assert rec.recommended == TopologyClass.INDEPENDENT


def test_recommend_dependent_subtasks_sequential():
    """Dependent subtasks → SEQUENTIAL."""
    rec = recommend_topology(
        num_subtasks=5, subtasks_independent=False, num_tools=3,
    )
    assert rec.recommended in (TopologyClass.SEQUENTIAL, TopologyClass.CENTRALIZED)


def test_recommend_many_tools_centralized():
    """Many tools with low error tolerance → CENTRALIZED."""
    rec = recommend_topology(
        num_subtasks=3, subtasks_independent=False, num_tools=20, error_tolerance=0.01,
    )
    assert rec.recommended == TopologyClass.CENTRALIZED


# ── analyze (integration) ─────────────────────────────────────────


def test_analyze_returns_complete_result():
    """analyze() returns all fields populated."""
    d = _diamond()
    result = analyze(d)
    assert isinstance(result, EpistemicAnalysis)
    assert set(result.profiles.keys()) == {"A", "B", "C", "D"}
    assert isinstance(result.partition, EpistemicPartition)
    assert isinstance(result.classification, TopologyClassification)
    assert isinstance(result.error_bound, ErrorAmplificationBound)
    assert isinstance(result.sequential, SequentialPenalty)
    assert isinstance(result.speedup, ParallelSpeedup)
    assert isinstance(result.density, ToolDensityAnalysis)


def test_analyze_diamond_expected_values():
    """Diamond topology produces expected classification and bounds."""
    d = _diamond()
    result = analyze(d)
    assert result.classification.topology_class == TopologyClass.HYBRID
    assert result.speedup.speedup > 1.0
    assert result.sequential.chain_length == 3
