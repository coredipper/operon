"""Tests for the categorical compiler formalization.

Verifies functor laws, certificate preservation (Prop 5.1),
and architecture extraction for all four compiler targets.
"""

import pytest

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.categorical import (
    Architecture,
    CompilationResult,
    CompilerFunctor,
    PreservationResult,
    extract_architecture,
    extract_compiled_architecture,
    swarms_functor,
    deerflow_functor,
    ralph_functor,
    scion_functor,
    langgraph_functor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_organism(budget: int = 1000, n_stages: int = 3):
    """Build a test organism with N stages."""
    fast = Nucleus(provider=MockProvider(responses={"classify": "ROUTE: a"}))
    deep = Nucleus(provider=MockProvider(responses={"a": "done"}))

    stage_defs = [
        ("intake", "Normalizer", "fixed"),
        ("router", "Classifier", "fixed"),
        ("executor", "Analyst", "fuzzy"),
        ("reviewer", "Reviewer", "fuzzy"),
        ("reporter", "Reporter", "fixed"),
    ]

    stages = [
        SkillStage(name=name, role=role, instructions=f"Do {name}.", mode=mode)
        for name, role, mode in stage_defs[:n_stages]
    ]

    return skill_organism(
        stages=stages,
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=budget, silent=True),
    )


# ---------------------------------------------------------------------------
# TestArchitecture
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_extract_from_organism(self):
        org = _make_organism()
        arch = extract_architecture(org)

        assert arch.stage_count == 3
        assert arch.stage_names == ("intake", "router", "executor")
        assert arch.is_sequential
        assert len(arch.edges) == 2
        assert len(arch.certificates) >= 1  # ATP certificate

    def test_certificate_theorems(self):
        org = _make_organism()
        arch = extract_architecture(org)

        assert "priority_gating" in arch.certificate_theorems

    def test_zero_budget_certificates(self):
        org = _make_organism(budget=0)
        arch = extract_architecture(org)

        # Certificate exists but won't verify
        assert "priority_gating" in arch.certificate_theorems

    def test_single_stage(self):
        org = _make_organism(n_stages=1)
        arch = extract_architecture(org)

        assert arch.stage_count == 1
        assert arch.edges == ()
        assert arch.is_sequential

    def test_interface_captures_modes(self):
        org = _make_organism()
        arch = extract_architecture(org)

        modes = dict(arch.interface)
        assert modes["intake"] == "fixed"
        assert modes["executor"] == "fuzzy"

    def test_architecture_is_frozen(self):
        org = _make_organism()
        arch = extract_architecture(org)

        with pytest.raises(AttributeError):
            arch.stage_names = ("tampered",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCompilerFunctor — functor laws
# ---------------------------------------------------------------------------

class TestCompilerFunctor:
    """Test functor laws for all four compiler targets."""

    @pytest.mark.parametrize("functor", [
        swarms_functor, deerflow_functor, ralph_functor, scion_functor,
    ], ids=["swarms", "deerflow", "ralph", "scion"])
    def test_compilation_returns_result(self, functor):
        org = _make_organism()
        result = functor.compile(org)

        assert isinstance(result, CompilationResult)
        assert isinstance(result.source_architecture, Architecture)
        assert isinstance(result.target_architecture, Architecture)
        assert isinstance(result.preservation, PreservationResult)

    @pytest.mark.parametrize("functor", [
        swarms_functor, deerflow_functor, ralph_functor, scion_functor,
    ], ids=["swarms", "deerflow", "ralph", "scion"])
    def test_certificate_preservation_prop_5_1(self, functor):
        """Prop 5.1: certificates in source appear in target and verify."""
        org = _make_organism()
        result = functor.compile(org)

        p = result.preservation
        assert p.certificate_preserved, (
            f"{functor.name}: source theorems {p.source.certificate_theorems} "
            f"not preserved in target {p.target.certificate_theorems}"
        )

    # Compilers that add/rename agents (ralph=hats, deerflow=skills,
    # scion=watcher) can't have exact graph/interface parity.
    # Swarms is the only 1:1 compiler.
    _STRICT_COMPILERS = [swarms_functor]

    @pytest.mark.parametrize("functor", _STRICT_COMPILERS, ids=["swarms"])
    def test_graph_preservation(self, functor):
        """Stage count and structure are preserved for 1:1 compilers."""
        org = _make_organism()
        result = functor.compile(org)
        assert result.preservation.graph_preserved

    @pytest.mark.parametrize("functor", _STRICT_COMPILERS, ids=["swarms"])
    def test_interface_preservation(self, functor):
        """Source stage names appear in target for 1:1 compilers."""
        org = _make_organism()
        result = functor.compile(org)
        assert result.preservation.interface_preserved

    @pytest.mark.parametrize("functor", _STRICT_COMPILERS, ids=["swarms"])
    def test_all_preserved(self, functor):
        """Full preservation for 1:1 compilers."""
        org = _make_organism()
        result = functor.compile(org)
        assert result.preservation.all_preserved

    def test_failing_certificate_still_preserved(self):
        """Budget=0 certificate exists in target but verify() returns False."""
        org = _make_organism(budget=0)
        result = swarms_functor.compile(org)

        # Certificate is structurally preserved (it's in the output)
        assert "priority_gating" in result.target_architecture.certificate_theorems

        # But it doesn't hold
        assert not all(
            v.holds for v in result.preservation.certificate_verifications
        )


# ---------------------------------------------------------------------------
# TestExtractCompiledArchitecture
# ---------------------------------------------------------------------------

class TestExtractCompiledArchitecture:
    def test_round_trip(self):
        """Architecture extracted from compiled output has correct structure."""
        org = _make_organism()
        compiled = swarms_functor.compile(org).compiled
        target = extract_compiled_architecture(compiled)

        assert target.stage_count == 3
        assert len(target.certificates) >= 1

    def test_empty_compiled(self):
        target = extract_compiled_architecture({})
        assert target.stage_count == 0
        assert target.certificates == ()

    def test_deerflow_hub_and_spoke(self):
        """DeerFlow: lead → each sub_agent (hub-and-spoke)."""
        compiled = {
            "assistant_id": "lead",
            "sub_agents": [{"name": "worker1"}, {"name": "worker2"}],
            "certificates": [],
        }
        target = extract_compiled_architecture(compiled)
        assert set(target.stage_names) == {"lead", "worker1", "worker2"}
        assert ("lead", "worker1") in target.edges
        assert ("lead", "worker2") in target.edges
        # No worker→worker edge (hub-and-spoke, not chain)
        assert ("worker1", "worker2") not in target.edges

    @pytest.mark.parametrize("edge_key", ["edges", "messaging", "events"])
    def test_explicit_empty_edges_no_fallback(self, edge_key):
        """Explicit empty edge keys should not trigger fallback synthesis."""
        compiled = {
            "agents": [{"name": "a"}, {"name": "b"}],
            edge_key: [],  # Intentionally empty
            "certificates": [],
        }
        target = extract_compiled_architecture(compiled)
        assert target.edges == ()  # No synthesized edges

    def test_ralph_events_extracted(self):
        """Ralph: edges from events field."""
        compiled = {
            "hats": [{"name": "h1"}, {"name": "h2"}],
            "events": [{"from": "h1", "event": "complete", "to": "h2"}],
            "certificates": [],
        }
        target = extract_compiled_architecture(compiled)
        assert ("h1", "h2") in target.edges


# ---------------------------------------------------------------------------
# TestRoundTripPreservation
# ---------------------------------------------------------------------------


class TestRoundTripPreservation:
    """Compile → decompile → analyze → verify certificates preserved.

    Tests Prop 5.1: structural properties are functorially stable under
    architectural refinement.
    """

    def test_deerflow_roundtrip_certificates_verified(self):
        """DeerFlow: compile → decompile → certificates still verify."""
        from operon_ai.convergence.deerflow_compiler import (
            organism_to_deerflow,
            deerflow_to_topology,
        )
        from operon_ai.convergence.swarms_adapter import analyze_external_topology
        from operon_ai.core.certificate import certificate_from_dict

        org = _make_organism()
        compiled = organism_to_deerflow(org)
        topology = deerflow_to_topology(compiled)

        # Analyze decompiled topology
        result = analyze_external_topology(topology)
        assert result.risk_score < 1.0

        # Verify certificates survive round-trip
        certs = topology.metadata.get("certificates", [])
        for cd in certs:
            restored = certificate_from_dict(cd)
            verification = restored.verify()
            assert verification.holds, (
                f"Certificate {restored.theorem} failed after DeerFlow round-trip"
            )

    def test_swarms_roundtrip_certificates_verified(self):
        """Swarms: compile → decompile → certificates still verify."""
        from operon_ai.convergence.swarms_compiler import (
            organism_to_swarms,
            swarms_to_topology,
        )
        from operon_ai.convergence.swarms_adapter import analyze_external_topology
        from operon_ai.core.certificate import certificate_from_dict

        org = _make_organism()
        compiled = organism_to_swarms(org)
        topology = swarms_to_topology(compiled)

        result = analyze_external_topology(topology)
        assert result.risk_score < 1.0

        certs = topology.metadata.get("certificates", [])
        for cd in certs:
            restored = certificate_from_dict(cd)
            verification = restored.verify()
            assert verification.holds, (
                f"Certificate {restored.theorem} failed after Swarms round-trip"
            )

    def test_swarms_roundtrip_graph_preserved(self):
        """Swarms 1:1 compiler preserves exact graph topology."""
        from operon_ai.convergence.swarms_compiler import (
            organism_to_swarms,
            swarms_to_topology,
        )

        org = _make_organism()
        source_arch = extract_architecture(org)
        compiled = organism_to_swarms(org)
        topology = swarms_to_topology(compiled)

        # Swarms should preserve exact stage names
        decompiled_agents = {a["name"] for a in topology.agents}
        source_stages = set(source_arch.stage_names)
        assert source_stages == decompiled_agents, (
            f"Stage names diverged: source={source_stages}, decompiled={decompiled_agents}"
        )

        # Swarms should preserve exact edges
        source_edges = set(source_arch.edges)
        decompiled_edges = set(topology.edges)
        assert source_edges == decompiled_edges, (
            f"Edges diverged: source={source_edges}, decompiled={decompiled_edges}"
        )

    def test_deerflow_roundtrip_stage_names_preserved(self):
        """DeerFlow: all source stage names appear in decompiled topology."""
        from operon_ai.convergence.deerflow_compiler import (
            organism_to_deerflow,
            deerflow_to_topology,
        )

        org = _make_organism()
        source_arch = extract_architecture(org)
        compiled = organism_to_deerflow(org)
        topology = deerflow_to_topology(compiled)

        decompiled_agents = {a["name"] for a in topology.agents}
        source_stages = set(source_arch.stage_names)

        # All source stages should be present (DeerFlow may reshape topology
        # but should not lose stages)
        assert source_stages.issubset(decompiled_agents), (
            f"Lost stages: {source_stages - decompiled_agents}"
        )

    def test_prop_5_1_certificate_set_embedding(self):
        """Prop 5.1 core: source certificate theorems ⊆ target theorems."""
        org = _make_organism()
        source_certs = org.collect_certificates()
        source_theorems = {c.theorem for c in source_certs}

        for functor in [swarms_functor, deerflow_functor]:
            result = functor.compile(org)
            assert result.preservation.certificate_preserved, (
                f"{functor.name}: certificates not preserved"
            )
            # Target theorems should contain all source theorems
            target_theorems = {
                c["theorem"] for c in result.target_architecture.certificates
            }
            assert source_theorems.issubset(target_theorems), (
                f"{functor.name}: source theorems {source_theorems} "
                f"not subset of target {target_theorems}"
            )


# ---------------------------------------------------------------------------
# TestLangGraphFunctor
# ---------------------------------------------------------------------------


class TestLangGraphFunctor:
    """Tests for the LangGraph functor modeling the real single-node graph."""

    def test_certificate_preserved(self):
        """Certificates survive the LangGraph functor."""
        org = _make_organism()
        result = langgraph_functor.compile(org)
        assert result.preservation.certificate_preserved

    def test_certificates_verify(self):
        """All certificates verify after LangGraph compilation."""
        org = _make_organism()
        result = langgraph_functor.compile(org)
        for v in result.preservation.certificate_verifications:
            assert v.holds, f"Certificate {v.certificate.theorem} failed"

    def test_real_graph_shape(self):
        """Target models the real LangGraph graph: one node per stage."""
        org = _make_organism()
        source = extract_architecture(org)
        result = langgraph_functor.compile(org)
        target = result.target_architecture
        assert target.stage_names == source.stage_names
        assert target.edges == source.edges

    def test_source_theorems_subset_of_target(self):
        """Source certificate theorems are preserved in target (Prop 5.1)."""
        org = _make_organism()
        source = extract_architecture(org)
        result = langgraph_functor.compile(org)
        assert source.certificate_theorems <= result.target_architecture.certificate_theorems

    def test_rejects_runtime_config(self):
        """LangGraph functor rejects RuntimeConfig."""
        from operon_ai.convergence.types import RuntimeConfig
        org = _make_organism()
        with pytest.raises(ValueError, match="RuntimeConfig is not supported"):
            langgraph_functor.compile(org, config=RuntimeConfig())

    def test_grouped_organism_preserves_certificates(self):
        """Grouped organism: certificates preserved, graph reshaped."""
        fast = Nucleus(provider=MockProvider(responses={"classify": "ROUTE: a"}))
        deep = Nucleus(provider=MockProvider(responses={"a": "done"}))
        org = skill_organism(
            stages=[
                [
                    SkillStage(name="a", role="A", instructions="do A", mode="fixed"),
                    SkillStage(name="b", role="B", instructions="do B", mode="fixed"),
                ],
                SkillStage(name="c", role="C", instructions="do C", mode="fuzzy"),
            ],
            fast_nucleus=fast,
            deep_nucleus=deep,
            budget=ATP_Store(budget=1000, silent=True),
        )
        result = langgraph_functor.compile(org)
        p = result.preservation

        # Certificates always preserved
        assert p.certificate_preserved

        # Graph and interface preserved (group-level verification)
        assert p.graph_preserved
        assert p.interface_preserved
        assert p.all_preserved

        # Target has 2 nodes (1 parallel group + 1 sequential stage)
        assert result.target_architecture.stage_count == 2
