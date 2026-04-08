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

    def test_explicit_empty_edges_no_fallback(self):
        """Explicit edges=[] should not trigger fallback synthesis."""
        compiled = {
            "agents": [{"name": "a"}, {"name": "b"}],
            "edges": [],  # Intentionally empty
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
