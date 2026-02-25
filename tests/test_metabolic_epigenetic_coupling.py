"""Tests for Metabolic-Epigenetic Coupling (Paper Section 6.1.1, Eq. 15).

Cost-Gated Retrieval: metabolic state gates which marker strengths
are accessible from HistoneStore. Low energy silences expensive context.
"""

import pytest
from operon_ai.state.metabolism import ATP_Store, MetabolicState, MetabolicAccessPolicy
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength


def _make_coupled_stores(budget: int, retrieval_cost: int = 5) -> tuple[ATP_Store, HistoneStore]:
    """Create an ATP_Store and HistoneStore coupled via MetabolicAccessPolicy."""
    atp = ATP_Store(budget=budget, silent=True)
    policy = MetabolicAccessPolicy(retrieval_cost=retrieval_cost)
    histones = HistoneStore(energy_gate=(atp, policy), silent=True)
    return atp, histones


def _seed_markers(histones: HistoneStore) -> None:
    """Add one marker at each strength level."""
    histones.add_marker("weak lesson", strength=MarkerStrength.WEAK, tags=["test"])
    histones.add_marker("moderate lesson", strength=MarkerStrength.MODERATE, tags=["test"])
    histones.add_marker("strong lesson", strength=MarkerStrength.STRONG, tags=["test"])
    histones.add_marker(
        "permanent lesson",
        marker_type=MarkerType.METHYLATION,
        strength=MarkerStrength.PERMANENT,
        tags=["test"],
    )


class TestMetabolicEpigeneticCoupling:
    """Tests for energy-gated epigenetic retrieval."""

    def test_normal_state_all_accessible(self):
        """NORMAL metabolic state grants access to all marker strengths."""
        atp, histones = _make_coupled_stores(budget=100)
        _seed_markers(histones)

        result = histones.retrieve_context(tags=["test"])
        assert len(result.markers) == 4
        assert atp.atp == 95  # 5 ATP consumed for retrieval

    def test_conserving_state_only_strong_plus(self):
        """CONSERVING state silences WEAK and MODERATE markers."""
        atp, histones = _make_coupled_stores(budget=100)
        _seed_markers(histones)

        # Drain to CONSERVING (below 30%)
        atp.consume(75, "drain")
        assert atp.get_state() == MetabolicState.CONSERVING

        result = histones.retrieve_context(tags=["test"])
        strengths = {m.strength for m in result.markers}
        assert MarkerStrength.WEAK not in strengths
        assert MarkerStrength.MODERATE not in strengths
        assert MarkerStrength.STRONG in strengths
        assert MarkerStrength.PERMANENT in strengths
        assert len(result.markers) == 2

    def test_starving_state_only_permanent(self):
        """STARVING state allows only PERMANENT markers."""
        atp, histones = _make_coupled_stores(budget=100)
        _seed_markers(histones)

        # Drain to STARVING (below 10%) — need priority >= 5 to consume while starving
        atp.consume(92, "drain")
        assert atp.get_state() == MetabolicState.STARVING

        result = histones.retrieve_context(tags=["test"])
        assert len(result.markers) == 1
        assert result.markers[0].strength == MarkerStrength.PERMANENT

    def test_dormant_state_nothing_accessible(self):
        """DORMANT state blocks all retrieval."""
        atp, histones = _make_coupled_stores(budget=100)
        _seed_markers(histones)

        atp.enter_dormancy()
        assert atp.get_state() == MetabolicState.DORMANT

        result = histones.retrieve_context(tags=["test"])
        assert len(result.markers) == 0
        assert result.formatted_context == ""

    def test_insufficient_atp_blocks_retrieval(self):
        """Retrieval blocked when ATP < retrieval_cost."""
        atp, histones = _make_coupled_stores(budget=100, retrieval_cost=10)
        _seed_markers(histones)

        # Drain to just below retrieval cost but stay in NORMAL range
        atp.consume(95, "drain")
        assert atp.atp == 5  # Less than retrieval_cost=10

        result = histones.retrieve_context(tags=["test"])
        assert len(result.markers) == 0
        assert atp.atp == 5  # ATP unchanged (retrieval failed)

    def test_no_energy_gate_unchanged_behavior(self):
        """Without energy_gate, HistoneStore behaves exactly as before."""
        histones = HistoneStore(silent=True)
        _seed_markers(histones)

        result = histones.retrieve_context(tags=["test"])
        assert len(result.markers) == 4  # All accessible, no ATP check

    def test_retrieval_consumes_atp(self):
        """Each successful retrieval deducts retrieval_cost ATP."""
        atp, histones = _make_coupled_stores(budget=100, retrieval_cost=10)
        _seed_markers(histones)

        histones.retrieve_context(tags=["test"])
        assert atp.atp == 90

        histones.retrieve_context(tags=["test"])
        assert atp.atp == 80

    def test_feasting_state_all_accessible(self):
        """FEASTING state (excess energy) grants full access."""
        atp, histones = _make_coupled_stores(budget=100)
        _seed_markers(histones)

        # At 100% budget, should be FEASTING or NORMAL
        assert atp.get_state() in (MetabolicState.NORMAL, MetabolicState.FEASTING)

        result = histones.retrieve_context(tags=["test"])
        assert len(result.markers) == 4

    def test_gated_retrievals_tracked(self):
        """Statistics track how many retrievals were gated."""
        atp, histones = _make_coupled_stores(budget=100)
        _seed_markers(histones)

        atp.enter_dormancy()
        histones.retrieve_context(tags=["test"])
        histones.retrieve_context(tags=["test"])

        stats = histones.get_statistics()
        assert stats["gated_retrievals"] == 2

    def test_custom_access_policy(self):
        """Custom MetabolicAccessPolicy overrides default thresholds."""
        atp = ATP_Store(budget=100, silent=True)
        # Custom: NORMAL requires MODERATE+, CONSERVING requires PERMANENT
        policy = MetabolicAccessPolicy(
            state_thresholds={
                MetabolicState.FEASTING: 1,
                MetabolicState.NORMAL: 2,       # MODERATE+
                MetabolicState.CONSERVING: 4,   # PERMANENT only
                MetabolicState.STARVING: None,
                MetabolicState.DORMANT: None,
            },
            retrieval_cost=5,
        )
        histones = HistoneStore(energy_gate=(atp, policy), silent=True)
        _seed_markers(histones)

        # NORMAL state with custom policy → only MODERATE+ (3 markers)
        result = histones.retrieve_context(tags=["test"])
        assert len(result.markers) == 3
        strengths = {m.strength for m in result.markers}
        assert MarkerStrength.WEAK not in strengths

    def test_state_transitions_change_access(self):
        """As ATP depletes through operations, accessible markers narrow."""
        atp, histones = _make_coupled_stores(budget=100, retrieval_cost=5)
        _seed_markers(histones)

        # Phase 1: NORMAL — all 4 markers
        result1 = histones.retrieve_context(tags=["test"])
        count_normal = len(result1.markers)

        # Drain to CONSERVING
        atp.consume(70, "drain")
        assert atp.get_state() == MetabolicState.CONSERVING

        # Phase 2: CONSERVING — only STRONG+ (2 markers)
        result2 = histones.retrieve_context(tags=["test"])
        count_conserving = len(result2.markers)

        assert count_normal > count_conserving
        assert count_conserving == 2
