"""Tests for mTOR/AMPK adaptive scaling module."""

import pytest

from operon_ai.state.metabolism import ATP_Store
from operon_ai.state.mtor import AMPKRatio, MTORScaler, ScalingState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(budget: int = 100, atp: int | None = None) -> ATP_Store:
    """Create a silent ATP_Store, optionally pre-drained to *atp*."""
    s = ATP_Store(budget=budget, silent=True)
    if atp is not None:
        s.atp = atp
    return s


def _scaler(budget: int = 100, atp: int | None = None, **kw) -> MTORScaler:
    """Create an MTORScaler backed by a silent ATP_Store."""
    return MTORScaler(atp_store=_store(budget, atp), **kw)


# ---------------------------------------------------------------------------
# 1. AMPKRatio correctly computes ratio from ATP levels
# ---------------------------------------------------------------------------


class TestAMPKRatio:
    def test_full_atp_gives_zero_ratio(self):
        r = AMPKRatio()
        r.update(atp_level=100, max_atp=100)
        assert r.current_ratio == pytest.approx(0.0)

    def test_empty_atp_gives_one_ratio(self):
        r = AMPKRatio()
        r.update(atp_level=0, max_atp=100)
        assert r.current_ratio == pytest.approx(1.0)

    def test_half_atp(self):
        r = AMPKRatio()
        r.update(atp_level=50, max_atp=100)
        assert r.current_ratio == pytest.approx(0.5)

    def test_zero_max_atp_no_division_error(self):
        r = AMPKRatio()
        r.update(atp_level=0, max_atp=0)
        assert r.current_ratio == pytest.approx(1.0)

    # 2. AMPKRatio tracks rate of change
    def test_rate_of_change_positive_on_depletion(self):
        r = AMPKRatio()
        r.update(atp_level=100, max_atp=100)  # ratio 0
        r.update(atp_level=50, max_atp=100)   # ratio 0.5
        assert r.rate_of_change == pytest.approx(0.5)

    def test_rate_of_change_negative_on_regen(self):
        r = AMPKRatio()
        r.update(atp_level=50, max_atp=100)   # ratio 0.5
        r.update(atp_level=100, max_atp=100)  # ratio 0
        assert r.rate_of_change == pytest.approx(-0.5)

    def test_rate_of_change_zero_when_stable(self):
        r = AMPKRatio()
        r.update(atp_level=70, max_atp=100)
        r.update(atp_level=70, max_atp=100)
        assert r.rate_of_change == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. MTORScaler starts in MAINTENANCE
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_default_state_is_maintenance(self):
        scaler = _scaler()
        assert scaler.state == ScalingState.MAINTENANCE

    def test_transitions_start_at_zero(self):
        scaler = _scaler()
        assert scaler._transitions == 0


# ---------------------------------------------------------------------------
# 4-6. State determination based on ATP level
# ---------------------------------------------------------------------------


class TestStateFromATPLevel:
    def test_growth_when_atp_full(self):
        """Ratio ~0 => GROWTH."""
        scaler = _scaler(budget=100, atp=100)
        state = scaler.update()
        # From MAINTENANCE, ratio 0.0 < growth_threshold(0.3) - hysteresis(0.05) = 0.25
        assert state == ScalingState.GROWTH

    def test_maintenance_at_moderate_atp(self):
        """Ratio ~0.5 stays MAINTENANCE."""
        scaler = _scaler(budget=100, atp=50)
        state = scaler.update()
        assert state == ScalingState.MAINTENANCE

    def test_conservation_when_atp_low(self):
        """Ratio 0.8 => CONSERVATION (above conservation_threshold + hysteresis)."""
        scaler = _scaler(budget=100, atp=20)
        state = scaler.update()
        # ratio = 0.8, conservation_threshold + h = 0.75 => CONSERVATION
        assert state == ScalingState.CONSERVATION

    def test_autophagy_when_atp_nearly_empty(self):
        """Ratio 0.95 => AUTOPHAGY."""
        scaler = _scaler(budget=100, atp=5)
        state = scaler.update()
        # ratio = 0.95, >= autophagy_threshold(0.9) => AUTOPHAGY
        assert state == ScalingState.AUTOPHAGY

    def test_autophagy_when_atp_zero(self):
        scaler = _scaler(budget=100, atp=0)
        state = scaler.update()
        assert state == ScalingState.AUTOPHAGY


# ---------------------------------------------------------------------------
# 7. Hysteresis prevents oscillation at boundaries
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_growth_sticky_at_boundary(self):
        """Once in GROWTH, stay there until ratio exceeds growth_threshold + hysteresis."""
        scaler = _scaler(budget=100, atp=100)
        scaler.update()  # -> GROWTH (ratio 0.0)
        assert scaler.state == ScalingState.GROWTH

        # Settle the rate by making a small step first, so rate_of_change is small
        scaler.atp_store.atp = 72  # ratio = 0.28, rate small
        scaler.update()
        assert scaler.state == ScalingState.GROWTH

        # Now ratio at boundary (0.32) — rate_of_change is small (0.04)
        # effective_ratio ≈ 0.32 + 0.1*0.04 = 0.324, below 0.35
        scaler.atp_store.atp = 68  # ratio = 0.32
        scaler.update()
        assert scaler.state == ScalingState.GROWTH  # Sticky — hasn't crossed 0.35

    def test_growth_exits_above_hysteresis(self):
        """GROWTH exits when ratio clearly above growth_threshold + hysteresis."""
        scaler = _scaler(budget=100, atp=100)
        scaler.update()  # -> GROWTH
        assert scaler.state == ScalingState.GROWTH

        scaler.atp_store.atp = 60  # ratio = 0.4, > 0.35
        scaler.update()
        assert scaler.state == ScalingState.MAINTENANCE

    def test_conservation_sticky_at_boundary(self):
        """Once in CONSERVATION, stay there until ratio drops below conservation_threshold - h."""
        scaler = _scaler(budget=100, atp=20)
        scaler.update()  # -> CONSERVATION (ratio 0.8)
        assert scaler.state == ScalingState.CONSERVATION

        # Set ATP so ratio is 0.68 (between 0.65 and 0.7)
        scaler.atp_store.atp = 32  # ratio = 0.68
        scaler.update()
        assert scaler.state == ScalingState.CONSERVATION  # Sticky

    def test_conservation_exits_below_hysteresis(self):
        """CONSERVATION exits when ratio drops below conservation_threshold - h."""
        scaler = _scaler(budget=100, atp=20)
        scaler.update()  # -> CONSERVATION
        assert scaler.state == ScalingState.CONSERVATION

        scaler.atp_store.atp = 40  # ratio = 0.6, < 0.65
        scaler.update()
        assert scaler.state == ScalingState.MAINTENANCE

    def test_autophagy_sticky_at_boundary(self):
        """Once in AUTOPHAGY, stay until ratio < autophagy_threshold - h."""
        scaler = _scaler(budget=100, atp=5)
        scaler.update()  # -> AUTOPHAGY (ratio 0.95)
        assert scaler.state == ScalingState.AUTOPHAGY

        # ratio 0.88, within hysteresis band (0.85-0.9)
        scaler.atp_store.atp = 12  # ratio = 0.88
        scaler.update()
        assert scaler.state == ScalingState.AUTOPHAGY  # Sticky


# ---------------------------------------------------------------------------
# 8. Rate sensitivity — rapid depletion triggers earlier transition
# ---------------------------------------------------------------------------


class TestRateSensitivity:
    def test_rapid_depletion_pushes_to_conservation(self):
        """Rapid ATP drain makes effective ratio higher, triggering CONSERVATION
        even when raw ratio is still in MAINTENANCE zone."""
        # High rate_sensitivity to amplify the effect
        scaler = _scaler(budget=100, atp=100, rate_sensitivity=2.0)
        scaler.update()  # ratio 0.0 -> GROWTH

        # Jump from 100 to 40 in one step: raw ratio 0.6 (MAINTENANCE zone)
        # rate_of_change = 0.6 - 0.0 = 0.6
        # effective = 0.6 + 2.0 * 0.6 = 1.8, clamped to 1.0 -> AUTOPHAGY
        scaler.atp_store.atp = 40
        state = scaler.update()
        assert state == ScalingState.AUTOPHAGY

    def test_rapid_regen_keeps_growth(self):
        """Rapid ATP regen lowers effective ratio, keeping GROWTH active."""
        scaler = _scaler(budget=100, atp=70, rate_sensitivity=1.0)
        scaler.update()  # ratio 0.3 -> stays MAINTENANCE (at boundary)

        # Regenerate: ATP goes from 70 to 95
        # ratio = 0.05, rate_of_change = 0.05 - 0.3 = -0.25
        # effective = 0.05 + 1.0 * (-0.25) = -0.20, clamped to 0.0 -> GROWTH
        scaler.atp_store.atp = 95
        state = scaler.update()
        assert state == ScalingState.GROWTH


# ---------------------------------------------------------------------------
# 9. recommended_workers returns correct values per state
# ---------------------------------------------------------------------------


class TestRecommendedWorkers:
    def test_growth_max_workers(self):
        scaler = _scaler(budget=100, atp=100)
        scaler.update()  # GROWTH
        # fraction 1.0: 1 + int(1.0 * 7) = 8
        assert scaler.recommended_workers() == 8

    def test_maintenance_moderate_workers(self):
        scaler = _scaler(budget=100, atp=50)
        scaler.update()  # MAINTENANCE
        # fraction 0.6: 1 + int(0.6 * 7) = 1 + 4 = 5
        assert scaler.recommended_workers() == 5

    def test_conservation_few_workers(self):
        scaler = _scaler(budget=100, atp=20)
        scaler.update()  # CONSERVATION
        # fraction 0.3: 1 + int(0.3 * 7) = 1 + 2 = 3
        assert scaler.recommended_workers() == 3

    def test_autophagy_min_workers(self):
        scaler = _scaler(budget=100, atp=0)
        scaler.update()  # AUTOPHAGY
        # fraction 0.0: 1 + int(0.0 * 7) = 1
        assert scaler.recommended_workers() == 1

    def test_custom_worker_range(self):
        scaler = _scaler(budget=100, atp=100, min_workers=2, max_workers=16)
        scaler.update()  # GROWTH
        # fraction 1.0: 2 + int(1.0 * 14) = 16
        assert scaler.recommended_workers() == 16

    def test_workers_never_below_min(self):
        scaler = _scaler(budget=100, atp=0, min_workers=3, max_workers=10)
        scaler.update()  # AUTOPHAGY
        assert scaler.recommended_workers() == 3


# ---------------------------------------------------------------------------
# 10. should_enable_feature gates correctly per state
# ---------------------------------------------------------------------------


class TestFeatureGating:
    def test_growth_allows_expensive(self):
        scaler = _scaler(budget=100, atp=100)
        scaler.update()
        assert scaler.should_enable_feature(feature_cost=100) is True

    def test_maintenance_allows_expensive(self):
        scaler = _scaler(budget=100, atp=50)
        scaler.update()
        assert scaler.should_enable_feature(feature_cost=100) is True

    def test_conservation_blocks_expensive(self):
        scaler = _scaler(budget=100, atp=20)
        scaler.update()
        assert scaler.should_enable_feature(feature_cost=50) is False

    def test_conservation_allows_cheap(self):
        scaler = _scaler(budget=100, atp=20)
        scaler.update()
        assert scaler.should_enable_feature(feature_cost=10) is True

    def test_autophagy_blocks_everything(self):
        scaler = _scaler(budget=100, atp=0)
        scaler.update()
        assert scaler.should_enable_feature(feature_cost=1) is False


# ---------------------------------------------------------------------------
# 11. Transitions counter increments on state change
# ---------------------------------------------------------------------------


class TestTransitionCounter:
    def test_no_transition_same_state(self):
        scaler = _scaler(budget=100, atp=50)
        scaler.update()  # MAINTENANCE -> MAINTENANCE
        scaler.update()
        assert scaler._transitions == 0

    def test_single_transition(self):
        scaler = _scaler(budget=100, atp=100)
        scaler.update()  # MAINTENANCE -> GROWTH
        assert scaler._transitions == 1

    def test_multiple_transitions(self):
        scaler = _scaler(budget=100, atp=100)
        scaler.update()  # -> GROWTH (1)

        scaler.atp_store.atp = 50
        scaler.update()  # -> MAINTENANCE (2)

        scaler.atp_store.atp = 20
        scaler.update()  # -> CONSERVATION (3)

        assert scaler._transitions == 3


# ---------------------------------------------------------------------------
# 12. Full lifecycle: deplete ATP gradually, verify state progression
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_gradual_depletion(self):
        """Deplete ATP from 100 to 0 in steps, verify state progression."""
        store = ATP_Store(budget=100, silent=True)
        scaler = MTORScaler(atp_store=store)

        states_seen: list[ScalingState] = []

        for atp in range(100, -1, -5):
            store.atp = atp
            state = scaler.update()
            if not states_seen or states_seen[-1] != state:
                states_seen.append(state)

        # Should see progression: GROWTH -> MAINTENANCE -> CONSERVATION -> AUTOPHAGY
        assert states_seen[0] == ScalingState.GROWTH
        assert ScalingState.MAINTENANCE in states_seen
        assert ScalingState.CONSERVATION in states_seen
        assert states_seen[-1] == ScalingState.AUTOPHAGY

        # Order must be monotonically "worsening"
        expected_order = [
            ScalingState.GROWTH,
            ScalingState.MAINTENANCE,
            ScalingState.CONSERVATION,
            ScalingState.AUTOPHAGY,
        ]
        assert states_seen == expected_order

    def test_gradual_recovery(self):
        """Recover ATP from 0 to 100 in steps, verify state recovery."""
        store = ATP_Store(budget=100, silent=True)
        store.atp = 0
        scaler = MTORScaler(atp_store=store)

        states_seen: list[ScalingState] = []

        for atp in range(0, 101, 5):
            store.atp = atp
            state = scaler.update()
            if not states_seen or states_seen[-1] != state:
                states_seen.append(state)

        # Should see recovery: AUTOPHAGY -> CONSERVATION -> MAINTENANCE -> GROWTH
        assert states_seen[0] == ScalingState.AUTOPHAGY
        assert states_seen[-1] == ScalingState.GROWTH

    def test_get_statistics(self):
        """Verify statistics dict has expected keys and values."""
        scaler = _scaler(budget=100, atp=100)
        scaler.update()

        stats = scaler.get_statistics()
        assert stats["state"] == "growth"
        assert stats["ampk_ratio"] == pytest.approx(0.0)
        assert stats["rate_of_change"] == pytest.approx(0.0)
        assert stats["recommended_workers"] == 8
        assert stats["transitions"] == 1
