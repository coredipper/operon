"""Tests for the cellular energy management system (Metabolism)."""

import pytest

from operon_ai.state.metabolism import (
    ATP_Store,
    EnergyType,
    MetabolicAccessPolicy,
    MetabolicState,
    _verify_priority_gating,
)


def test_metabolic_access_policy_get_min_strength():
    """Test getting minimum marker strength for each metabolic state."""
    policy = MetabolicAccessPolicy()

    assert policy.get_min_strength(MetabolicState.FEASTING) == 1
    assert policy.get_min_strength(MetabolicState.NORMAL) == 1
    assert policy.get_min_strength(MetabolicState.CONSERVING) == 3
    assert policy.get_min_strength(MetabolicState.STARVING) == 4
    assert policy.get_min_strength(MetabolicState.DORMANT) is None


def test_verify_priority_gating():
    """Test derivation replay for ATP priority gating guarantee."""
    # Valid params
    valid_params = {
        "budget": 100,
        "priority_threshold_starving": 10,
        "priority_threshold_dormant": 20,
    }
    holds, data = _verify_priority_gating(valid_params)
    assert holds is True
    assert data["budget"] == 100
    assert data["threshold_starving"] == 10
    assert data["threshold_dormant"] == 20

    # Invalid budget
    invalid_budget = valid_params.copy()
    invalid_budget["budget"] = 0
    holds, _ = _verify_priority_gating(invalid_budget)
    assert holds is False

    # Invalid thresholds ordering
    invalid_threshold = valid_params.copy()
    invalid_threshold["priority_threshold_starving"] = 30
    holds, _ = _verify_priority_gating(invalid_threshold)
    assert holds is False


class TestATPStore:
    def test_initialization(self):
        """Test ATP_Store initializes with correct values."""
        store = ATP_Store(
            budget=100,
            gtp_budget=50,
            nadh_reserve=20,
            max_debt=10,
            silent=True
        )
        assert store.atp == 100
        assert store.max_atp == 100
        assert store.gtp == 50
        assert store.nadh == 20
        assert store.get_balance(EnergyType.ATP) == 100
        assert store.get_balance(EnergyType.GTP) == 50
        assert store.get_balance(EnergyType.NADH) == 20
        assert store.get_state() == MetabolicState.NORMAL

    def test_consume_success(self):
        """Test successful energy consumption."""
        store = ATP_Store(budget=100, silent=True)
        assert store.consume(20, "operation") is True
        assert store.atp == 80

        # Test GTP consumption
        store_gtp = ATP_Store(budget=100, gtp_budget=50, silent=True)
        assert store_gtp.consume(10, "premium_op", EnergyType.GTP) is True
        assert store_gtp.gtp == 40

    def test_consume_failure(self):
        """Test consumption fails when out of energy and no debt allowed."""
        store = ATP_Store(budget=10, silent=True)
        assert store.consume(20, "expensive_op") is False
        assert store.atp == 10  # Balance unchanged

    def test_consume_with_debt(self):
        """Test consumption goes into debt if allowed and configured."""
        store = ATP_Store(budget=10, max_debt=20, silent=True)
        assert store.consume(20, "critical_op", allow_debt=True) is True
        assert store.atp == 0
        assert store.get_debt() == 10

    def test_regenerate(self):
        """Test regenerating energy and paying off debt."""
        store = ATP_Store(budget=100, silent=True)
        store.consume(50)
        assert store.atp == 50

        store.regenerate(20)
        assert store.atp == 70

        # Test debt repayment
        # Budget must be greater than zero to avoid division by zero when calculating ratio
        store_debt = ATP_Store(budget=100, max_debt=50, silent=True)
        store_debt.atp = 0
        store_debt.consume(20, allow_debt=True)
        assert store_debt.get_debt() == 20
        assert store_debt.atp == 0

        store_debt.regenerate(30)
        assert store_debt.get_debt() == 0
        assert store_debt.atp == 10

    def test_transfer_to(self):
        """Test transferring energy between stores."""
        store_a = ATP_Store(budget=100, silent=True)
        store_b = ATP_Store(budget=100, silent=True)
        store_b.atp = 0

        assert store_a.transfer_to(store_b, 40) is True
        assert store_a.atp == 60
        assert store_b.atp == 40

        # Test insufficient funds
        assert store_a.transfer_to(store_b, 100) is False

    def test_convert_nadh_to_atp(self):
        """Test converting NADH reserve to ATP."""
        store = ATP_Store(budget=100, nadh_reserve=50, silent=True)
        store.consume(60)  # atp is now 40

        converted = store.convert_nadh_to_atp(30)
        assert converted == 30
        assert store.atp == 70
        assert store.nadh == 20

        # Test max capacity limit
        converted = store.convert_nadh_to_atp(50)
        assert converted == 20  # Can only convert up to max_atp (100 - 70 = 30, but nadh is 20)
        assert store.atp == 90
        assert store.nadh == 0

    def test_state_transitions(self):
        """Test transitions between metabolic states."""
        store = ATP_Store(budget=100, silent=True)
        assert store.get_state() == MetabolicState.NORMAL  # 1.0 ratio initializes to normal

        # manually trigger an update
        store._update_state()
        assert store.get_state() == MetabolicState.FEASTING  # 1.0 ratio

        store.consume(20)
        assert store.get_state() == MetabolicState.NORMAL  # 0.8 ratio

        store.consume(60)
        assert store.get_state() == MetabolicState.CONSERVING  # 0.2 ratio

        store.consume(15) # 0.05 ratio
        assert store.get_state() == MetabolicState.STARVING

    def test_enter_exit_dormancy(self):
        """Test dormancy state."""
        store = ATP_Store(budget=100, silent=True)
        store.enter_dormancy()
        assert store.get_state() == MetabolicState.DORMANT

        store.exit_dormancy()
        assert store.get_state() == MetabolicState.FEASTING

    def test_apply_debt_interest(self):
        """Test debt interest application."""
        store = ATP_Store(budget=100, max_debt=100, debt_interest=0.1, silent=True)
        store.atp = 0
        store.consume(50, allow_debt=True)
        assert store.get_debt() == 50

        store.apply_debt_interest()
        assert store.get_debt() == 55
