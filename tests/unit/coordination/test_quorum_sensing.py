"""Tests for biologically-faithful quorum sensing."""

from operon_ai.coordination.quorum_sensing import (
    AutoinducerSignal,
    QuorumSensingBio,
    SignalEnvironment,
)


class TestAutoinducerSignal:
    def test_frozen(self):
        s = AutoinducerSignal("a1", "AI-1", 0.5, 1.0)
        assert s.agent_id == "a1"
        assert s.concentration == 0.5


class TestSignalEnvironment:
    def test_deposit_and_concentration(self):
        env = SignalEnvironment(decay_half_life=5.0)
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 0.0))
        # At t=0, no decay
        assert env.get_concentration("AI-1", 0.0) == 1.0

    def test_exponential_decay(self):
        env = SignalEnvironment(decay_half_life=5.0)
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 0.0))
        # At t=5 (one half-life), concentration should be ~0.5
        c = env.get_concentration("AI-1", 5.0)
        assert abs(c - 0.5) < 0.01

    def test_two_half_lives(self):
        env = SignalEnvironment(decay_half_life=5.0)
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 0.0))
        c = env.get_concentration("AI-1", 10.0)
        assert abs(c - 0.25) < 0.01

    def test_accumulation(self):
        env = SignalEnvironment(decay_half_life=5.0)
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 0.0))
        env.deposit(AutoinducerSignal("a2", "AI-1", 1.0, 0.0))
        assert env.get_concentration("AI-1", 0.0) == 2.0

    def test_empty_signal_type(self):
        env = SignalEnvironment()
        assert env.get_concentration("AI-1", 0.0) == 0.0

    def test_prune_removes_decayed(self):
        env = SignalEnvironment(decay_half_life=1.0, noise_floor=0.01)
        env.deposit(AutoinducerSignal("a1", "AI-1", 0.1, 0.0))
        # After many half-lives, signal is below noise floor
        removed = env.prune(50.0)
        assert removed == 1
        assert env.total_signals == 0

    def test_prune_keeps_fresh(self):
        env = SignalEnvironment(decay_half_life=5.0, noise_floor=0.01)
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 10.0))
        removed = env.prune(10.0)
        assert removed == 0
        assert env.total_signals == 1

    def test_clear(self):
        env = SignalEnvironment()
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 0.0))
        env.clear()
        assert env.total_signals == 0

    def test_multiple_signal_types(self):
        env = SignalEnvironment(decay_half_life=5.0)
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 0.0))
        env.deposit(AutoinducerSignal("a1", "AI-2", 2.0, 0.0))
        assert env.get_concentration("AI-1", 0.0) == 1.0
        assert env.get_concentration("AI-2", 0.0) == 2.0


class TestQuorumSensingBio:
    def test_threshold_scales_with_population(self):
        qs = QuorumSensingBio(population_size=10, threshold_base=10.0)
        t10 = qs._threshold()
        qs.population_size = 100
        t100 = qs._threshold()
        # log(100) > log(10), so threshold increases
        assert t100 > t10
        # But sublinearly — not 10x
        assert t100 < t10 * 3

    def test_no_activation_without_signals(self):
        qs = QuorumSensingBio(population_size=5)
        assert not qs.should_activate("AI-1", 0.0)
        assert qs.get_activation_level("AI-1", 0.0) == 0.0

    def test_activation_with_enough_signal(self):
        qs = QuorumSensingBio(population_size=5, threshold_base=1.0)
        # Deposit more than threshold
        for i in range(20):
            qs.produce_signal(f"a{i}", suspicion=1.0, current_time=0.0)
        assert qs.should_activate("AI-1", 0.0)

    def test_no_activation_below_threshold(self):
        qs = QuorumSensingBio(population_size=10, threshold_base=10.0)
        # Single weak signal — well below threshold
        qs.produce_signal("a1", suspicion=0.1, current_time=0.0)
        assert not qs.should_activate("AI-1", 0.0)

    def test_decay_prevents_stale_activation(self):
        qs = QuorumSensingBio(
            population_size=5,
            threshold_base=1.0,
            environment=SignalEnvironment(decay_half_life=1.0),
        )
        # Deposit enough to activate at t=0
        for i in range(20):
            qs.produce_signal(f"a{i}", suspicion=1.0, current_time=0.0)
        assert qs.should_activate("AI-1", 0.0)
        # After many half-lives, signals have decayed
        assert not qs.should_activate("AI-1", 50.0)

    def test_gradual_accumulation(self):
        qs = QuorumSensingBio(
            population_size=5,
            threshold_base=1.0,
            environment=SignalEnvironment(decay_half_life=10.0),
        )
        # Add signals one at a time
        for t in range(50):
            qs.produce_signal("a1", suspicion=0.5, current_time=float(t))
            if qs.should_activate("AI-1", float(t)):
                assert t > 0  # Shouldn't activate on first signal
                break
        else:
            raise AssertionError("Never activated despite repeated signaling")

    def test_suspicion_proportional_to_concentration(self):
        qs = QuorumSensingBio(population_size=5, threshold_base=1.0)
        qs.produce_signal("a1", suspicion=0.1, current_time=0.0)
        low = qs.get_activation_level("AI-1", 0.0)
        qs.reset()
        qs.produce_signal("a1", suspicion=1.0, current_time=0.0)
        high = qs.get_activation_level("AI-1", 0.0)
        assert high > low * 5  # 10x suspicion => ~10x activation

    def test_separate_signal_types(self):
        qs = QuorumSensingBio(population_size=5, threshold_base=1.0)
        qs.produce_signal("a1", suspicion=1.0, current_time=0.0, signal_type="AI-1")
        qs.produce_signal("a2", suspicion=1.0, current_time=0.0, signal_type="AI-2")
        # Each type tracked independently
        level_1 = qs.get_activation_level("AI-1", 0.0)
        level_2 = qs.get_activation_level("AI-2", 0.0)
        assert level_1 == level_2  # Same signal, same level
        # But if we add more to AI-1 only
        qs.produce_signal("a3", suspicion=1.0, current_time=0.0, signal_type="AI-1")
        assert qs.get_activation_level("AI-1", 0.0) > qs.get_activation_level("AI-2", 0.0)

    def test_reset(self):
        qs = QuorumSensingBio(population_size=5, threshold_base=1.0)
        qs.produce_signal("a1", suspicion=1.0, current_time=0.0)
        qs.reset()
        assert qs.get_activation_level("AI-1", 0.0) == 0.0

    def test_statistics(self):
        qs = QuorumSensingBio(population_size=5, threshold_base=1.0)
        qs.produce_signal("a1", suspicion=0.5, current_time=0.0)
        stats = qs.get_statistics(0.0)
        assert stats["population_size"] == 5
        assert stats["total_signals"] == 1
        assert stats["AI-1_concentration"] == 0.5

    def test_negative_suspicion_clamped(self):
        qs = QuorumSensingBio(population_size=5, threshold_base=1.0)
        qs.produce_signal("a1", suspicion=-0.5, current_time=0.0)
        # Negative suspicion clamped to 0
        assert qs.get_activation_level("AI-1", 0.0) == 0.0

    def test_out_of_order_deposit_preserves_earlier_signals(self):
        """Backfilled deposit at earlier timestamp must not prune signals
        that are still valid at the earlier time."""
        env = SignalEnvironment(decay_half_life=5.0, noise_floor=0.01)
        # Deposit at t=10
        env.deposit(AutoinducerSignal("a1", "AI-1", 1.0, 10.0))
        # Backfill at t=5 — should NOT prune the t=10 signal
        env.deposit(AutoinducerSignal("a2", "AI-1", 0.5, 5.0))
        # Both signals should be present
        assert env.total_signals == 2
        # Concentration at t=10 should include both
        c = env.get_concentration("AI-1", 10.0)
        assert c > 1.0  # t=10 signal (full) + t=5 signal (decayed but present)

    def test_noise_floor_in_read_only_period(self):
        """Signals that decay below noise_floor should return 0 concentration
        even without any new deposits (read-only period)."""
        env = SignalEnvironment(decay_half_life=1.0, noise_floor=0.01)
        env.deposit(AutoinducerSignal("a1", "AI-1", 0.1, 0.0))
        # At t=0, concentration is 0.1 (above noise floor)
        assert env.get_concentration("AI-1", 0.0) > 0
        # After many half-lives, signal decays below noise_floor
        # No new deposits — pure read-only
        c = env.get_concentration("AI-1", 50.0)
        assert c == 0.0  # Below noise floor, should not contribute
        # But the signal is still stored (not mutated by read)
        assert env.total_signals == 1


class TestCalibration:
    """Tests for auto-calibrated thresholds (categorical certificate)."""

    def test_calibrate_scales_with_population(self):
        qs5 = QuorumSensingBio(population_size=5)
        qs5.calibrate()
        t5 = qs5._threshold()

        qs20 = QuorumSensingBio(population_size=20)
        qs20.calibrate()
        t20 = qs20._threshold()

        # Threshold scales linearly with N (not log) when calibrated
        assert t20 > t5
        assert abs(t20 / t5 - 4.0) < 0.01  # 20/5 = 4x

    def test_calibrate_prevents_false_activation(self):
        """Normal traffic at steady state should not trigger activation."""
        import random
        for n in [5, 10, 20, 50]:
            qs = QuorumSensingBio(
                population_size=n,
                environment=SignalEnvironment(decay_half_life=5.0),
            )
            qs.calibrate()

            rng = random.Random(42)
            # Run normal traffic for enough steps to reach steady state
            for t in range(60):
                for i in range(n):
                    susp = max(0, 0.15 + rng.gauss(0, 0.05))
                    qs.produce_signal(f"a{i}", susp, float(t))

            # Should NOT activate (certificate guarantee)
            assert not qs.should_activate("AI-1", 59.0), (
                f"False activation at N={n}"
            )

    def test_calibrate_allows_detection_at_compromise(self):
        """Compromised traffic at 40% should trigger activation."""
        import random
        qs = QuorumSensingBio(
            population_size=20,
            environment=SignalEnvironment(decay_half_life=5.0),
        )
        qs.calibrate()

        rng = random.Random(42)
        n_compromised = 8  # 40%
        for t in range(60):
            for i in range(20):
                if i < n_compromised:
                    susp = max(0, 0.8 + rng.gauss(0, 0.1))
                else:
                    susp = max(0, 0.15 + rng.gauss(0, 0.05))
                qs.produce_signal(f"a{i}", susp, float(t))

        assert qs.should_activate("AI-1", 59.0)

    def test_calibrate_adjusts_when_population_changes(self):
        """Re-calibrating after population change preserves guarantee."""
        qs = QuorumSensingBio(population_size=10)
        qs.calibrate()
        t10 = qs._threshold()

        # Simulate compilation to a framework with more agents
        qs.population_size = 30
        # Must re-calibrate — threshold should change
        qs.calibrate()
        t30 = qs._threshold()

        assert t30 > t10
        assert abs(t30 / t10 - 3.0) < 0.01  # 30/10 = 3x

    def test_certificate_holds_across_populations(self):
        """The no-false-activation guarantee holds for any N."""
        import random
        for n in [5, 10, 20, 50]:
            qs = QuorumSensingBio(
                population_size=n,
                environment=SignalEnvironment(decay_half_life=5.0),
            )
            qs.calibrate()

            rng = random.Random(n)  # Deterministic per N
            for t in range(80):
                for i in range(n):
                    susp = max(0, 0.15 + rng.gauss(0, 0.05))
                    qs.produce_signal(f"a{i}", susp, float(t))

            level = qs.get_activation_level("AI-1", 79.0)
            assert level < 1.0, (
                f"Certificate violation at N={n}: activation={level:.3f}"
            )

    def test_calibrate_with_non_unit_interval(self):
        """Certificate holds when emission interval != 1.0."""
        import random
        for dt in [0.5, 2.0, 5.0]:
            qs = QuorumSensingBio(
                population_size=10,
                environment=SignalEnvironment(decay_half_life=5.0),
                emission_interval=dt,
            )
            qs.calibrate()

            rng = random.Random(42)
            t = 0.0
            for _ in range(100):
                for i in range(10):
                    susp = max(0, 0.15 + rng.gauss(0, 0.05))
                    qs.produce_signal(f"a{i}", susp, t)
                t += dt

            level = qs.get_activation_level("AI-1", t - dt)
            assert level < 1.0, (
                f"Certificate violation at dt={dt}: activation={level:.3f}"
            )

    def test_backward_compat_uncalibrated(self):
        """Without calibrate(), original log-scaling is used."""
        qs = QuorumSensingBio(population_size=10, threshold_base=10.0)
        import math
        expected = math.log(10) * 10.0
        assert abs(qs._threshold() - expected) < 0.01
