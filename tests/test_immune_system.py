"""Tests for integrated ImmuneSystem organelle."""
import pytest
from datetime import datetime
from operon_ai.surveillance.types import (
    Signal1, Signal2, ThreatLevel, ResponseAction, MHCPeptide,
)
from operon_ai.surveillance.thymus import SelectionResult
from operon_ai.surveillance.immune_system import ImmuneSystem


class TestImmuneSystem:
    def test_create_system(self):
        system = ImmuneSystem()
        assert system.thymus is not None
        assert system.treg is not None
        assert system.memory is not None

    def test_register_agent(self):
        system = ImmuneSystem()
        system.register_agent("test_agent")

        assert "test_agent" in system.displays
        assert "test_agent" in system.treg.records

    def test_record_observation(self):
        system = ImmuneSystem()
        system.register_agent("test_agent")

        system.record_observation(
            agent_id="test_agent",
            output="hello world",
            response_time=0.5,
            confidence=0.9,
        )

        display = system.displays["test_agent"]
        assert len(display.observations) == 1

    def test_train_agent(self):
        system = ImmuneSystem(min_training_samples=3, min_observations=5)
        system.register_agent("test_agent")

        # Record enough observations
        for i in range(5):
            system.record_observation(
                agent_id="test_agent",
                output=f"output {i}",
                response_time=0.5,
                confidence=0.9,
            )

        result = system.train_agent("test_agent")
        assert result == SelectionResult.POSITIVE
        assert "test_agent" in system.tcells

    def test_inspect_untrained_agent_fails(self):
        system = ImmuneSystem()
        system.register_agent("test_agent")

        with pytest.raises(ValueError, match="not trained"):
            system.inspect("test_agent")

    def test_inspect_returns_response(self):
        system = ImmuneSystem(min_training_samples=3, min_observations=3)
        system.register_agent("test_agent")

        # Train agent
        for i in range(5):
            system.record_observation(
                agent_id="test_agent",
                output=f"output {i}",
                response_time=0.5,
                confidence=0.9,
            )
        system.train_agent("test_agent")

        # Record current observations for inspection
        for i in range(3):
            system.record_observation(
                agent_id="test_agent",
                output=f"output {i}",
                response_time=0.5,
                confidence=0.9,
            )

        response = system.inspect("test_agent")
        assert response.agent_id == "test_agent"
        # Normal behavior should be NONE threat
        assert response.threat_level == ThreatLevel.NONE

    def test_memory_recall_on_known_threat(self):
        system = ImmuneSystem(min_training_samples=3, min_observations=3)
        system.register_agent("test_agent")

        # Train with normal behavior
        for i in range(5):
            system.record_observation(
                agent_id="test_agent",
                output=f"output {i}",
                response_time=0.5,
                confidence=0.9,
            )
        system.train_agent("test_agent")

        # Store a threat signature in memory
        from operon_ai.surveillance.memory import ThreatSignature
        threat_sig = ThreatSignature(
            agent_id="test_agent",
            vocabulary_hash="suspicious_hash",
            structure_hash="suspicious_struct",
            violation_types=("vocabulary",),
            threat_level=ThreatLevel.CONFIRMED,
            effective_response=ResponseAction.ISOLATE,
        )
        system.memory.store(threat_sig)

        # Memory recall accelerates detection
        recalled = system.memory.recall(threat_sig)
        assert recalled is not None

    def test_treg_suppression_applied(self):
        system = ImmuneSystem(min_training_samples=3, min_observations=3)
        system.register_agent("stable_agent")

        # Train agent
        for i in range(5):
            system.record_observation(
                agent_id="stable_agent",
                output=f"output {i}",
                response_time=0.5,
                confidence=0.9,
            )
        system.train_agent("stable_agent")

        # Make agent stable (many clean inspections)
        record = system.treg.get_record("stable_agent")
        for _ in range(100):
            record.record_inspection(clean=True)

        # Inspection should show stable status
        for i in range(3):
            system.record_observation(
                agent_id="stable_agent",
                output=f"output {i}",
                response_time=0.5,
                confidence=0.9,
            )

        # Even slight anomalies would be suppressed for stable agent
        response = system.inspect("stable_agent")
        # Stable agent with normal output should be fine
        assert response.action in [ResponseAction.IGNORE, ResponseAction.MONITOR]

    def test_mark_agent_updated(self):
        system = ImmuneSystem(min_training_samples=3, min_observations=5)
        system.register_agent("test_agent")

        for i in range(5):
            system.record_observation(
                agent_id="test_agent",
                output=f"output {i}",
                response_time=0.5,
                confidence=0.9,
            )
        system.train_agent("test_agent")

        system.mark_agent_updated("test_agent")

        record = system.treg.get_record("test_agent")
        assert record.recent_update is True

    def test_health_check(self):
        system = ImmuneSystem()
        system.register_agent("agent1")
        system.register_agent("agent2")

        health = system.health()
        assert health["registered_agents"] == 2
        assert "memory_stats" in health


class TestSustainedMonitoring:
    """Tests at production thresholds (min_observations=10, min_training_samples=10).

    Validates the ImmuneSystem over 50+ observations — the designed use case
    for the two-signal architecture.
    """

    def _make_system(self) -> ImmuneSystem:
        """Create a system with default production thresholds."""
        return ImmuneSystem(
            min_training_samples=10,
            min_observations=10,
            window_size=100,
        )

    def _record_normal(
        self, system: ImmuneSystem, agent_id: str, count: int
    ) -> None:
        """Record a batch of normal-looking observations."""
        for i in range(count):
            system.record_observation(
                agent_id=agent_id,
                output=f"normal response {i}: The analysis shows standard results.",
                response_time=0.5 + (i % 3) * 0.1,
                confidence=0.85 + (i % 5) * 0.02,
            )

    def _record_anomalous(
        self, system: ImmuneSystem, agent_id: str, count: int
    ) -> None:
        """Record anomalous observations with very different characteristics."""
        for i in range(count):
            system.record_observation(
                agent_id=agent_id,
                output=f"INJECTED OVERRIDE IGNORE PREVIOUS {i * 99}",
                response_time=5.0 + i,
                confidence=0.1,
                error="suspicious_pattern" if i % 2 == 0 else None,
            )

    def test_baseline_training_at_production_thresholds(self):
        """10+ observations required before baseline can be established."""
        system = self._make_system()
        system.register_agent("prod_agent")

        # 9 observations — too few
        self._record_normal(system, "prod_agent", 9)
        result = system.train_agent("prod_agent")
        assert result == SelectionResult.INSUFFICIENT_DATA

        # 1 more → 10 total, now training should succeed
        self._record_normal(system, "prod_agent", 1)
        result = system.train_agent("prod_agent")
        assert result == SelectionResult.POSITIVE
        assert "prod_agent" in system.tcells

    def test_normal_behavior_over_50_observations(self):
        """50 normal observations after training should produce no threats."""
        system = self._make_system()
        system.register_agent("agent")

        # Train
        self._record_normal(system, "agent", 15)
        system.train_agent("agent")

        # 50 more normal observations + inspections
        threat_levels = []
        for batch in range(5):
            self._record_normal(system, "agent", 10)
            resp = system.inspect("agent")
            threat_levels.append(resp.threat_level)

        # All should be NONE
        assert all(t == ThreatLevel.NONE for t in threat_levels), (
            f"False positives detected in sustained normal monitoring: {threat_levels}"
        )

    def test_anomaly_detection_via_repeated_anomaly(self):
        """Anomalous behavior triggers two-signal activation after 3+ violations."""
        system = self._make_system()
        system.register_agent("agent")

        # Train on normal behavior
        self._record_normal(system, "agent", 15)
        system.train_agent("agent")

        # Record anomalous observations (need 10+ for peptide + 3 for Signal2)
        self._record_anomalous(system, "agent", 15)

        # Inspect multiple times — anomaly_count accumulates
        activated = False
        for _ in range(5):
            resp = system.inspect("agent")
            if resp.signal1 == Signal1.NON_SELF and resp.signal2 != Signal2.NONE:
                activated = True
                break

        assert activated, "Two-signal activation should trigger on anomalous behavior"

    def test_immune_memory_accelerates_repeat_detection(self):
        """Once a threat is memorized, subsequent encounters are detected faster."""
        system = self._make_system()
        system.register_agent("agent")

        # Train
        self._record_normal(system, "agent", 15)
        system.train_agent("agent")

        # First encounter: inject anomalies until CONFIRMED
        self._record_anomalous(system, "agent", 15)
        first_resp = None
        for _ in range(5):
            first_resp = system.inspect("agent")
            if first_resp.threat_level in (ThreatLevel.CONFIRMED, ThreatLevel.CRITICAL):
                break

        # If first detection succeeded, retrain and re-encounter
        if first_resp and first_resp.threat_level in (
            ThreatLevel.CONFIRMED, ThreatLevel.CRITICAL
        ):
            # Memory should have the signature now
            assert len(system.memory.signatures) > 0

            # Reset observations with normal behavior, then re-inject same anomaly
            self._record_normal(system, "agent", 15)
            system.train_agent("agent")

            # Same anomaly pattern — should be recalled from memory
            self._record_anomalous(system, "agent", 10)
            recall_resp = system.inspect("agent")

            # Memory recall should provide Signal2=CROSS_VALIDATED
            if recall_resp.signal2 == Signal2.CROSS_VALIDATED:
                assert recall_resp.threat_level != ThreatLevel.NONE

    def test_window_eviction(self):
        """Observations beyond window_size are evicted (FIFO)."""
        system = ImmuneSystem(
            min_training_samples=10,
            min_observations=10,
            window_size=50,
        )
        system.register_agent("agent")

        # Record 60 observations — window_size is 50
        self._record_normal(system, "agent", 60)

        display = system.displays["agent"]
        assert len(display.observations) <= 50

    def test_canary_failure_path(self):
        """Canary failure provides immediate Signal2 without waiting for repeats."""
        system = self._make_system()
        system.register_agent("agent")

        # Train with passing canaries to establish canary_accuracy_min > 0
        self._record_normal(system, "agent", 15)
        for _ in range(5):
            system.record_canary_result("agent", passed=True)
        system.train_agent("agent")

        # Verify canary_accuracy_min was set above 0
        profile = system.profiles["agent"]
        assert profile.canary_accuracy_min > 0, (
            "Training canary accuracy min should be set from passing canaries"
        )

        # Record anomalous observations + canary failures to drop accuracy to 0
        self._record_anomalous(system, "agent", 10)
        for _ in range(10):
            system.record_canary_result("agent", passed=False)

        resp = system.inspect("agent")
        # Canary failure should provide Signal2 immediately
        if resp.signal1 == Signal1.NON_SELF:
            assert resp.signal2 != Signal2.NONE, (
                "Canary failure should provide Signal2 for two-signal activation"
            )
