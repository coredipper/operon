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
