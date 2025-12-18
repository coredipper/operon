# tests/test_surveillance_types.py
"""Tests for surveillance system types."""
import pytest
from datetime import datetime
from operon_ai.surveillance.types import (
    Signal1, Signal2, ThreatLevel, ResponseAction,
    MHCPeptide, ActivationState,
)


class TestEnums:
    def test_signal1_values(self):
        assert Signal1.SELF.value == "self"
        assert Signal1.NON_SELF.value == "non_self"
        assert Signal1.UNKNOWN.value == "unknown"

    def test_signal2_values(self):
        assert Signal2.NONE.value == "none"
        assert Signal2.CANARY_FAILED.value == "canary"
        assert Signal2.CROSS_VALIDATED.value == "cross"
        assert Signal2.REPEATED_ANOMALY.value == "repeat"

    def test_threat_level_values(self):
        assert ThreatLevel.NONE.value == "none"
        assert ThreatLevel.SUSPICIOUS.value == "suspicious"
        assert ThreatLevel.CONFIRMED.value == "confirmed"
        assert ThreatLevel.CRITICAL.value == "critical"

    def test_response_action_values(self):
        assert ResponseAction.IGNORE.value == "ignore"
        assert ResponseAction.MONITOR.value == "monitor"
        assert ResponseAction.ISOLATE.value == "isolate"
        assert ResponseAction.SHUTDOWN.value == "shutdown"


class TestMHCPeptide:
    def test_create_peptide(self):
        peptide = MHCPeptide(
            agent_id="test_agent",
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc123",
            structure_hash="def456",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=("timeout",),
        )
        assert peptide.agent_id == "test_agent"
        assert peptide.output_length_mean == 100.0

    def test_peptide_similarity_same_agent(self):
        peptide1 = MHCPeptide(
            agent_id="test",
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc",
            structure_hash="def",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=(),
        )
        peptide2 = MHCPeptide(
            agent_id="test",
            timestamp=datetime.utcnow(),
            output_length_mean=105.0,  # Similar
            output_length_std=10.0,
            response_time_mean=0.52,   # Similar
            response_time_std=0.1,
            vocabulary_hash="abc",     # Same
            structure_hash="def",      # Same
            confidence_mean=0.88,      # Similar
            confidence_std=0.05,
            error_rate=0.02,           # Similar
            error_types=(),
        )
        similarity = peptide1.similarity(peptide2)
        assert similarity > 0.8  # High similarity

    def test_peptide_similarity_different_agent(self):
        peptide1 = MHCPeptide(
            agent_id="agent_a",
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc",
            structure_hash="def",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=(),
        )
        peptide2 = MHCPeptide(
            agent_id="agent_b",  # Different agent
            timestamp=datetime.utcnow(),
            output_length_mean=100.0,
            output_length_std=10.0,
            response_time_mean=0.5,
            response_time_std=0.1,
            vocabulary_hash="abc",
            structure_hash="def",
            confidence_mean=0.9,
            confidence_std=0.05,
            error_rate=0.01,
            error_types=(),
        )
        assert peptide1.similarity(peptide2) == 0.0


class TestActivationState:
    def test_create_state(self):
        state = ActivationState(agent_id="test")
        assert state.agent_id == "test"
        assert state.signal1 == Signal1.SELF
        assert state.signal2 == Signal2.NONE

    def test_is_activated_requires_both_signals(self):
        state = ActivationState(agent_id="test")
        assert state.is_activated is False

        state.signal1 = Signal1.NON_SELF
        assert state.is_activated is False  # Still needs signal2

        state.signal2 = Signal2.CANARY_FAILED
        assert state.is_activated is True
