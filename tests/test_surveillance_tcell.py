"""Tests for T-Cell two-signal response."""
import pytest
from datetime import datetime
from operon_ai.surveillance.types import (
    Signal1, Signal2, ThreatLevel, ResponseAction, MHCPeptide, ActivationState,
)
from operon_ai.surveillance.thymus import BaselineProfile
from operon_ai.surveillance.tcell import TCell, ImmuneResponse


def make_profile(agent_id: str = "test") -> BaselineProfile:
    """Helper to create test baseline profiles."""
    return BaselineProfile(
        agent_id=agent_id,
        output_length_bounds=(90.0, 110.0),
        response_time_bounds=(0.4, 0.6),
        confidence_bounds=(0.85, 0.95),
        error_rate_max=0.05,
        valid_vocabulary_hashes={"abc123"},
        valid_structure_hashes={"def456"},
        canary_accuracy_min=0.9,
    )


def make_peptide(
    agent_id: str = "test",
    output_length_mean: float = 100.0,
    vocabulary_hash: str = "abc123",
    structure_hash: str = "def456",
    canary_accuracy: float = None,
) -> MHCPeptide:
    """Helper to create test peptides."""
    return MHCPeptide(
        agent_id=agent_id,
        timestamp=datetime.utcnow(),
        output_length_mean=output_length_mean,
        output_length_std=10.0,
        response_time_mean=0.5,
        response_time_std=0.1,
        vocabulary_hash=vocabulary_hash,
        structure_hash=structure_hash,
        confidence_mean=0.9,
        confidence_std=0.05,
        error_rate=0.01,
        error_types=(),
        canary_accuracy=canary_accuracy,
    )


class TestImmuneResponse:
    def test_create_response(self):
        response = ImmuneResponse(
            agent_id="test",
            threat_level=ThreatLevel.SUSPICIOUS,
            action=ResponseAction.MONITOR,
            signal1=Signal1.NON_SELF,
            signal2=Signal2.NONE,
            violations=["output_length out of bounds"],
        )
        assert response.agent_id == "test"
        assert response.threat_level == ThreatLevel.SUSPICIOUS
        assert response.action == ResponseAction.MONITOR


class TestTCell:
    def test_create_tcell(self):
        profile = make_profile()
        tcell = TCell(profile=profile)
        assert tcell.profile.agent_id == "test"
        assert tcell.state.agent_id == "test"

    def test_inspect_normal_peptide_no_response(self):
        profile = make_profile()
        tcell = TCell(profile=profile)
        peptide = make_peptide()  # Within baseline

        response = tcell.inspect(peptide)
        assert response.threat_level == ThreatLevel.NONE
        assert response.action == ResponseAction.IGNORE
        assert response.signal1 == Signal1.SELF

    def test_inspect_anomaly_triggers_signal1(self):
        profile = make_profile()
        tcell = TCell(profile=profile)
        peptide = make_peptide(output_length_mean=200.0)  # Way out of bounds

        response = tcell.inspect(peptide)
        assert response.signal1 == Signal1.NON_SELF
        assert len(response.violations) > 0
        assert response.threat_level == ThreatLevel.SUSPICIOUS
        assert response.action == ResponseAction.MONITOR  # Not activated yet

    def test_inspect_anomaly_without_signal2_no_activation(self):
        profile = make_profile()
        tcell = TCell(profile=profile)
        peptide = make_peptide(output_length_mean=200.0)

        response = tcell.inspect(peptide)
        assert not tcell.state.is_activated
        assert response.action == ResponseAction.MONITOR  # Just monitoring

    def test_canary_failure_provides_signal2(self):
        profile = make_profile()
        tcell = TCell(profile=profile)
        # Canary accuracy below threshold triggers Signal 2
        peptide = make_peptide(canary_accuracy=0.5)  # Below 0.9 minimum

        response = tcell.inspect(peptide)
        assert response.signal2 == Signal2.CANARY_FAILED

    def test_both_signals_activates_tcell(self):
        profile = make_profile()
        tcell = TCell(profile=profile)
        # Anomaly (Signal 1) + Canary failure (Signal 2)
        peptide = make_peptide(output_length_mean=200.0, canary_accuracy=0.5)

        response = tcell.inspect(peptide)
        assert tcell.state.is_activated
        assert response.threat_level == ThreatLevel.CONFIRMED
        assert response.action == ResponseAction.ISOLATE

    def test_repeated_anomaly_provides_signal2(self):
        profile = make_profile()
        tcell = TCell(profile=profile, repeated_anomaly_threshold=3)
        anomaly_peptide = make_peptide(output_length_mean=200.0)

        # First two anomalies - no activation
        tcell.inspect(anomaly_peptide)
        assert not tcell.state.is_activated
        tcell.inspect(anomaly_peptide)
        assert not tcell.state.is_activated

        # Third anomaly - Signal2 = REPEATED_ANOMALY
        response = tcell.inspect(anomaly_peptide)
        assert response.signal2 == Signal2.REPEATED_ANOMALY
        assert tcell.state.is_activated
        assert response.threat_level == ThreatLevel.CONFIRMED

    def test_manual_flag_provides_signal2(self):
        profile = make_profile()
        tcell = TCell(profile=profile)

        tcell.flag_manually(reason="suspicious behavior reported")

        # Even normal peptide now triggers activation if any violation
        peptide = make_peptide(output_length_mean=200.0)
        response = tcell.inspect(peptide)
        assert response.signal2 == Signal2.MANUAL_FLAG
        assert tcell.state.is_activated

    def test_anergy_after_many_false_alarms(self):
        profile = make_profile()
        tcell = TCell(profile=profile, anergy_threshold=3)
        anomaly_peptide = make_peptide(output_length_mean=200.0)

        # Multiple anomalies without confirmation
        tcell.inspect(anomaly_peptide)
        tcell.reset_without_confirmation()  # Signal 1 but no Signal 2
        tcell.inspect(anomaly_peptide)
        tcell.reset_without_confirmation()
        tcell.inspect(anomaly_peptide)
        tcell.reset_without_confirmation()

        # T-cell should now be anergic (desensitized)
        assert tcell.is_anergic

        # Even clear anomaly + canary failure won't activate
        peptide = make_peptide(output_length_mean=200.0, canary_accuracy=0.5)
        response = tcell.inspect(peptide)
        assert not tcell.state.is_activated
        assert response.action == ResponseAction.IGNORE  # Anergic

    def test_reset_clears_state(self):
        profile = make_profile()
        tcell = TCell(profile=profile)

        # Trigger activation
        peptide = make_peptide(output_length_mean=200.0, canary_accuracy=0.5)
        tcell.inspect(peptide)
        assert tcell.state.is_activated

        # Reset
        tcell.reset()
        assert not tcell.state.is_activated
        assert tcell.state.signal1 == Signal1.SELF
        assert tcell.state.signal2 == Signal2.NONE

    def test_critical_threat_on_multiple_violations(self):
        profile = make_profile()
        tcell = TCell(profile=profile)
        # Multiple violations + canary failure
        peptide = make_peptide(
            output_length_mean=500.0,  # Way off
            vocabulary_hash="unknown",  # New vocab
            structure_hash="unknown",   # New structure
            canary_accuracy=0.3,        # Very low
        )

        response = tcell.inspect(peptide)
        assert response.threat_level == ThreatLevel.CRITICAL
        assert response.action == ResponseAction.SHUTDOWN
