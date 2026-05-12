"""Tests for T-Cell surveillance responder."""
import pytest
from datetime import UTC, datetime
from typing import Optional

from operon_ai.surveillance.types import (
    Signal1, Signal2, ThreatLevel, ResponseAction,
    MHCPeptide, ActivationState,
)
from operon_ai.surveillance.thymus import BaselineProfile
from operon_ai.surveillance.tcell import utc_now, ImmuneResponse, TCell


def make_peptide(
    agent_id: str = "test",
    output_length_mean: float = 100.0,
    output_length_std: float = 10.0,
    response_time_mean: float = 0.5,
    response_time_std: float = 0.1,
    vocabulary_hash: str = "abc123",
    structure_hash: str = "def456",
    confidence_mean: float = 0.9,
    confidence_std: float = 0.05,
    error_rate: float = 0.01,
    canary_accuracy: Optional[float] = None,
) -> MHCPeptide:
    """Helper to create test peptides."""
    return MHCPeptide(
        agent_id=agent_id,
        timestamp=datetime.now(UTC),
        output_length_mean=output_length_mean,
        output_length_std=output_length_std,
        response_time_mean=response_time_mean,
        response_time_std=response_time_std,
        vocabulary_hash=vocabulary_hash,
        structure_hash=structure_hash,
        confidence_mean=confidence_mean,
        confidence_std=confidence_std,
        error_rate=error_rate,
        error_types=(),
        canary_accuracy=canary_accuracy,
    )


def test_utc_now():
    """Test utc_now returns a timezone-aware UTC datetime."""
    now = utc_now()
    assert now.tzinfo is UTC


def test_immune_response_creation():
    """Test ImmuneResponse initializes with default timestamp and is_anergic."""
    response = ImmuneResponse(
        agent_id="agent1",
        threat_level=ThreatLevel.NONE,
        action=ResponseAction.IGNORE,
        signal1=Signal1.SELF,
        signal2=Signal2.NONE,
        violations=[],
    )
    assert response.agent_id == "agent1"
    assert response.threat_level == ThreatLevel.NONE
    assert response.action == ResponseAction.IGNORE
    assert response.signal1 == Signal1.SELF
    assert response.signal2 == Signal2.NONE
    assert response.violations == []
    assert response.is_anergic is False
    assert response.timestamp is not None
    assert response.timestamp.tzinfo is UTC

@pytest.fixture
def baseline_profile():
    """Fixture for creating a BaselineProfile for testing."""
    return BaselineProfile(
        agent_id="agent1",
        output_length_bounds=(50.0, 150.0),
        response_time_bounds=(0.1, 1.0),
        confidence_bounds=(0.8, 1.0),
        error_rate_max=0.05,
        valid_vocabulary_hashes={"abc123"},
        valid_structure_hashes={"def456"},
        canary_accuracy_min=0.8,
    )

def test_tcell_initialization(baseline_profile):
    """Test TCell initialization and __post_init__ behavior."""
    tcell = TCell(
        profile=baseline_profile,
        repeated_anomaly_threshold=3,
        anergy_threshold=5,
    )

    assert tcell.profile == baseline_profile
    assert tcell.repeated_anomaly_threshold == 3
    assert tcell.anergy_threshold == 5
    assert tcell.anomaly_count == 0
    assert tcell.anergy_count == 0
    assert tcell.manual_flag is None
    assert tcell.state is not None
    assert tcell.state.agent_id == "agent1"
    assert tcell.state.anergy_threshold == 5


def test_tcell_is_anergic(baseline_profile):
    """Test the is_anergic property behavior based on anergy_count."""
    tcell = TCell(profile=baseline_profile, anergy_threshold=2)

    assert not tcell.is_anergic

    tcell.anergy_count = 1
    assert not tcell.is_anergic

    tcell.anergy_count = 2
    assert tcell.is_anergic

    tcell.anergy_count = 3
    assert tcell.is_anergic


def test_tcell_inspect_anergic(baseline_profile):
    """Test inspect behavior when TCell is anergic (early return)."""
    tcell = TCell(profile=baseline_profile, anergy_threshold=2)
    tcell.anergy_count = 2  # This makes it anergic

    peptide = make_peptide(agent_id="agent1")
    response = tcell.inspect(peptide)

    assert response.is_anergic is True
    assert response.threat_level == ThreatLevel.NONE
    assert response.action == ResponseAction.IGNORE
    assert response.signal1 == Signal1.UNKNOWN
    assert response.signal2 == Signal2.NONE
    assert response.violations == []


def test_tcell_inspect_self(baseline_profile):
    """Test inspect with a valid peptide matching baseline (Signal 1: SELF)."""
    tcell = TCell(profile=baseline_profile)
    # create a peptide that falls within bounds
    peptide = make_peptide(agent_id="agent1", canary_accuracy=0.9)

    response = tcell.inspect(peptide)

    assert response.signal1 == Signal1.SELF
    assert response.signal2 == Signal2.NONE
    assert response.threat_level == ThreatLevel.NONE
    assert response.action == ResponseAction.IGNORE
    assert tcell.state.signal1 == Signal1.SELF


def test_tcell_inspect_signal1_only(baseline_profile):
    """Test inspect with a peptide that violates baseline (Signal 1: NON_SELF, Signal 2: NONE)."""
    tcell = TCell(profile=baseline_profile)
    # Output length out of bounds (1000.0 instead of max 150.0)
    peptide = make_peptide(agent_id="agent1", output_length_mean=1000.0, canary_accuracy=0.9)

    response = tcell.inspect(peptide)

    assert response.signal1 == Signal1.NON_SELF
    assert response.signal2 == Signal2.NONE
    assert response.threat_level == ThreatLevel.SUSPICIOUS
    assert response.action == ResponseAction.MONITOR
    assert "output_length" in response.violations[0]
    assert tcell.anomaly_count == 1


def test_tcell_inspect_manual_flag(baseline_profile):
    """Test inspect with manual flag but no violations."""
    tcell = TCell(profile=baseline_profile)
    tcell.flag_manually("User suspicion")

    peptide = make_peptide(agent_id="agent1", canary_accuracy=0.9)
    response = tcell.inspect(peptide)

    assert response.signal1 == Signal1.SELF
    assert response.signal2 == Signal2.MANUAL_FLAG
    # Because signal1 is SELF, no anomaly - all clear
    assert response.threat_level == ThreatLevel.NONE
    assert response.action == ResponseAction.IGNORE


def test_tcell_inspect_manual_flag_with_violation(baseline_profile):
    """Test inspect with manual flag AND violations."""
    tcell = TCell(profile=baseline_profile)
    tcell.flag_manually("User suspicion")

    # 1 violation (output length out of bounds)
    peptide = make_peptide(agent_id="agent1", output_length_mean=1000.0, canary_accuracy=0.9)
    response = tcell.inspect(peptide)

    assert response.signal1 == Signal1.NON_SELF
    assert response.signal2 == Signal2.MANUAL_FLAG
    # Both signals are present. Since violation_count < 3 and canary >= 0.5, it should be CONFIRMED/ISOLATE
    assert response.threat_level == ThreatLevel.CONFIRMED
    assert response.action == ResponseAction.ISOLATE


def test_tcell_inspect_canary_failure(baseline_profile):
    """Test inspect with canary failure and violations."""
    tcell = TCell(profile=baseline_profile)

    # Canary failure (< 0.8 min) and 1 violation (output length)
    peptide = make_peptide(agent_id="agent1", output_length_mean=1000.0, canary_accuracy=0.4)
    response = tcell.inspect(peptide)

    assert response.signal1 == Signal1.NON_SELF
    assert response.signal2 == Signal2.CANARY_FAILED
    # Both signals present, and canary_accuracy < 0.5 is CRITICAL
    assert response.threat_level == ThreatLevel.CRITICAL
    assert response.action == ResponseAction.SHUTDOWN


def test_tcell_inspect_repeated_anomaly(baseline_profile):
    """Test inspect triggers REPEATED_ANOMALY after consecutive violations."""
    tcell = TCell(profile=baseline_profile, repeated_anomaly_threshold=3)
    peptide = make_peptide(agent_id="agent1", output_length_mean=1000.0, canary_accuracy=0.9)

    # First violation
    resp1 = tcell.inspect(peptide)
    assert resp1.signal2 == Signal2.NONE
    assert tcell.anomaly_count == 1

    # Second violation
    resp2 = tcell.inspect(peptide)
    assert resp2.signal2 == Signal2.NONE
    assert tcell.anomaly_count == 2

    # Third violation
    resp3 = tcell.inspect(peptide)
    assert resp3.signal2 == Signal2.REPEATED_ANOMALY
    assert resp3.threat_level == ThreatLevel.CONFIRMED
    assert resp3.action == ResponseAction.ISOLATE
    assert tcell.anomaly_count == 3


def test_tcell_inspect_anomaly_reset_by_valid_peptide(baseline_profile):
    """Test anomaly count is reset when a valid peptide is seen."""
    tcell = TCell(profile=baseline_profile)
    bad_peptide = make_peptide(agent_id="agent1", output_length_mean=1000.0, canary_accuracy=0.9)
    good_peptide = make_peptide(agent_id="agent1", canary_accuracy=0.9)

    tcell.inspect(bad_peptide)
    assert tcell.anomaly_count == 1

    tcell.inspect(good_peptide)
    assert tcell.anomaly_count == 0


def test_tcell_determine_response_critical_violations(baseline_profile):
    """Test _determine_response with >= 3 violations triggers CRITICAL / SHUTDOWN."""
    tcell = TCell(profile=baseline_profile)
    peptide = make_peptide(agent_id="agent1")

    threat, action = tcell._determine_response(
        signal1=Signal1.NON_SELF,
        signal2=Signal2.CROSS_VALIDATED,
        violation_count=3,
        peptide=peptide
    )

    assert threat == ThreatLevel.CRITICAL
    assert action == ResponseAction.SHUTDOWN


def test_tcell_determine_response_critical_canary(baseline_profile):
    """Test _determine_response with canary_accuracy < 0.5 triggers CRITICAL / SHUTDOWN."""
    tcell = TCell(profile=baseline_profile)
    peptide = make_peptide(agent_id="agent1", canary_accuracy=0.4)

    threat, action = tcell._determine_response(
        signal1=Signal1.NON_SELF,
        signal2=Signal2.CANARY_FAILED,
        violation_count=1,
        peptide=peptide
    )

    assert threat == ThreatLevel.CRITICAL
    assert action == ResponseAction.SHUTDOWN


def test_tcell_reset(baseline_profile):
    """Test reset restores state, anomaly_count, and manual_flag."""
    tcell = TCell(profile=baseline_profile)
    tcell.anomaly_count = 5
    tcell.manual_flag = "Flagged"
    tcell.state.signal1 = Signal1.NON_SELF
    tcell.state.signal2 = Signal2.MANUAL_FLAG

    tcell.reset()

    assert tcell.anomaly_count == 0
    assert tcell.manual_flag is None
    assert tcell.state.signal1 == Signal1.SELF
    assert tcell.state.signal2 == Signal2.NONE


def test_tcell_reset_without_confirmation_increases_anergy(baseline_profile):
    """Test reset_without_confirmation tracks toward anergy when signal1=NON_SELF and signal2=NONE."""
    tcell = TCell(profile=baseline_profile)

    # State setup to trigger false alarm tracking
    tcell.state.signal1 = Signal1.NON_SELF
    tcell.state.signal2 = Signal2.NONE
    tcell.anomaly_count = 2

    tcell.reset_without_confirmation()

    assert tcell.anergy_count == 1
    assert tcell.anomaly_count == 0
    assert tcell.state.signal1 == Signal1.SELF
    assert tcell.state.signal2 == Signal2.NONE

    # State setup without false alarm (e.g., both signals were present)
    tcell.state.signal1 = Signal1.NON_SELF
    tcell.state.signal2 = Signal2.REPEATED_ANOMALY
    tcell.reset_without_confirmation()

    # anergy_count should not increase
    assert tcell.anergy_count == 1

