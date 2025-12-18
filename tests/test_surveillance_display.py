"""Tests for MHC Display fingerprint collector."""
import pytest
from datetime import datetime, timedelta
from operon_ai.surveillance.types import MHCPeptide
from operon_ai.surveillance.display import MHCDisplay, Observation


class TestObservation:
    def test_create_observation(self):
        obs = Observation(
            output="hello world",
            response_time=0.5,
            confidence=0.9,
            error=None,
        )
        assert obs.output == "hello world"
        assert obs.response_time == 0.5
        assert obs.confidence == 0.9
        assert obs.error is None

    def test_observation_with_error(self):
        obs = Observation(
            output=None,
            response_time=1.0,
            confidence=0.0,
            error="timeout",
        )
        assert obs.error == "timeout"


class TestMHCDisplay:
    def test_create_display(self):
        display = MHCDisplay(agent_id="test_agent", window_size=10)
        assert display.agent_id == "test_agent"
        assert display.window_size == 10
        assert len(display.observations) == 0

    def test_record_observation(self):
        display = MHCDisplay(agent_id="test", window_size=10)
        display.record(
            output="hello",
            response_time=0.5,
            confidence=0.9,
        )
        assert len(display.observations) == 1

    def test_window_size_enforced(self):
        display = MHCDisplay(agent_id="test", window_size=3)
        display.record(output="a", response_time=0.1, confidence=0.9)
        display.record(output="b", response_time=0.2, confidence=0.8)
        display.record(output="c", response_time=0.3, confidence=0.7)
        display.record(output="d", response_time=0.4, confidence=0.6)
        # Oldest should be dropped
        assert len(display.observations) == 3
        assert display.observations[0].output == "b"  # 'a' was dropped

    def test_generate_peptide_insufficient_data(self):
        display = MHCDisplay(agent_id="test", window_size=10, min_observations=5)
        display.record(output="a", response_time=0.1, confidence=0.9)
        display.record(output="b", response_time=0.2, confidence=0.8)
        peptide = display.generate_peptide()
        assert peptide is None  # Not enough observations

    def test_generate_peptide_success(self):
        display = MHCDisplay(agent_id="test", window_size=10, min_observations=3)
        display.record(output="hello", response_time=0.5, confidence=0.9)
        display.record(output="world", response_time=0.6, confidence=0.8)
        display.record(output="test!", response_time=0.4, confidence=0.85)

        peptide = display.generate_peptide()
        assert peptide is not None
        assert peptide.agent_id == "test"
        assert peptide.output_length_mean == 5.0  # len("hello")=5, len("world")=5, len("test!")=5
        assert peptide.response_time_mean == pytest.approx(0.5, rel=0.01)
        assert peptide.confidence_mean == pytest.approx(0.85, rel=0.01)

    def test_generate_peptide_tracks_errors(self):
        display = MHCDisplay(agent_id="test", window_size=10, min_observations=3)
        display.record(output="ok", response_time=0.5, confidence=0.9)
        display.record(output=None, response_time=1.0, confidence=0.0, error="timeout")
        display.record(output="ok", response_time=0.5, confidence=0.9)
        display.record(output=None, response_time=0.5, confidence=0.0, error="parse_error")

        peptide = display.generate_peptide()
        assert peptide is not None
        assert peptide.error_rate == 0.5  # 2 errors out of 4
        assert "timeout" in peptide.error_types
        assert "parse_error" in peptide.error_types

    def test_vocabulary_hash_consistent(self):
        display = MHCDisplay(agent_id="test", window_size=10, min_observations=2)
        display.record(output="hello world", response_time=0.5, confidence=0.9)
        display.record(output="hello there", response_time=0.5, confidence=0.9)
        peptide1 = display.generate_peptide()

        # Same vocabulary should give same hash
        display2 = MHCDisplay(agent_id="test", window_size=10, min_observations=2)
        display2.record(output="there hello", response_time=0.3, confidence=0.8)
        display2.record(output="world hello", response_time=0.6, confidence=0.7)
        peptide2 = display2.generate_peptide()

        assert peptide1.vocabulary_hash == peptide2.vocabulary_hash  # Same words

    def test_structure_hash_detects_format(self):
        display = MHCDisplay(agent_id="test", window_size=10, min_observations=2)
        display.record(output='{"key": "value"}', response_time=0.5, confidence=0.9)
        display.record(output='{"other": 123}', response_time=0.5, confidence=0.9)
        peptide_json = display.generate_peptide()

        display2 = MHCDisplay(agent_id="test", window_size=10, min_observations=2)
        display2.record(output="plain text here", response_time=0.5, confidence=0.9)
        display2.record(output="more plain text", response_time=0.5, confidence=0.9)
        peptide_text = display2.generate_peptide()

        # Different structures should have different hashes
        assert peptide_json.structure_hash != peptide_text.structure_hash

    def test_record_canary_result(self):
        display = MHCDisplay(agent_id="test", window_size=10, min_observations=2)
        display.record(output="a", response_time=0.5, confidence=0.9)
        display.record(output="b", response_time=0.5, confidence=0.9)

        display.record_canary_result(passed=True)
        display.record_canary_result(passed=True)
        display.record_canary_result(passed=False)

        peptide = display.generate_peptide()
        assert peptide.canary_accuracy == pytest.approx(2/3, rel=0.01)

    def test_clear_observations(self):
        display = MHCDisplay(agent_id="test", window_size=10)
        display.record(output="a", response_time=0.5, confidence=0.9)
        display.record(output="b", response_time=0.5, confidence=0.9)

        display.clear()
        assert len(display.observations) == 0
        assert len(display.canary_results) == 0
