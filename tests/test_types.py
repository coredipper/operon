"""Tests for core types: Signal, ActionProtein, FoldedProtein, CellState, Pathway."""

import pytest
from operon_ai.core.types import (
    Signal,
    SignalType,
    SignalStrength,
    ActionProtein,
    ActionType,
    FoldedProtein,
    CellState,
    Pathway,
)


class TestSignal:
    """Tests for the Signal dataclass."""

    def test_signal_creation_with_content(self):
        """Signal can be created with just content."""
        signal = Signal(content="Hello, world!")
        assert signal.content == "Hello, world!"
        assert signal.source == "User"
        assert signal.metadata == {}

    def test_signal_creation_with_all_fields(self):
        """Signal can be created with all fields."""
        signal = Signal(
            content="Test message",
            source="System",
            signal_type=SignalType.INTERNAL,
            strength=SignalStrength.STRONG,
            metadata={"priority": "high"}
        )
        assert signal.content == "Test message"
        assert signal.source == "System"
        assert signal.signal_type == SignalType.INTERNAL
        assert signal.strength == SignalStrength.STRONG
        assert signal.metadata == {"priority": "high"}

    def test_signal_default_type_and_strength(self):
        """Signal defaults to EXTERNAL type and MODERATE strength."""
        signal = Signal(content="Test")
        assert signal.signal_type == SignalType.EXTERNAL
        assert signal.strength == SignalStrength.MODERATE

    def test_signal_with_metadata(self):
        """with_metadata() creates new signal with added metadata."""
        signal = Signal(content="Test", metadata={"key1": "value1"})
        new_signal = signal.with_metadata(key2="value2", key3="value3")

        # Original unchanged
        assert signal.metadata == {"key1": "value1"}
        # New signal has both
        assert new_signal.metadata == {"key1": "value1", "key2": "value2", "key3": "value3"}
        # Same content
        assert new_signal.content == signal.content

    def test_signal_amplify(self):
        """amplify() increases signal strength."""
        signal = Signal(content="Test", strength=SignalStrength.WEAK)

        amplified = signal.amplify()
        assert amplified.strength == SignalStrength.MODERATE

        amplified2 = amplified.amplify()
        assert amplified2.strength == SignalStrength.STRONG

        amplified3 = amplified2.amplify()
        assert amplified3.strength == SignalStrength.SATURATING

    def test_signal_amplify_at_max(self):
        """amplify() at SATURATING stays at SATURATING."""
        signal = Signal(content="Test", strength=SignalStrength.SATURATING)
        amplified = signal.amplify()
        assert amplified.strength == SignalStrength.SATURATING

    def test_signal_attenuate(self):
        """attenuate() decreases signal strength."""
        signal = Signal(content="Test", strength=SignalStrength.SATURATING)

        attenuated = signal.attenuate()
        assert attenuated.strength == SignalStrength.STRONG

        attenuated2 = attenuated.attenuate()
        assert attenuated2.strength == SignalStrength.MODERATE

        attenuated3 = attenuated2.attenuate()
        assert attenuated3.strength == SignalStrength.WEAK

    def test_signal_attenuate_at_min(self):
        """attenuate() at WEAK stays at WEAK."""
        signal = Signal(content="Test", strength=SignalStrength.WEAK)
        attenuated = signal.attenuate()
        assert attenuated.strength == SignalStrength.WEAK

    def test_signal_types(self):
        """All SignalType values are valid."""
        for signal_type in SignalType:
            signal = Signal(content="Test", signal_type=signal_type)
            assert signal.signal_type == signal_type

    def test_signal_timestamp(self):
        """Signal gets a timestamp on creation."""
        signal = Signal(content="Test")
        assert signal.timestamp is not None


class TestActionProtein:
    """Tests for the ActionProtein dataclass."""

    def test_action_protein_creation(self):
        """ActionProtein can be created with required fields."""
        protein = ActionProtein(
            action_type="EXECUTE",
            payload="Running task",
            confidence=0.95
        )
        assert protein.action_type == "EXECUTE"
        assert protein.payload == "Running task"
        assert protein.confidence == 0.95
        assert protein.metadata == {}

    def test_action_protein_with_metadata(self):
        """ActionProtein can include metadata."""
        protein = ActionProtein(
            action_type="BLOCK",
            payload="Blocked by safety check",
            confidence=1.0,
            metadata={"reason": "safety"}
        )
        assert protein.metadata["reason"] == "safety"

    def test_action_protein_types(self):
        """ActionProtein supports various action types."""
        for action_type in ["EXECUTE", "BLOCK", "PERMIT", "DEFER", "FAILURE", "UNKNOWN"]:
            protein = ActionProtein(
                action_type=action_type,
                payload="test",
                confidence=0.5
            )
            assert protein.action_type == action_type

    def test_action_protein_is_success(self):
        """is_success() returns True for successful action types."""
        success_types = ["EXECUTE", "PERMIT", "SUCCESS"]
        for action_type in success_types:
            protein = ActionProtein(action_type=action_type, payload="", confidence=1.0)
            assert protein.is_success() is True

        failure_types = ["BLOCK", "FAILURE", "UNKNOWN", "DEFER"]
        for action_type in failure_types:
            protein = ActionProtein(action_type=action_type, payload="", confidence=1.0)
            assert protein.is_success() is False

    def test_action_protein_is_blocking(self):
        """is_blocking() returns True for blocking action types."""
        blocking_types = ["BLOCK", "FAILURE"]
        for action_type in blocking_types:
            protein = ActionProtein(action_type=action_type, payload="", confidence=1.0)
            assert protein.is_blocking() is True

        non_blocking_types = ["EXECUTE", "PERMIT", "DEFER", "UNKNOWN"]
        for action_type in non_blocking_types:
            protein = ActionProtein(action_type=action_type, payload="", confidence=1.0)
            assert protein.is_blocking() is False

    def test_action_protein_with_confidence(self):
        """with_confidence() creates new protein with adjusted confidence."""
        protein = ActionProtein(action_type="EXECUTE", payload="test", confidence=0.5)

        new_protein = protein.with_confidence(0.9)

        assert protein.confidence == 0.5  # Original unchanged
        assert new_protein.confidence == 0.9
        assert new_protein.action_type == protein.action_type
        assert new_protein.payload == protein.payload

    def test_action_protein_timestamp(self):
        """ActionProtein gets a timestamp on creation."""
        protein = ActionProtein(action_type="EXECUTE", payload="", confidence=1.0)
        assert protein.timestamp is not None


class TestFoldedProtein:
    """Tests for the FoldedProtein generic dataclass."""

    def test_valid_folded_protein(self):
        """FoldedProtein can represent valid structured output."""
        protein = FoldedProtein(
            valid=True,
            structure={"command": "SELECT", "table": "users"},
            raw_peptide_chain='{"command": "SELECT", "table": "users"}'
        )
        assert protein.valid is True
        assert protein.structure["command"] == "SELECT"
        assert protein.error_trace is None

    def test_invalid_folded_protein(self):
        """FoldedProtein can represent failed validation."""
        protein = FoldedProtein(
            valid=False,
            raw_peptide_chain="invalid json {",
            error_trace="JSONDecodeError: Expecting property name"
        )
        assert protein.valid is False
        assert protein.structure is None
        assert "JSONDecodeError" in protein.error_trace

    def test_folded_protein_generic_type(self):
        """FoldedProtein works with different structure types."""
        # With dict
        dict_protein = FoldedProtein(valid=True, structure={"key": "value"})
        assert dict_protein.structure["key"] == "value"

        # With list
        list_protein = FoldedProtein(valid=True, structure=[1, 2, 3])
        assert list_protein.structure == [1, 2, 3]

        # With string
        str_protein = FoldedProtein(valid=True, structure="hello")
        assert str_protein.structure == "hello"

    def test_folded_protein_map_success(self):
        """map() applies function to valid protein."""
        protein = FoldedProtein(valid=True, structure=5)

        doubled = protein.map(lambda x: x * 2)

        assert doubled.valid is True
        assert doubled.structure == 10

    def test_folded_protein_map_invalid(self):
        """map() returns self for invalid protein."""
        protein = FoldedProtein(valid=False, error_trace="Error")

        result = protein.map(lambda x: x * 2)

        assert result.valid is False
        assert result is protein

    def test_folded_protein_map_error(self):
        """map() handles errors in the mapping function."""
        protein = FoldedProtein(valid=True, structure="not a number")

        result = protein.map(lambda x: int(x))  # Will fail

        assert result.valid is False
        assert result.error_trace is not None

    def test_folded_protein_folding_attempts(self):
        """FoldedProtein tracks folding attempts."""
        protein = FoldedProtein(valid=True, structure={}, folding_attempts=3)
        assert protein.folding_attempts == 3


class TestCellState:
    """Tests for the CellState dataclass."""

    def test_cell_state_creation(self):
        """CellState can be created with agent name."""
        state = CellState(agent_name="TestAgent")
        assert state.agent_name == "TestAgent"
        assert state.phase == "active"
        assert state.health_score == 1.0
        assert state.energy_level == 1.0

    def test_cell_state_full_creation(self):
        """CellState can be created with all fields."""
        state = CellState(
            agent_name="TestAgent",
            phase="senescent",
            health_score=0.5,
            energy_level=0.3,
            memory_count=10,
            operations_count=100,
            errors_count=5
        )
        assert state.phase == "senescent"
        assert state.health_score == 0.5
        assert state.energy_level == 0.3
        assert state.memory_count == 10

    def test_cell_state_is_healthy(self):
        """is_healthy() returns correct status based on health and energy."""
        # Healthy: good health and energy
        healthy = CellState(agent_name="Test", health_score=0.8, energy_level=0.5)
        assert healthy.is_healthy() is True

        # Unhealthy: low health
        low_health = CellState(agent_name="Test", health_score=0.3, energy_level=0.5)
        assert low_health.is_healthy() is False

        # Unhealthy: low energy
        low_energy = CellState(agent_name="Test", health_score=0.8, energy_level=0.05)
        assert low_energy.is_healthy() is False

    def test_cell_state_error_rate(self):
        """error_rate() calculates correct error ratio."""
        state = CellState(
            agent_name="Test",
            operations_count=100,
            errors_count=10
        )
        assert state.error_rate() == 0.1

    def test_cell_state_error_rate_zero_ops(self):
        """error_rate() returns 0 when no operations."""
        state = CellState(agent_name="Test", operations_count=0)
        assert state.error_rate() == 0.0

    def test_cell_state_timestamp(self):
        """CellState has a timestamp."""
        state = CellState(agent_name="Test")
        assert state.timestamp is not None


class TestPathway:
    """Tests for the Pathway dataclass."""

    def test_pathway_creation(self):
        """Pathway can be created with name."""
        pathway = Pathway(name="TestPathway")
        assert pathway.name == "TestPathway"
        assert pathway.stages == []
        assert pathway.current_stage == 0

    def test_pathway_with_stages(self):
        """Pathway can be created with stages."""
        pathway = Pathway(
            name="SignalPath",
            stages=["receptor", "transducer", "effector"]
        )
        assert len(pathway.stages) == 3
        assert pathway.stages[0] == "receptor"

    def test_pathway_advance(self):
        """advance() moves to next stage."""
        pathway = Pathway(
            name="Test",
            stages=["s1", "s2", "s3"]
        )

        assert pathway.current_stage == 0
        result = pathway.advance()
        assert result is True
        assert pathway.current_stage == 1

        result = pathway.advance()
        assert result is True
        assert pathway.current_stage == 2

    def test_pathway_advance_at_end(self):
        """advance() returns False when at last stage."""
        pathway = Pathway(
            name="Test",
            stages=["s1", "s2"]
        )
        pathway.current_stage = 1  # At last stage

        result = pathway.advance()
        assert result is False
        assert pathway.current_stage == 1  # Unchanged

    def test_pathway_current_stage_name(self):
        """current_stage_name() returns correct stage name."""
        pathway = Pathway(
            name="Test",
            stages=["receptor", "transducer", "effector"]
        )

        assert pathway.current_stage_name() == "receptor"

        pathway.advance()
        assert pathway.current_stage_name() == "transducer"

    def test_pathway_current_stage_name_empty(self):
        """current_stage_name() returns None for empty pathway."""
        pathway = Pathway(name="Empty", stages=[])
        assert pathway.current_stage_name() is None

    def test_pathway_is_complete(self):
        """is_complete() returns True when at last stage."""
        pathway = Pathway(
            name="Test",
            stages=["s1", "s2", "s3"]
        )

        assert pathway.is_complete() is False

        pathway.advance()  # s2
        assert pathway.is_complete() is False

        pathway.advance()  # s3 (last)
        assert pathway.is_complete() is True

    def test_pathway_is_complete_single_stage(self):
        """is_complete() works with single stage."""
        pathway = Pathway(name="Single", stages=["only"])
        assert pathway.is_complete() is True

    def test_pathway_tracks_signals(self):
        """Pathway can track signals."""
        pathway = Pathway(name="Test", stages=["s1"])
        signal = Signal(content="Test signal")

        pathway.signals.append(signal)
        assert len(pathway.signals) == 1
        assert pathway.signals[0].content == "Test signal"


class TestSignalStrength:
    """Tests for SignalStrength enum."""

    def test_signal_strength_values(self):
        """SignalStrength has correct numerical values."""
        assert SignalStrength.WEAK.value == 1
        assert SignalStrength.MODERATE.value == 2
        assert SignalStrength.STRONG.value == 3
        assert SignalStrength.SATURATING.value == 4

    def test_signal_strength_ordering(self):
        """SignalStrength values are ordered correctly."""
        strengths = list(SignalStrength)
        for i in range(len(strengths) - 1):
            assert strengths[i].value < strengths[i + 1].value


class TestSignalType:
    """Tests for SignalType enum."""

    def test_signal_type_values(self):
        """SignalType has expected values."""
        assert SignalType.EXTERNAL.value == "external"
        assert SignalType.INTERNAL.value == "internal"
        assert SignalType.PARACRINE.value == "paracrine"
        assert SignalType.ENDOCRINE.value == "endocrine"
        assert SignalType.AUTOCRINE.value == "autocrine"

    def test_all_signal_types_defined(self):
        """All expected signal types are defined."""
        types = {t.value for t in SignalType}
        expected = {"external", "internal", "paracrine", "endocrine", "autocrine"}
        assert types == expected
