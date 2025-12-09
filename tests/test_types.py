"""Tests for core types: Signal, ActionProtein, FoldedProtein."""

import pytest
from operon.core.types import Signal, ActionProtein, FoldedProtein


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
            metadata={"priority": "high"}
        )
        assert signal.content == "Test message"
        assert signal.source == "System"
        assert signal.metadata == {"priority": "high"}

    def test_signal_metadata_is_mutable(self):
        """Signal metadata can be modified after creation."""
        signal = Signal(content="Test")
        signal.metadata["key"] = "value"
        assert signal.metadata["key"] == "value"


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
        for action_type in ["EXECUTE", "BLOCK", "PERMIT", "FAILURE", "UNKNOWN"]:
            protein = ActionProtein(
                action_type=action_type,
                payload="test",
                confidence=0.5
            )
            assert protein.action_type == action_type


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
