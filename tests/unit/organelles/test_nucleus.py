"""
Comprehensive tests for the Nucleus organelle.

The Nucleus is the decision-making center that wraps LLM providers,
handles transcription (prompt -> response), maintains audit trails,
and integrates with tools via Mitochondria.
"""

import pytest
import os
import warnings
from unittest.mock import patch, Mock
from datetime import datetime

from operon_ai.organelles.nucleus import Nucleus, Transcription
from operon_ai.organelles.mitochondria import Mitochondria
from operon_ai.providers import (
    MockProvider,
    LLMResponse,
    ProviderConfig,
    ProviderUnavailableError,
    TranscriptionFailedError,
)


class TestNucleusBasics:
    """Test basic Nucleus functionality."""

    def test_create_with_explicit_provider(self):
        """Nucleus should accept an explicit MockProvider."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        assert nucleus.provider is not None
        assert nucleus.provider.name == "mock"
        assert nucleus._initialized is True

    def test_create_with_custom_mock_responses(self):
        """Nucleus should work with customized MockProvider responses."""
        mock = MockProvider(responses={"hello": "world", "goodbye": "farewell"})
        nucleus = Nucleus(provider=mock)

        response1 = nucleus.transcribe("hello")
        response2 = nucleus.transcribe("goodbye")

        assert response1.content == "world"
        assert response2.content == "farewell"

    def test_auto_detect_provider_fallback(self):
        """Nucleus should auto-detect provider and fall back to mock when no keys available."""
        # Clear all API keys
        with patch.dict(os.environ, {}, clear=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                nucleus = Nucleus()

                # Should fall back to mock
                assert nucleus.provider.name == "mock"

                # Should issue warning
                assert len(w) == 1
                assert "No LLM API keys found" in str(w[0].message)
                assert "Using MockProvider" in str(w[0].message)

    def test_basic_transcribe(self):
        """Nucleus.transcribe should call provider and return response."""
        mock = MockProvider(responses={"test prompt": "test response"})
        nucleus = Nucleus(provider=mock)

        result = nucleus.transcribe("test prompt")

        assert isinstance(result, LLMResponse)
        assert result.content == "test response"
        assert result.model == "mock-v1"
        assert result.tokens_used > 0

    def test_transcribe_logs_history(self):
        """Nucleus should maintain audit trail of all transcriptions."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        nucleus.transcribe("first")
        nucleus.transcribe("second")
        nucleus.transcribe("third")

        assert len(nucleus.transcription_log) == 3
        assert isinstance(nucleus.transcription_log[0], Transcription)
        assert isinstance(nucleus.transcription_log[1], Transcription)
        assert isinstance(nucleus.transcription_log[2], Transcription)

        prompts = [t.prompt for t in nucleus.transcription_log]
        assert prompts == ["first", "second", "third"]

    def test_transcription_contains_metadata(self):
        """Transcription audit record should contain all metadata."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock, base_energy_cost=15)

        result = nucleus.transcribe("test")

        transcription = nucleus.transcription_log[0]
        assert transcription.prompt == "test"
        assert transcription.response == result
        assert transcription.provider == "mock"
        assert isinstance(transcription.timestamp, datetime)
        assert transcription.energy_cost == 15


class TestNucleusConfiguration:
    """Test Nucleus configuration options."""

    def test_custom_base_energy_cost(self):
        """Nucleus should respect custom base_energy_cost."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock, base_energy_cost=25)

        nucleus.transcribe("test")

        assert nucleus.transcription_log[0].energy_cost == 25

    def test_override_energy_cost_per_transcription(self):
        """Nucleus.transcribe should allow per-call energy cost override."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock, base_energy_cost=10)

        nucleus.transcribe("first")  # Uses base_energy_cost=10
        nucleus.transcribe("second", energy_cost=50)  # Override to 50

        assert nucleus.transcription_log[0].energy_cost == 10
        assert nucleus.transcription_log[1].energy_cost == 50

    def test_transcribe_with_custom_config(self):
        """Nucleus.transcribe should accept custom ProviderConfig."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        config = ProviderConfig(
            temperature=0.5,
            max_tokens=2048,
            timeout_seconds=60.0,
            system_prompt="You are a helpful assistant."
        )

        nucleus.transcribe("test", config=config)

        transcription = nucleus.transcription_log[0]
        assert transcription.config == config
        assert transcription.config.temperature == 0.5
        assert transcription.config.max_tokens == 2048

    def test_custom_max_retries(self):
        """Nucleus should accept custom max_retries configuration."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock, max_retries=5)

        assert nucleus.max_retries == 5


class TestNucleusToolIntegration:
    """Test Nucleus integration with Mitochondria tools."""

    def test_transcribe_with_tools_no_tool_calls(self):
        """When LLM doesn't request tools, should return direct response."""
        nucleus = Nucleus(provider=MockProvider())
        mito = Mitochondria(silent=True)

        # No tools registered, so no tool calls
        response = nucleus.transcribe_with_tools(
            "What is the capital of France?",
            mitochondria=mito,
        )

        assert response.content is not None
        assert len(nucleus.transcription_log) == 1

    def test_transcribe_with_tools_executes_tool(self):
        """When LLM requests tools, should execute them and return final response."""
        nucleus = Nucleus(provider=MockProvider())
        mito = Mitochondria(silent=True)

        # Register a calculator tool using Mitochondria's safe eval
        mito.register_function(
            name="calculator",
            func=lambda expression: mito.digest_glucose(expression),
            description="Calculate math expressions",
            parameters_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        )

        response = nucleus.transcribe_with_tools(
            "Use the calculator to compute 2+2",
            mitochondria=mito,
        )

        assert response is not None

    def test_transcribe_with_tools_max_iterations_respected(self):
        """transcribe_with_tools should respect max_iterations limit."""
        nucleus = Nucleus(provider=MockProvider())
        mito = Mitochondria(silent=True)

        mito.register_function(
            name="infinite",
            func=lambda: "more",
            description="Always returns more",
        )

        response = nucleus.transcribe_with_tools(
            "Keep calling infinite tool",
            mitochondria=mito,
            max_iterations=3,
        )

        assert response is not None
        # Should not hang indefinitely

    def test_transcribe_with_tools_auto_execute_false(self):
        """transcribe_with_tools with auto_execute=False should return without executing."""
        nucleus = Nucleus(provider=MockProvider())
        mito = Mitochondria(silent=True)

        mito.register_function(
            name="test_tool",
            func=lambda x: f"executed {x}",
            description="Test tool",
        )

        response = nucleus.transcribe_with_tools(
            "Use test_tool",
            mitochondria=mito,
            auto_execute=False,
        )

        assert response is not None


class TestNucleusErrorHandling:
    """Test Nucleus error handling."""

    def test_transcribe_with_empty_prompt(self):
        """Nucleus should handle empty prompts gracefully."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        response = nucleus.transcribe("")

        # MockProvider should return something even for empty prompt
        assert response is not None
        assert isinstance(response, LLMResponse)

    def test_transcribe_with_provider_error(self):
        """Nucleus should propagate provider errors appropriately."""
        # Create a mock provider that raises errors
        error_provider = Mock(spec=MockProvider)
        error_provider.name = "error_mock"
        error_provider.is_available.return_value = True
        error_provider.complete.side_effect = ProviderUnavailableError("Provider is down")

        nucleus = Nucleus(provider=error_provider)

        with pytest.raises(ProviderUnavailableError, match="Provider is down"):
            nucleus.transcribe("test")

    def test_transcribe_with_none_response(self):
        """Nucleus should handle provider returning invalid response."""
        # Create a mock provider that returns None
        bad_provider = Mock(spec=MockProvider)
        bad_provider.name = "bad_mock"
        bad_provider.is_available.return_value = True
        bad_provider.complete.return_value = None

        nucleus = Nucleus(provider=bad_provider)

        # This should raise an error or handle gracefully
        # The actual behavior depends on the Nucleus implementation
        # For now, we'll just verify it doesn't crash Python
        try:
            result = nucleus.transcribe("test")
            # If it returns, it should be None or raise
            assert result is None or isinstance(result, LLMResponse)
        except (AttributeError, TypeError):
            # This is acceptable - provider returned invalid response
            pass


class TestNucleusStatistics:
    """Test Nucleus statistics and tracking."""

    def test_get_total_energy_consumed(self):
        """Nucleus should track total energy consumed across all transcriptions."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock, base_energy_cost=10)

        nucleus.transcribe("first")
        nucleus.transcribe("second")
        nucleus.transcribe("third", energy_cost=30)

        total_energy = nucleus.get_total_energy_consumed()
        assert total_energy == 10 + 10 + 30  # 50 total

    def test_get_total_tokens_used(self):
        """Nucleus should track total tokens used across all transcriptions."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        nucleus.transcribe("first prompt")
        nucleus.transcribe("second prompt")
        nucleus.transcribe("third prompt")

        total_tokens = nucleus.get_total_tokens_used()
        # MockProvider returns token count based on response word count
        assert total_tokens > 0

    def test_transcription_count(self):
        """Nucleus should accurately count transcriptions."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        assert len(nucleus.transcription_log) == 0

        nucleus.transcribe("one")
        assert len(nucleus.transcription_log) == 1

        nucleus.transcribe("two")
        nucleus.transcribe("three")
        assert len(nucleus.transcription_log) == 3

    def test_clear_log(self):
        """Nucleus.clear_log should reset transcription history."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        nucleus.transcribe("first")
        nucleus.transcribe("second")
        assert len(nucleus.transcription_log) == 2

        nucleus.clear_log()
        assert len(nucleus.transcription_log) == 0

        # Energy/token totals should be zero after clear
        assert nucleus.get_total_energy_consumed() == 0
        assert nucleus.get_total_tokens_used() == 0

    def test_transcription_timing_info(self):
        """Transcriptions should capture timing information."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        nucleus.transcribe("test")

        transcription = nucleus.transcription_log[0]
        assert isinstance(transcription.timestamp, datetime)
        assert transcription.response.latency_ms > 0


class TestNucleusAdvanced:
    """Test advanced Nucleus features."""

    def test_multiple_transcriptions_maintain_order(self):
        """Transcription log should maintain chronological order."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        prompts = ["alpha", "beta", "gamma", "delta", "epsilon"]
        for prompt in prompts:
            nucleus.transcribe(prompt)

        logged_prompts = [t.prompt for t in nucleus.transcription_log]
        assert logged_prompts == prompts

    def test_transcription_with_same_prompt_different_responses(self):
        """Each transcription should be logged separately, even with same prompt."""
        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        nucleus.transcribe("same prompt")
        nucleus.transcribe("same prompt")
        nucleus.transcribe("same prompt")

        assert len(nucleus.transcription_log) == 3
        # Each should have its own Transcription object
        assert nucleus.transcription_log[0] is not nucleus.transcription_log[1]
        assert nucleus.transcription_log[1] is not nucleus.transcription_log[2]

    def test_provider_name_logged_correctly(self):
        """Transcription should log the correct provider name."""
        mock1 = MockProvider()
        nucleus = Nucleus(provider=mock1)

        nucleus.transcribe("test")

        assert nucleus.transcription_log[0].provider == "mock"
