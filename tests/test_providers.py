"""Tests for LLM providers."""

import pytest
from operon_ai.providers import LLMProvider, LLMResponse, ProviderConfig


class TestLLMProviderProtocol:
    """Test the provider protocol definition."""

    def test_llm_response_dataclass(self):
        """LLMResponse should hold completion data."""
        response = LLMResponse(
            content="Hello, world!",
            model="test-model",
            tokens_used=10,
            latency_ms=100.0,
        )
        assert response.content == "Hello, world!"
        assert response.model == "test-model"
        assert response.tokens_used == 10
        assert response.latency_ms == 100.0

    def test_provider_config_defaults(self):
        """ProviderConfig should have sensible defaults."""
        config = ProviderConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.timeout_seconds == 30.0
