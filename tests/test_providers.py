"""Tests for LLM providers."""

import os
import pytest
from unittest.mock import patch, MagicMock
from operon_ai.providers import LLMProvider, LLMResponse, ProviderConfig, MockProvider


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


class TestMockProvider:
    """Test the mock provider for testing/fallback."""

    def test_mock_provider_name(self):
        """MockProvider should identify itself."""
        provider = MockProvider()
        assert provider.name == "mock"

    def test_mock_provider_is_always_available(self):
        """MockProvider should always be available."""
        provider = MockProvider()
        assert provider.is_available() is True

    def test_mock_provider_returns_response(self):
        """MockProvider should return a valid response."""
        provider = MockProvider()
        response = provider.complete("Hello")
        assert isinstance(response, LLMResponse)
        assert response.content != ""
        assert response.model == "mock-v1"

    def test_mock_provider_with_custom_responses(self):
        """MockProvider should use custom responses when provided."""
        responses = {"hello": "world", "foo": "bar"}
        provider = MockProvider(responses=responses)

        response = provider.complete("hello")
        assert response.content == "world"

        response = provider.complete("foo")
        assert response.content == "bar"

    def test_mock_provider_default_response(self):
        """MockProvider should use default for unknown prompts."""
        provider = MockProvider(default_response="I don't know")
        response = provider.complete("unknown prompt")
        assert response.content == "I don't know"


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_openai_provider_name(self):
        """OpenAIProvider should identify itself."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from operon_ai.providers import OpenAIProvider
            provider = OpenAIProvider()
            assert provider.name == "openai"

    def test_openai_provider_not_available_without_key(self):
        """OpenAIProvider should not be available without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove key if present
            os.environ.pop("OPENAI_API_KEY", None)
            from operon_ai.providers import OpenAIProvider
            provider = OpenAIProvider()
            assert provider.is_available() is False

    def test_openai_provider_available_with_key(self):
        """OpenAIProvider should be available with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from operon_ai.providers import OpenAIProvider
            provider = OpenAIProvider()
            assert provider.is_available() is True

    def test_openai_provider_uses_env_key(self):
        """OpenAIProvider should read key from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            from operon_ai.providers import OpenAIProvider
            provider = OpenAIProvider()
            assert provider._api_key == "sk-test123"

    def test_openai_provider_accepts_explicit_key(self):
        """OpenAIProvider should accept explicit API key."""
        from operon_ai.providers import OpenAIProvider
        provider = OpenAIProvider(api_key="sk-explicit")
        assert provider._api_key == "sk-explicit"
        assert provider.is_available() is True
