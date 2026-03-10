"""Tests for OpenAICompatibleProvider."""

import pytest
from unittest.mock import patch, MagicMock
from operon_ai.providers import OpenAICompatibleProvider, LLMResponse
from operon_ai.providers.base import ProviderUnavailableError


class TestOpenAICompatibleProvider:
    """Test OpenAI-compatible provider implementation."""

    def test_name(self):
        provider = OpenAICompatibleProvider(
            api_key="not-needed", base_url="http://localhost:1234/v1", model="local-model"
        )
        assert provider.name == "openai-compatible"

    def test_is_available_with_base_url(self):
        provider = OpenAICompatibleProvider(
            api_key="not-needed", base_url="http://localhost:1234/v1", model="local-model"
        )
        assert provider.is_available() is True

    def test_model_and_api_key_required(self):
        # Both model and api_key are required (no defaults)
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            OpenAICompatibleProvider(base_url="http://localhost:1234/v1")

    def test_is_not_available_without_base_url(self):
        provider = OpenAICompatibleProvider(api_key="not-needed", base_url="", model="local-model")
        assert provider.is_available() is False

    def test_accepts_custom_api_key(self):
        provider = OpenAICompatibleProvider(
            api_key="sk-custom", base_url="http://localhost:1234/v1", model="local-model"
        )
        assert provider.api_key == "sk-custom"

    def test_get_client_raises_without_base_url(self):
        provider = OpenAICompatibleProvider(api_key="not-needed", base_url="", model="local-model")
        with pytest.raises(ProviderUnavailableError, match="base_url is required"):
            provider._get_client()

    def test_get_client_passes_api_key_as_is(self):
        provider = OpenAICompatibleProvider(
            api_key="not-needed", base_url="http://localhost:1234/v1", model="local-model"
        )
        mock_openai = MagicMock()
        with patch.dict(
            "sys.modules",
            {"openai": MagicMock(OpenAI=mock_openai)},
        ):
            provider._get_client()
            mock_openai.assert_called_once_with(
                base_url="http://localhost:1234/v1",
                api_key="not-needed",
            )

    def test_get_client_passes_custom_api_key(self):
        provider = OpenAICompatibleProvider(
            api_key="sk-custom", base_url="http://localhost:1234/v1", model="local-model"
        )
        mock_openai = MagicMock()
        with patch.dict(
            "sys.modules",
            {"openai": MagicMock(OpenAI=mock_openai)},
        ):
            provider._get_client()
            mock_openai.assert_called_once_with(
                base_url="http://localhost:1234/v1",
                api_key="sk-custom",
            )

    def test_complete_with_tools_signature(self):
        provider = OpenAICompatibleProvider(
            api_key="not-needed", base_url="http://localhost:1234/v1", model="local-model"
        )
        assert hasattr(provider, "complete_with_tools")

    def test_get_client_rejects_empty_api_key(self):
        provider = OpenAICompatibleProvider(
            api_key="", base_url="http://localhost:1234/v1", model="local-model"
        )
        with pytest.raises(ProviderUnavailableError, match="api_key must be a non-empty string"):
            provider._get_client()

    def test_get_client_passes_api_key_for_remote(self):
        provider = OpenAICompatibleProvider(
            api_key="sk-together", base_url="https://api.together.ai/v1", model="mistral-7b"
        )
        mock_openai = MagicMock()
        with patch.dict(
            "sys.modules",
            {"openai": MagicMock(OpenAI=mock_openai)},
        ):
            provider._get_client()
            mock_openai.assert_called_once_with(
                base_url="https://api.together.ai/v1",
                api_key="sk-together",
            )
