"""Tests for OpenAICompatibleProvider."""

import pytest
from unittest.mock import patch, MagicMock
from operon_ai.providers import OpenAICompatibleProvider, LLMResponse
from operon_ai.providers.base import ProviderUnavailableError


class TestOpenAICompatibleProvider:
    """Test OpenAI-compatible provider implementation."""

    def test_name(self):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1", model="local-model"
        )
        assert provider.name == "openai-compatible"

    def test_is_available_with_base_url(self):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1", model="local-model"
        )
        assert provider.is_available() is True

    def test_is_not_available_without_base_url(self):
        provider = OpenAICompatibleProvider(base_url="", model="local-model")
        assert provider.is_available() is False

    def test_api_key_is_optional(self):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1", model="local-model"
        )
        assert provider.api_key is None
        assert provider.is_available() is True

    def test_accepts_explicit_api_key(self):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model="local-model",
            api_key="sk-custom",
        )
        assert provider.api_key == "sk-custom"

    def test_get_client_raises_without_base_url(self):
        provider = OpenAICompatibleProvider(base_url="", model="local-model")
        with pytest.raises(ProviderUnavailableError, match="base_url is required"):
            provider._get_client()

    def test_get_client_passes_base_url(self):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1", model="local-model"
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

    def test_get_client_passes_api_key_when_provided(self):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model="local-model",
            api_key="sk-custom",
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
            base_url="http://localhost:1234/v1", model="local-model"
        )
        assert hasattr(provider, "complete_with_tools")
