import pytest
import os
from unittest.mock import patch, MagicMock

def test_gemini_provider_name():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        from operon_ai.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider()
        assert provider.name == "gemini"

def test_gemini_provider_not_available_without_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GEMINI_API_KEY", None)
        from operon_ai.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider()
        assert not provider.is_available()

def test_gemini_provider_available_with_key():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        from operon_ai.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider()
        assert provider.is_available()

def test_gemini_provider_default_model():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        from operon_ai.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider()
        assert provider.model == "gemini-2.0-flash"
