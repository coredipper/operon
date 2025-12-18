import pytest
from unittest.mock import patch
import os

def test_openai_complete_with_tools_signature():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        from operon_ai.providers import OpenAIProvider
        provider = OpenAIProvider()
        assert hasattr(provider, 'complete_with_tools')

def test_anthropic_complete_with_tools_signature():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        from operon_ai.providers import AnthropicProvider
        provider = AnthropicProvider()
        assert hasattr(provider, 'complete_with_tools')
