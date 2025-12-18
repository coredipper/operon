"""
LLM Providers: Pluggable backends for the Nucleus organelle.
============================================================

Provides a Protocol-based abstraction for LLM services, allowing
the Nucleus to work with any compatible backend (OpenAI, Anthropic, etc.)
"""

from .base import (
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    ToolSchema,
    ToolCall,
    ToolResult,
    NucleusError,
    ProviderUnavailableError,
    QuotaExhaustedError,
    TranscriptionFailedError,
)
from .mock import MockProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderConfig",
    "ToolSchema",
    "ToolCall",
    "ToolResult",
    "NucleusError",
    "ProviderUnavailableError",
    "QuotaExhaustedError",
    "TranscriptionFailedError",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
]
