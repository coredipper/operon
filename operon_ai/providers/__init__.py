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
    NucleusError,
    ProviderUnavailableError,
    QuotaExhaustedError,
    TranscriptionFailedError,
)
from .mock import MockProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderConfig",
    "NucleusError",
    "ProviderUnavailableError",
    "QuotaExhaustedError",
    "TranscriptionFailedError",
    "MockProvider",
]
