"""
Base types and protocol for LLM providers.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from datetime import datetime


# =============================================================================
# Exceptions
# =============================================================================

class NucleusError(Exception):
    """Base error for Nucleus/Provider operations."""
    pass


class ProviderUnavailableError(NucleusError):
    """No API key, network down, or provider unreachable."""
    pass


class QuotaExhaustedError(NucleusError):
    """API rate limit or budget exceeded."""
    pass


class TranscriptionFailedError(NucleusError):
    """LLM returned invalid or empty response."""
    pass


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for LLM provider behavior."""
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    system_prompt: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM completion."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: dict | None = None


# =============================================================================
# Protocol
# =============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """
    Abstract interface for any LLM backend.

    Implementations must provide:
    - complete(): Send prompt, get response
    - name: Provider identifier for logging
    - is_available(): Check if provider can be used
    """

    @property
    def name(self) -> str:
        """Provider name for logging/debugging."""
        ...

    def is_available(self) -> bool:
        """Check if provider is configured and reachable."""
        ...

    def complete(
        self,
        prompt: str,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        """
        Send prompt to LLM and get response.

        Args:
            prompt: The user/assistant prompt to complete
            config: Optional configuration overrides

        Returns:
            LLMResponse with completion and metadata

        Raises:
            ProviderUnavailableError: If provider cannot be reached
            QuotaExhaustedError: If rate limited or out of budget
            TranscriptionFailedError: If response is invalid
        """
        ...
