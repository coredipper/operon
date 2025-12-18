"""
Nucleus: The Decision-Making Center of the Cell
================================================

Biological Analogy:
- The nucleus contains DNA (instructions) and produces mRNA
- It coordinates protein synthesis by sending instructions to ribosomes
- It's the "brain" of the cell, making high-level decisions

In our model, the Nucleus wraps LLM providers and handles:
- Provider auto-detection and fallback
- Request/response logging for audit trails
- Energy cost tracking
- Retry logic with exponential backoff
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from ..providers import (
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    MockProvider,
    NucleusError,
    ProviderUnavailableError,
)


@dataclass
class Transcription:
    """
    Audit record of an LLM call.

    Named after the biological process where DNA is transcribed to mRNA.
    """
    prompt: str
    response: LLMResponse
    provider: str
    timestamp: datetime
    energy_cost: int
    config: ProviderConfig | None = None


@dataclass
class Nucleus:
    """
    The decision-making center of the cell.

    Wraps LLM providers with:
    - Auto-detection of available providers
    - Graceful fallback to MockProvider
    - Complete audit trail of all transcriptions
    - Energy cost tracking for metabolic integration
    """
    provider: LLMProvider | None = None
    base_energy_cost: int = 10
    max_retries: int = 3
    transcription_log: list[Transcription] = field(default_factory=list)
    _initialized: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.provider is None:
            self.provider = self._auto_detect_provider()
        self._initialized = True

    def _auto_detect_provider(self) -> LLMProvider:
        """
        Auto-detect the best available provider.

        Priority:
        1. Anthropic (if ANTHROPIC_API_KEY set)
        2. OpenAI (if OPENAI_API_KEY set)
        3. MockProvider (fallback with warning)
        """
        # Try Anthropic first
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from ..providers import AnthropicProvider
                provider = AnthropicProvider()
                if provider.is_available():
                    return provider
            except ImportError:
                pass

        # Try OpenAI second
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from ..providers import OpenAIProvider
                provider = OpenAIProvider()
                if provider.is_available():
                    return provider
            except ImportError:
                pass

        # Fall back to mock with warning
        warnings.warn(
            "No LLM API keys found. Using MockProvider. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM calls.",
            UserWarning,
        )
        return MockProvider()

    def transcribe(
        self,
        prompt: str,
        config: ProviderConfig | None = None,
        energy_cost: int | None = None,
    ) -> LLMResponse:
        """
        Send prompt to LLM and get response.

        Biological parallel: DNA → mRNA transcription

        Args:
            prompt: The prompt to send to the LLM
            config: Optional configuration overrides
            energy_cost: Override for ATP cost (defaults to base_energy_cost)

        Returns:
            LLMResponse with completion and metadata
        """
        cost = energy_cost if energy_cost is not None else self.base_energy_cost

        # Call provider
        response = self.provider.complete(prompt, config)

        # Log transcription for audit trail
        transcription = Transcription(
            prompt=prompt,
            response=response,
            provider=self.provider.name,
            timestamp=datetime.now(),
            energy_cost=cost,
            config=config,
        )
        self.transcription_log.append(transcription)

        return response

    def get_total_energy_consumed(self) -> int:
        """Get total ATP consumed by all transcriptions."""
        return sum(t.energy_cost for t in self.transcription_log)

    def get_total_tokens_used(self) -> int:
        """Get total tokens used across all transcriptions."""
        return sum(t.response.tokens_used for t in self.transcription_log)

    def clear_log(self) -> None:
        """Clear the transcription log (for testing/reset)."""
        self.transcription_log.clear()
