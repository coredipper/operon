"""
Anthropic LLM Provider.

Wraps the Anthropic API (Claude 3, Claude 3.5, etc.) for use with the Nucleus.
"""

import os
import time
from dataclasses import dataclass, field

from .base import (
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    ProviderUnavailableError,
    QuotaExhaustedError,
    TranscriptionFailedError,
)


@dataclass
class AnthropicProvider:
    """
    Anthropic API provider for Claude models.

    Requires either ANTHROPIC_API_KEY environment variable or explicit api_key.
    """
    api_key: str | None = None
    model: str = "claude-sonnet-4-20250514"
    _client: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")

    @property
    def name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self._api_key)

    def _get_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            if not self.is_available():
                raise ProviderUnavailableError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
                )
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key)
            except ImportError:
                raise ProviderUnavailableError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def complete(
        self,
        prompt: str,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        """Send prompt to Anthropic and get response."""
        config = config or ProviderConfig()
        client = self._get_client()

        start = time.perf_counter()

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=config.max_tokens,
                system=config.system_prompt or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                timeout=config.timeout_seconds,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract text from response
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            tokens_used = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage else 0
            )

            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=tokens_used,
                latency_ms=elapsed_ms,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise QuotaExhaustedError(f"Anthropic rate limit: {e}")
            if "api key" in error_str or "authentication" in error_str:
                raise ProviderUnavailableError(f"Anthropic auth error: {e}")
            raise TranscriptionFailedError(f"Anthropic error: {e}")
