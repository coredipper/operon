"""
OpenAI-Compatible LLM Provider.

Wraps any OpenAI-compatible API (LM Studio, Ollama, vLLM, Together AI, etc.)
for use with the Nucleus. Accepts a custom base_url and model name.
"""

from dataclasses import dataclass

from .base import ProviderUnavailableError
from .openai_base import OpenAIBaseProvider


@dataclass
class OpenAICompatibleProvider(OpenAIBaseProvider):
    """
    Provider for any OpenAI-compatible API endpoint.

    Works with LM Studio, Ollama, vLLM, Together AI, Groq, and other
    services that expose an OpenAI-compatible chat completions API.

    Args:
        api_key: Required API key (must be non-empty). For local servers that don't require
            authentication, pass any non-empty string like "not-needed" or "dummy".
        base_url: The API base URL (e.g. "http://localhost:1234/v1" for LM Studio).
    """
    api_key: str = ""
    base_url: str = ""

    @property
    def name(self) -> str:
        return "openai-compatible"

    def is_available(self) -> bool:
        """Check if required configuration is set."""
        if not self.model:
            return False
        if not self.base_url:
            return False
        if not self.api_key:
            return False
        return True

    def _get_client(self):
        """Lazy-load the OpenAI client with custom base_url."""
        if self._client is None:
            # Validate config before attempting the import so users get
            # clear configuration errors even when openai is not installed.
            if not self.model:
                raise ProviderUnavailableError(
                    "model is required for OpenAICompatibleProvider."
                )

            if not self.api_key:
                raise ProviderUnavailableError(
                    "api_key must be a non-empty string. For local servers that don't require authentication, pass any non-empty string like 'not-needed' or 'dummy'."
                )

            if not self.base_url:
                raise ProviderUnavailableError(
                    "base_url is required for OpenAICompatibleProvider."
                )

            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderUnavailableError(
                    "openai package not installed. Run: pip install openai"
                )

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client
