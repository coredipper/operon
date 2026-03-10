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
        base_url: The API base URL (e.g. "http://localhost:1234/v1" for LM Studio).
        model: The model name to use (e.g. "local-model" or as listed by the server).
        api_key: Optional API key. Many local servers don't require one.
    """
    base_url: str = ""
    model: str = ""
    api_key: str | None = None

    @property
    def name(self) -> str:
        return "openai-compatible"

    def is_available(self) -> bool:
        """Check if base_url is configured."""
        return bool(self.base_url)

    def _get_client(self):
        """Lazy-load the OpenAI client with custom base_url."""
        if self._client is None:
            if not self.is_available():
                raise ProviderUnavailableError(
                    "base_url is required for OpenAICompatibleProvider."
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "not-needed",
                )
            except ImportError:
                raise ProviderUnavailableError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client
