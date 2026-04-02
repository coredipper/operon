"""
OpenAI LLM Provider.

Wraps the OpenAI API (GPT-4, GPT-3.5-turbo, etc.) for use with the Nucleus.
"""

import os
from dataclasses import dataclass

from .base import ProviderUnavailableError
from .openai_base import OpenAIBaseProvider


@dataclass
class OpenAIProvider(OpenAIBaseProvider):
    """
    OpenAI API provider for GPT models.

    Requires either OPENAI_API_KEY environment variable or explicit api_key.
    """
    api_key: str | None = None
    model: str = "gpt-5.4-mini"

    def __post_init__(self):
        self._api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self._api_key)

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            if not self.is_available():
                raise ProviderUnavailableError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ProviderUnavailableError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client
