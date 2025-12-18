"""
OpenAI LLM Provider.

Wraps the OpenAI API (GPT-4, GPT-3.5-turbo, etc.) for use with the Nucleus.
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
class OpenAIProvider:
    """
    OpenAI API provider for GPT models.

    Requires either OPENAI_API_KEY environment variable or explicit api_key.
    """
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    _client: object = field(init=False, repr=False, default=None)

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

    def complete(
        self,
        prompt: str,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        """Send prompt to OpenAI and get response."""
        config = config or ProviderConfig()
        client = self._get_client()

        start = time.perf_counter()

        try:
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            choice = response.choices[0]
            content = choice.message.content or ""

            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=elapsed_ms,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise QuotaExhaustedError(f"OpenAI rate limit: {e}")
            if "api key" in error_str or "authentication" in error_str:
                raise ProviderUnavailableError(f"OpenAI auth error: {e}")
            raise TranscriptionFailedError(f"OpenAI error: {e}")

    def complete_with_tools(
        self,
        prompt: str,
        tools: list["ToolSchema"],
        config: ProviderConfig | None = None,
    ) -> tuple[LLMResponse, list["ToolCall"]]:
        """Send prompt with tools to OpenAI."""
        if not self.is_available():
            raise ProviderUnavailableError("OPENAI_API_KEY not set")

        from .base import ToolSchema, ToolCall

        config = config or ProviderConfig()
        client = self._get_client()
        start = time.perf_counter()

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                }
            }
            for t in tools
        ]

        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                tools=openai_tools if openai_tools else None,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            choice = response.choices[0]
            content = choice.message.content or ""

            tool_calls = []
            if choice.message.tool_calls:
                import json
                for tc in choice.message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    ))

            return (
                LLMResponse(
                    content=content,
                    model=response.model,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                    latency_ms=elapsed_ms,
                    raw_response=response.model_dump(),
                ),
                tool_calls,
            )
        except Exception as e:
            raise TranscriptionFailedError(f"OpenAI error: {e}")
