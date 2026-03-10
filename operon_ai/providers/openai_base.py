"""
Shared base for OpenAI and OpenAI-compatible providers.

Contains the common completion and tool-calling logic used by both
OpenAIProvider and OpenAICompatibleProvider.
"""

import time
from dataclasses import dataclass, field

from .base import (
    LLMResponse,
    ProviderConfig,
    ProviderUnavailableError,
    QuotaExhaustedError,
    TranscriptionFailedError,
)


@dataclass
class OpenAIBaseProvider:
    """
    Base class with shared OpenAI-format completion logic.

    Subclasses must set `_client` via their own `_get_client()` and
    define `model`, `name`, and `is_available()`.
    """
    model: str = ""
    _client: object = field(init=False, repr=False, default=None)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def _get_client(self):
        raise NotImplementedError

    def _handle_error(self, e: Exception) -> None:
        """Categorize and re-raise exceptions."""
        error_str = str(e).lower()
        if "rate" in error_str or "quota" in error_str:
            raise QuotaExhaustedError(f"{self.name} rate limit: {e}")
        if "api key" in error_str or "authentication" in error_str:
            raise ProviderUnavailableError(f"{self.name} auth error: {e}")
        if "connection" in error_str or "refused" in error_str:
            raise ProviderUnavailableError(f"{self.name} connection error: {e}")
        raise TranscriptionFailedError(f"{self.name} error: {e}")

    def complete(
        self,
        prompt: str,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        """Send prompt to the OpenAI-format endpoint and get response."""
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

        except (ProviderUnavailableError, QuotaExhaustedError, TranscriptionFailedError):
            raise
        except Exception as e:
            self._handle_error(e)

    def complete_with_tools(
        self,
        prompt: str,
        tools: list["ToolSchema"],
        config: ProviderConfig | None = None,
    ) -> tuple[LLMResponse, list["ToolCall"]]:
        """Send prompt with tools to the OpenAI-format endpoint."""
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
        except (ProviderUnavailableError, QuotaExhaustedError, TranscriptionFailedError):
            raise
        except Exception as e:
            self._handle_error(e)
