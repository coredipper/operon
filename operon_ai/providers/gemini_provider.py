"""
Gemini LLM Provider using google-genai SDK.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime

from .base import (
    LLMResponse,
    ProviderConfig,
    ProviderUnavailableError,
    TranscriptionFailedError,
    ToolSchema,
    ToolCall,
)


@dataclass
class GeminiProvider:
    """
    Google Gemini provider using the google-genai SDK.

    Requires:
        pip install google-genai
        GEMINI_API_KEY environment variable
    """
    model: str = "gemini-flash-latest"
    _client: object | None = None

    def __post_init__(self):
        self._api_key = os.environ.get("GEMINI_API_KEY")
        if self._api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return self._api_key is not None

    def complete(
        self,
        prompt: str,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        """Send prompt to Gemini and get response."""
        if not self.is_available():
            raise ProviderUnavailableError("GEMINI_API_KEY not set")

        if self._client is None:
            raise ProviderUnavailableError(
                "google-genai not installed. Run: pip install google-genai"
            )

        config = config or ProviderConfig()
        start_time = time.time()

        try:
            from google.genai import types

            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
            if config.system_prompt:
                gen_config.system_instruction = config.system_prompt

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_config,
            )

            latency = (time.time() - start_time) * 1000
            content = response.text if response.text else ""
            tokens = len(content.split()) + len(prompt.split())

            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens,
                latency_ms=latency,
                timestamp=datetime.now(),
                raw_response={"candidates": [{"content": content}]},
            )

        except Exception as e:
            raise TranscriptionFailedError(f"Gemini API error: {e}")

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[ToolSchema],
        config: ProviderConfig | None = None,
    ) -> tuple[LLMResponse, list[ToolCall]]:
        """Send prompt with tool definitions."""
        if not self.is_available():
            raise ProviderUnavailableError("GEMINI_API_KEY not set")

        if self._client is None:
            raise ProviderUnavailableError(
                "google-genai not installed. Run: pip install google-genai"
            )

        config = config or ProviderConfig()
        start_time = time.time()

        try:
            from google.genai import types

            function_declarations = []
            for tool in tools:
                func_decl = types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters_schema,
                )
                function_declarations.append(func_decl)

            gemini_tools = [types.Tool(function_declarations=function_declarations)]

            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                tools=gemini_tools,
            )
            if config.system_prompt:
                gen_config.system_instruction = config.system_prompt

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_config,
            )

            latency = (time.time() - start_time) * 1000

            tool_calls = []
            content = ""

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for i, part in enumerate(candidate.content.parts):
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            tool_calls.append(ToolCall(
                                id=f"call_{i}",
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                            ))
                        elif hasattr(part, 'text') and part.text:
                            content += part.text

            tokens = len(content.split()) + len(prompt.split())

            return (
                LLMResponse(
                    content=content,
                    model=self.model,
                    tokens_used=tokens,
                    latency_ms=latency,
                    timestamp=datetime.now(),
                    raw_response={"response": str(response)},
                ),
                tool_calls,
            )

        except Exception as e:
            raise TranscriptionFailedError(f"Gemini API error: {e}")
