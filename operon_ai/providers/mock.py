"""
Mock LLM Provider for testing and fallback.

When no API keys are available, the Nucleus falls back to this provider
which returns predefined or pattern-matched responses.
"""

import time
import re
from dataclasses import dataclass, field

from .base import LLMProvider, LLMResponse, ProviderConfig


@dataclass
class MockProvider:
    """
    Mock LLM provider for testing and graceful fallback.

    Provides deterministic responses for testing, or falls back
    to pattern-matched defaults when no real provider is available.
    """
    responses: dict[str, str] = field(default_factory=dict)
    default_response: str = "This is a mock response. Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM calls."
    latency_ms: float = 10.0  # Simulated latency

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        prompt: str,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        """Return mock response based on prompt matching."""
        start = time.perf_counter()

        # Check for exact match in responses dict
        prompt_lower = prompt.lower().strip()
        for key, value in self.responses.items():
            if key.lower() in prompt_lower:
                content = value
                break
        else:
            # Use pattern-based defaults for common cases
            content = self._pattern_response(prompt)

        # Simulate some latency
        elapsed = (time.perf_counter() - start) * 1000 + self.latency_ms

        return LLMResponse(
            content=content,
            model="mock-v1",
            tokens_used=len(content.split()),
            latency_ms=elapsed,
        )

    def _pattern_response(self, prompt: str) -> str:
        """Generate response based on prompt patterns."""
        prompt_lower = prompt.lower()

        # Code generation patterns
        if "write" in prompt_lower and ("function" in prompt_lower or "code" in prompt_lower):
            return "```python\ndef example():\n    return 'mock implementation'\n```"

        # Code review patterns
        if "review" in prompt_lower or "analyze" in prompt_lower:
            return "APPROVED: Code looks acceptable. No critical issues found."

        # Calculation patterns
        if match := re.search(r"calculate\s+(\d+)\s*\+\s*(\d+)", prompt_lower):
            result = int(match.group(1)) + int(match.group(2))
            return f"The result is {result}."

        # Safety check patterns
        if any(word in prompt_lower for word in ["safe", "dangerous", "risk"]):
            if any(word in prompt_lower for word in ["delete", "rm -rf", "drop table"]):
                return "UNSAFE: This operation could cause data loss."
            return "SAFE: This operation appears safe to proceed."

        return self.default_response
