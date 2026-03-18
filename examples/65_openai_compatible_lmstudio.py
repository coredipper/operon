"""
Example 65: OpenAI-Compatible Provider with LM Studio
======================================================

Demonstrates using a local LLM (via LM Studio) through the
OpenAICompatibleProvider. This works with any OpenAI-compatible
server: LM Studio, Ollama, vLLM, Together AI, Groq, etc.

Prerequisites:
- LM Studio running locally with the model loaded
- openai package installed: pip install openai

Usage:
    python examples/65_openai_compatible_lmstudio.py
"""

from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import OpenAICompatibleProvider, ProviderConfig


def main():
    # Point to your local LM Studio server
    provider = OpenAICompatibleProvider(
        api_key="not-needed",
        base_url="http://localhost:1234/v1",
        model="qwen/qwen3.5-35b-a3b",
    )

    print(f"Provider: {provider.name}")
    print(f"Model:    {provider.model}")
    print(f"Endpoint: {provider.base_url}")
    print()

    # Use it directly
    config = ProviderConfig(
        system_prompt="You are a helpful assistant. Be concise.",
        temperature=0.7,
        max_tokens=256,
    )

    response = provider.complete("What is category theory in one paragraph?", config)
    print(f"Response: {response.content}")
    print(f"Tokens:   {response.tokens_used}")
    print(f"Latency:  {response.latency_ms:.0f}ms")
    print()

    # Or use it through Nucleus for audit trails and energy tracking
    nucleus = Nucleus(provider=provider, base_energy_cost=10)
    response = nucleus.transcribe("Give me a Python one-liner to flatten a list.", config)
    print(f"Nucleus response: {response.content}")
    print(f"Energy consumed:  {nucleus.get_total_energy_consumed()} ATP")


if __name__ == "__main__":
    main()
