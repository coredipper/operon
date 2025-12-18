#!/usr/bin/env python3
"""
Example 21: LLM Tool Use Integration
====================================

Demonstrates the Nucleus-Mitochondria integration where:
1. Mitochondria provides tool definitions (calculator, search, etc.)
2. Nucleus sends prompts with tool schemas to the LLM
3. LLM can request tool execution
4. Results flow back for final response

This is analogous to how the biological nucleus coordinates
with mitochondria for ATP production - here we produce
"computational energy" through tool execution.

Run with API key for real LLM:
    GEMINI_API_KEY=... python examples/21_llm_tool_use.py
    OPENAI_API_KEY=... python examples/21_llm_tool_use.py
    ANTHROPIC_API_KEY=... python examples/21_llm_tool_use.py

Or run without for mock demonstration:
    python examples/21_llm_tool_use.py
"""

import os
from operon_ai import (
    Nucleus,
    Mitochondria,
    SimpleTool,
    ProviderConfig,
)


def main():
    print("=" * 60)
    print("Example 21: LLM Tool Use Integration")
    print("=" * 60)

    # Create Mitochondria with tools
    mito = Mitochondria(silent=False)

    # Register a calculator tool that uses Mitochondria's safe AST evaluation
    def safe_calculate(expression: str) -> str:
        """Use Mitochondria's built-in safe math evaluation."""
        result = mito.metabolize(expression)
        if result.success and result.atp:
            return str(result.atp.value)
        return f"Error: {result.error}"

    mito.register_function(
        name="calculator",
        func=safe_calculate,
        description="Evaluate a mathematical expression safely and return the result",
        parameters_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    )

    # Register a weather tool (mock)
    def get_weather(city: str) -> str:
        weather_data = {
            "london": "Cloudy, 12C",
            "tokyo": "Sunny, 22C",
            "new york": "Rainy, 8C",
            "paris": "Partly cloudy, 15C",
        }
        return weather_data.get(city.lower(), f"Weather for {city}: Sunny, 20C")

    mito.register_function(
        name="get_weather",
        func=get_weather,
        description="Get the current weather for a city",
        parameters_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for"
                }
            },
            "required": ["city"]
        }
    )

    # Create Nucleus (auto-detects provider)
    nucleus = Nucleus()
    print(f"\nUsing provider: {nucleus.provider.name}")

    # Show available tools
    print("\nRegistered tools:")
    for schema in mito.export_tool_schemas():
        print(f"  - {schema.name}: {schema.description}")

    # Example 1: Calculator
    print("\n" + "-" * 40)
    print("Query 1: Math calculation")
    print("-" * 40)

    response = nucleus.transcribe_with_tools(
        "What is 15 * 7 + 23? Use the calculator tool to compute this.",
        mitochondria=mito,
        config=ProviderConfig(temperature=0.0),
    )
    print(f"Response: {response.content}")

    # Example 2: Weather
    print("\n" + "-" * 40)
    print("Query 2: Weather lookup")
    print("-" * 40)

    response = nucleus.transcribe_with_tools(
        "What's the weather like in Tokyo right now?",
        mitochondria=mito,
        config=ProviderConfig(temperature=0.0),
    )
    print(f"Response: {response.content}")

    # Example 3: Combined
    print("\n" + "-" * 40)
    print("Query 3: Multi-tool query")
    print("-" * 40)

    response = nucleus.transcribe_with_tools(
        "If the temperature in London is in Celsius, use the calculator to convert it to Fahrenheit (multiply by 9/5 and add 32).",
        mitochondria=mito,
        config=ProviderConfig(temperature=0.0),
        max_iterations=5,
    )
    print(f"Response: {response.content}")

    # Statistics
    print("\n" + "=" * 60)
    print("Session Statistics")
    print("=" * 60)
    print(f"Total energy consumed: {nucleus.get_total_energy_consumed()} ATP")
    print(f"Total tokens used: {nucleus.get_total_tokens_used()}")
    print(f"Mitochondria health: {mito.get_statistics()['health']}")
    print(f"Mitochondria operations: {mito.get_statistics()['operations_count']}")


if __name__ == "__main__":
    main()
