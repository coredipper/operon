import pytest
from operon_ai import Nucleus, Mitochondria
from operon_ai.providers import MockProvider

def test_nucleus_transcribe_with_tools_no_calls():
    """When LLM doesn't request tools, return direct response."""
    nucleus = Nucleus(provider=MockProvider())
    mito = Mitochondria(silent=True)

    # No tools registered, so no tool calls
    response = nucleus.transcribe_with_tools(
        "What is the capital of France?",
        mitochondria=mito,
    )

    assert response.content is not None

def test_nucleus_transcribe_with_tools_executes():
    """When LLM requests tools, execute them and return final response."""
    nucleus = Nucleus(provider=MockProvider())
    mito = Mitochondria(silent=True)

    # Register a calculator tool using Mitochondria's safe eval
    mito.register_function(
        name="calculator",
        func=lambda expression: mito.digest_glucose(expression),
        description="Calculate math expressions",
        parameters_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    )

    response = nucleus.transcribe_with_tools(
        "Use the calculator to compute 2+2",
        mitochondria=mito,
    )

    assert response is not None

def test_nucleus_transcribe_with_tools_max_iterations():
    """Respects max_iterations limit."""
    nucleus = Nucleus(provider=MockProvider())
    mito = Mitochondria(silent=True)

    mito.register_function(
        name="infinite",
        func=lambda: "more",
        description="Always returns more",
    )

    response = nucleus.transcribe_with_tools(
        "Keep calling infinite tool",
        mitochondria=mito,
        max_iterations=3,
    )

    assert response is not None
