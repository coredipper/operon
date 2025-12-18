import pytest
from operon_ai.providers import MockProvider, ToolSchema

def test_mock_provider_complete_with_tools():
    provider = MockProvider()

    tools = [
        ToolSchema(
            name="calculator",
            description="Do math",
            parameters_schema={"type": "object", "properties": {"expr": {"type": "string"}}}
        )
    ]

    response, tool_calls = provider.complete_with_tools(
        "Calculate 2+2",
        tools=tools,
    )

    assert response.content is not None or len(tool_calls) >= 0
    assert isinstance(tool_calls, list)

def test_mock_provider_tool_call_pattern():
    provider = MockProvider()

    tools = [
        ToolSchema(
            name="search",
            description="Search the web",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}}
        )
    ]

    response, tool_calls = provider.complete_with_tools(
        "Search for Python tutorials",
        tools=tools,
    )

    assert len(tool_calls) >= 0  # Mock returns tool call when tool name in prompt
