import pytest

def test_tool_types_exported():
    from operon_ai import ToolSchema, ToolCall, ToolResult
    assert ToolSchema is not None
    assert ToolCall is not None
    assert ToolResult is not None

def test_gemini_provider_exported():
    from operon_ai import GeminiProvider
    assert GeminiProvider is not None
