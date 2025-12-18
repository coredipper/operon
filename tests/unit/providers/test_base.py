import pytest
from operon_ai.providers import ToolSchema, ToolCall, ToolResult

def test_tool_schema_creation():
    schema = ToolSchema(
        name="calculator",
        description="Perform arithmetic",
        parameters_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    )
    assert schema.name == "calculator"
    assert "expression" in schema.parameters_schema["properties"]

def test_tool_call_creation():
    call = ToolCall(
        id="call_123",
        name="calculator",
        arguments={"expression": "2 + 2"}
    )
    assert call.id == "call_123"
    assert call.arguments["expression"] == "2 + 2"

def test_tool_result_creation():
    result = ToolResult(
        call_id="call_123",
        output="4",
        success=True
    )
    assert result.success
    assert result.output == "4"

def test_tool_result_failure():
    result = ToolResult(
        call_id="call_123",
        output=None,
        success=False,
        error="Division by zero"
    )
    assert not result.success
    assert result.error == "Division by zero"
