import pytest
from operon_ai import SimpleTool, Mitochondria
from operon_ai.providers import ToolSchema

def test_simple_tool_with_parameters_schema():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"}
        },
        "required": ["x", "y"]
    }

    tool = SimpleTool(
        name="add",
        description="Add two numbers",
        func=lambda x, y: x + y,
        parameters_schema=schema
    )

    assert tool.parameters_schema == schema
    assert tool.execute(x=2, y=3) == 5

def test_simple_tool_default_empty_schema():
    tool = SimpleTool(
        name="noop",
        description="Does nothing",
        func=lambda: None
    )
    assert tool.parameters_schema == {"type": "object", "properties": {}}

def test_mitochondria_export_tool_schemas():
    mito = Mitochondria(silent=True)

    mito.register_function(
        name="add",
        func=lambda x, y: x + y,
        description="Add two numbers",
        parameters_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "First number"},
                "y": {"type": "number", "description": "Second number"}
            },
            "required": ["x", "y"]
        }
    )
    mito.register_function(
        name="multiply",
        func=lambda x, y: x * y,
        description="Multiply two numbers",
        parameters_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            "required": ["x", "y"]
        }
    )

    schemas = mito.export_tool_schemas()

    assert len(schemas) == 2
    assert all(isinstance(s, ToolSchema) for s in schemas)

    add_schema = next(s for s in schemas if s.name == "add")
    assert add_schema.description == "Add two numbers"
    assert "x" in add_schema.parameters_schema["properties"]

def test_mitochondria_execute_tool_call():
    from operon_ai.providers import ToolCall, ToolResult

    mito = Mitochondria(silent=True)
    mito.register_function(
        name="add",
        func=lambda x, y: x + y,
        description="Add two numbers"
    )

    call = ToolCall(id="call_1", name="add", arguments={"x": 5, "y": 3})
    result = mito.execute_tool_call(call)

    assert isinstance(result, ToolResult)
    assert result.call_id == "call_1"
    assert result.success
    assert result.output == "8"

def test_mitochondria_execute_tool_call_failure():
    from operon_ai.providers import ToolCall, ToolResult

    mito = Mitochondria(silent=True)
    mito.register_function(
        name="divide",
        func=lambda x, y: x / y,
        description="Divide numbers"
    )

    call = ToolCall(id="call_2", name="divide", arguments={"x": 5, "y": 0})
    result = mito.execute_tool_call(call)

    assert not result.success
    assert "division by zero" in result.error.lower()

def test_mitochondria_execute_tool_call_unknown():
    from operon_ai.providers import ToolCall, ToolResult

    mito = Mitochondria(silent=True)

    call = ToolCall(id="call_3", name="unknown_tool", arguments={})
    result = mito.execute_tool_call(call)

    assert not result.success
    assert "unknown tool" in result.error.lower()
