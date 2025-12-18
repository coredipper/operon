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
