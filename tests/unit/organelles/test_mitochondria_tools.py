import pytest
from operon_ai import SimpleTool, Mitochondria

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
