import pytest
from unittest.mock import MagicMock

from operon_ai.convergence.guarded_graph import compile_guarded_graph
from langchain_core.messages import AIMessage, HumanMessage

def test_guarded_graph_agent_node_exception():
    # Setup mock models
    fast_model = MagicMock()
    deep_model = MagicMock()

    # Configure the model to raise an exception on invoke
    error_msg = "Simulated API Error"
    fast_model.invoke.side_effect = Exception(error_msg)

    # Configure deep model to also raise an exception to ensure it always fails during escalation
    deep_model.invoke.side_effect = Exception(error_msg)

    # We need a minimal compiled dictionary representing an organism
    compiled_organism = {
        "organism_name": "test_org",
        "stages": [
            {
                "name": "stage1",
                "role": "test_role",
                "instructions": "test_instructions",
                "tools": []
            }
        ]
    }

    # Compile the graph using the mocks
    graph = compile_guarded_graph(
        compiled=compiled_organism,
        fast_model=fast_model,
        deep_model=deep_model
    )

    # Invoke the graph with a simple human message
    result = graph.invoke({
        "messages": [HumanMessage(content="Hello")],
        "use_deep": False
    })

    # Since fast_model and deep_model both raise exceptions, the node will return an
    # AIMessage containing the error text, and eventually set _pending_action to FAILURE.
    # The post_guard sees this FAILURE and the watcher logic will trigger retries and escalation,
    # and ultimately a halt due to the continuous failure, which LangGraph returns.

    # Verify that the final messages contain the expected error message
    assert "messages" in result

    error_messages = [
        msg for msg in result["messages"]
        if isinstance(msg, AIMessage) and "Error: Simulated API Error" in msg.content
    ]
    assert len(error_messages) > 0, "Expected to find an AIMessage with the exception error string"

    # Also verify that the watcher logged these failures
    assert "intervention_log" in result
    assert result["halted"] is True
