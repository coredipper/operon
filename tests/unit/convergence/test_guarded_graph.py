import pytest

# Check for both LangGraph and LangChain OpenAI so that this test is isolated appropriately
pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from operon_ai.convergence.guarded_graph import compile_guarded_graph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

class ErrorFakeChatModel(FakeListChatModel):
    def invoke(self, *args, **kwargs):
        raise Exception("Simulated API Error")

def test_guarded_graph_agent_node_exception():
    fake_model = ErrorFakeChatModel(responses=["dummy"])

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

    # We use compile_guarded_graph, but to test just the agent_node failure directly,
    # we can use the inner make_agent if we extract it, or since it creates a state graph,
    # we can compile the graph and invoke the specific node. Let's invoke the whole graph but
    # carefully verify the target behavior (which is that an AIMessage containing the error is produced).
    graph = compile_guarded_graph(
        compiled=compiled_organism,
        fast_model=fake_model,
        deep_model=fake_model
    )

    result = graph.invoke({
        "messages": [HumanMessage(content="Hello")],
        "use_deep": False
    })

    # Verify that the final messages contain the expected error message
    assert "messages" in result

    error_messages = [
        msg for msg in result["messages"]
        if isinstance(msg, AIMessage) and "Error: Simulated API Error" in msg.content
    ]
    assert len(error_messages) > 0, "Expected to find an AIMessage with the exception error string"
