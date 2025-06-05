from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage
from langgraph.prebuilt.tool_node import ToolNode
from typing import Annotated, List, TypedDict
from rag import DocumentQA
from weather import CityWeather

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    output: str

def extract_output(state: AgentState) -> AgentState:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            return {**state, "output": msg.content}
    return {**state, "output": "⚠️ No response"}

def context_aware_router(state: AgentState) -> AgentState:
    # Check the last user message
    user_message = None
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            user_message = msg
            break

    # If the question is about "president of the USA"
    if user_message and "president of the usa" in user_message.content.lower():
        # Let the tools handle it (DocumentQA)
        return state

    # If the question is "Where was he born?" or similar, but no prior context
    if user_message and "where was he born" in user_message.content.lower():
        # Check if previous message contains an answer about a person
        found_context = False
        for msg in reversed(state["messages"]):
            if msg is user_message:
                continue
            if hasattr(msg, "content") and "president of the usa" in msg.content.lower():
                found_context = True
                break
        if not found_context:
            # No context, respond with clarification
            state = {**state, "messages": state["messages"] + [
                type(user_message)(
                    content="I'm sorry, but I can't provide the information you're looking for because your question is missing some context. Could you please provide more details?",
                    type="ai"
                )
            ]}
            return state
    return state
    # Example: Check if

def build_graph():
    builder = StateGraph(AgentState)

    tool_node = ToolNode([DocumentQA, CityWeather])

    # Use the context_aware_router instead of a dummy router
    builder.add_node("router", context_aware_router)
    builder.add_node("tools", tool_node)
    builder.add_node("extract_output", extract_output)

    builder.add_edge("router", "tools")
    builder.add_edge("tools", "extract_output")
    builder.add_edge("extract_output", END)

    builder.set_entry_point("router")
    return builder.compile()

