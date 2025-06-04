from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from agent_app.rag import DocumentQA
from agent_app.weather import CityWeather

# ✅ Define your state schema
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def build_graph():
    builder = StateGraph(AgentState)  # ✅ pass the state schema

    # Define ToolNode
    tool_node = ToolNode([DocumentQA, CityWeather])

    # Add nodes
    builder.add_node("tools", tool_node)
    builder.add_node("router", lambda state: state)

    # Add conditional logic
    builder.add_conditional_edges("router", tools_condition)

    # Edges
    builder.add_edge("tools", END)

    # Entry point
    builder.set_entry_point("router")

    return builder.compile()
