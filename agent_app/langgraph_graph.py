from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
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
    """
    Generic router: routes to DocumentQA only for ComputePool queries,
    and to CityWeather for weather-related queries. Handles ambiguous follow-ups.
    """
    DOCQA_KEYWORDS = [
        "computepool", "compute pool", "nodepool", "node pool", "intel internal"
    ]
    WEATHER_KEYWORDS = [
        "weather", "temperature", "forecast", "humidity", "rain", "climate"
    ]
    AMBIGUOUS_FOLLOWUPS = [
        "where was it launched", "who can use it", "what is it", "where is it available"
    ]

    user_message = None
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            user_message = msg
            break

    if not user_message or not hasattr(user_message, "content"):
        return state

    user_content = user_message.content.lower()

    # Route to DocumentQA only for Intel Tiber Developer Cloud topics
    if any(keyword in user_content for keyword in DOCQA_KEYWORDS):
        return state

    # Route to CityWeather for weather-related queries
    if any(keyword in user_content for keyword in WEATHER_KEYWORDS):
        return state

    # Handle ambiguous follow-up questions lacking context
    if any(phrase in user_content for phrase in AMBIGUOUS_FOLLOWUPS):
        found_context = False
        for msg in reversed(state["messages"]):
            if msg is user_message:
                continue
            if hasattr(msg, "content") and any(
                keyword in msg.content.lower() for keyword in DOCQA_KEYWORDS
            ):
                found_context = True
                break
        if not found_context:
            clarification = (
                "I'm sorry, but I can't provide the information you're looking for because your question is missing some context. "
                "Could you please provide more details about Intel Tiber Developer Cloud?"
            )
            state = {
                **state,
                "messages": state["messages"] + [
                    type(user_message)(content=clarification, type="ai")
                ]
            }
            return state

    return state

def build_graph():
    builder = StateGraph(AgentState)

    tool_node = ToolNode([DocumentQA, CityWeather])

    builder.add_node("router", context_aware_router)
    builder.add_node("tools", tool_node)
    builder.add_node("extract_output", extract_output)

    builder.add_edge("router", "tools")
    builder.add_edge("tools", "extract_output")
    builder.add_edge("extract_output", END)

    builder.set_entry_point("router")
    return builder.compile()