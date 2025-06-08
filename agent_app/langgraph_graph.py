from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.prebuilt.tool_node import ToolNode
from config import get_llm
from rag import DocumentQA
from weather import CityWeather
import logging


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    output: str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)
llm = get_llm()
tools = [DocumentQA, CityWeather]
llm_with_tools = llm.bind_tools(tools)


def router(state: AgentState) -> AgentState:
    """LLM generates tool calls, actual tool run handled by ToolNode."""
    system_message = SystemMessage(
        content="You are a helpful assistant. Use the CityWeather tool for weather questions and DocumentQA for document questions. Do not just repeat the user's question."
    )
    logger.info("Iside router")
    logger.debug(f"talk_to_llm State: {state}")
    messages = [system_message] + state["messages"]
    ai_msg = llm_with_tools.invoke(messages)
    logger.debug(f"talk_to_llm Response: {ai_msg.content}")  # Debug print
    return {
                **state,
                "messages": state["messages"] + [ai_msg],
                "next": "extract_output"
            }

def extract_output(state: AgentState) -> AgentState:
    """Extracts final AI/tool message to populate output and update state."""
    logger.info("Iside extract_output")
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            return {
                **state,
                "output": msg.content,
                "messages": state["messages"] + [AIMessage(content=msg.content)]
            }
    return {
        **state,
        "output": "⚠️ No response",
        "messages": state["messages"] + [AIMessage(content="⚠️ No response")]
    }

    
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router)
    builder.add_node("tools", ToolNode([DocumentQA, CityWeather]))
    builder.add_node("extract_output", extract_output)

    builder.add_edge("router", "tools")
    builder.add_edge("tools", "extract_output")
    builder.add_edge("extract_output", END)

    builder.set_entry_point("router")
    return builder.compile()
