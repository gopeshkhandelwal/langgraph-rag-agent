import sys
import os
from langchain_core.messages import HumanMessage, ToolMessage
from config import get_llm
from langgraph_graph import build_graph
from rag import DocumentQA
from weather import CityWeather

# Fix module import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Configuration ===
MAX_MESSAGES = 6
history = []

if __name__ == "__main__":
    print("\n🌐 LangGraph Agent ready. Type your query or 'quit' to exit.\n")
    graph = build_graph()
    llm = get_llm()

    while True:
        user_input = input("📝 Your Query: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Exiting. Goodbye!")
            break

        try:
            # Step 1: Add user message
            history.append(HumanMessage(content=user_input))
            history = history[-MAX_MESSAGES:]

            # Step 2: LLM generates tool call (if any)
            ai_msg = llm.bind_tools([DocumentQA, CityWeather]).invoke(history)
            history.append(ai_msg)

            # Step 3: Process tool calls
            for call in ai_msg.tool_calls or []:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_id = call["id"]

                if tool_name == "DocumentQA":
                    result = DocumentQA.invoke(tool_args)
                elif tool_name == "CityWeather":
                    result = CityWeather.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                history.append(ToolMessage(tool_call_id=tool_id, content=result))

            # Step 4: Graph invocation
            state = {"messages": history}
            response = graph.invoke(state)

            # Step 5: Print the final AI response
            last_message = response.get("messages")[-1]
            print("\n✅", last_message.content, "\n")

        except Exception as e:
            print(f"⚠️ Error: {e}\n")
