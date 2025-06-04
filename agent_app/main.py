import sys
import os
from langchain_core.messages import HumanMessage
from agent_app.config import get_llm
from agent_app.langgraph_graph import build_graph
from agent_app.rag import DocumentQA
from agent_app.weather import CityWeather

# Fix path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    print("\n🌐 LangGraph Agent ready. Type your query or 'quit' to exit.\n")
    graph = build_graph()
    llm = get_llm()

    while True:
        user_input = input("📝 Your Query: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("👋 Exiting. Goodbye!")
            break

        try:
            # Step 1: Get tool-calling LLM response
            messages = [HumanMessage(content=user_input)]
            ai_msg = llm.bind_tools([DocumentQA, CityWeather]).invoke(messages)

            # Step 2: Feed into graph
            state = {"messages": [*messages, ai_msg]}
            response = graph.invoke(state)

            print("\n✅", response.get("messages")[-1].content, "\n")

        except Exception as e:
            print(f"⚠️ Error: {e}\n")
