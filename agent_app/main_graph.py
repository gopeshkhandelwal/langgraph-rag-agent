from agent_app.langgraph_graph import build_graph

if __name__ == "__main__":
    print("\n🌐 LangGraph Agent ready. Type your query or 'quit' to exit.\n")
    graph = build_graph()

    while True:
        user_input = input("📝 Your Query: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("👋 Exiting. Goodbye!")
            break
        try:
            response = graph.invoke({"input": user_input})
            print("\n✅", response["output"], "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
