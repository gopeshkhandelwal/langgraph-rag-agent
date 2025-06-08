import sys
import os
from langchain_core.messages import HumanMessage, AIMessage
from config import get_llm
from langgraph_graph import build_graph
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
# Fix import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Configuration ===
MAX_MESSAGES = 10 # Maximum number of messages to keep in history
history = []

if __name__ == "__main__":
    logger.info(f"\n🌐 LangGraph Agent ready. Type your query or 'quit' to exit.\n")
    graph = build_graph()

    while True:
        user_input = input("📝 Your Query: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            logger.info("👋 Exiting. Goodbye!")
            break

        try:
            # Step 1: Add user input to history
            logger.debug(f"Original History ({len(history)} messages):")
            for i, msg in enumerate(history):
                logger.debug(f"  {i + 1}: {msg.content}")
            history.append(HumanMessage(content=user_input))

            logger.info(f"\n🔄 Processing your query...\n")  # <-- Always prints before graph.invoke

            # Step 2: Build state and invoke graph
            state = {"messages": history}
            response = graph.invoke(state)
            last_message = response.get("messages")[-1]
            logger.info(f"\n✅ {last_message.content}\n")
            
            # Step 3: Add agent response to history for context in next turn
            history.append(last_message)
            history = history[-MAX_MESSAGES:]  # Trim after both user and agent messages

            logger.debug(f"Updated History ({len(history)} messages):")
            for i, msg in enumerate(history):
                logger.debug(f"  {i + 1}: {msg.content}")

        except Exception as e:
            logger.error(f"⚠️ Error: {e}\n")
