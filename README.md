# LangGraph RAG Agent

This project demonstrates a Retrieval-Augmented Generation (RAG) agent built using [LangGraph](https://docs.langchain.com/langgraph/) and [LangChain], using `ToolNode` and `ToolExecutor` to invoke tools like document-based Q&A and weather APIs.

---

## 📦 Features

- ✅ LangGraph-based agent orchestration
- 📄 `DocumentQA`: Answer questions from internal documents via FAISS + HuggingFace embeddings
- ☁️ `CityWeather`: Fetch real-time weather using OpenWeather API
- 🔧 ToolExecutor + ToolNode integration (latest best practices)

---

## 🚀 Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Environment Variables

Create a `.env` file in the root folder with:

```env
OPENAI_API_BASE_VLLM=http://your-server:8000/v1
OPENAI_MODEL_VLLM=vllm-fork-with-llama-2-7b-chat-hf
OPENWEATHER_API_KEY=your_openweather_api_key
```

---

## 🛠 Build VectorStore

```bash
python utils/build_vectorstore.py
```

Make sure `docs/itac.txt` exists and contains content to embed.

---

## ▶️ Run the Agent

```bash
PYTHONPATH=. python agent_app/main.py
```

This will launch the interactive RAG agent with support for:
- `DocumentQA`: file-based queries
- `CityWeather`: real-time city weather

---

## 🧹 Developer Utilities

| Command         | Description                  |
|----------------|------------------------------|
| `make build-vectorstore` | Build the FAISS vector DB |
| `make run`              | Run the agent             |
| `make lint`             | Lint code with flake8     |
| `make format`           | Autoformat with black     |
| `make clean`            | Remove `*.pyc` files      |

---

## 📁 Folder Structure

```
agent_app/
  ├── config.py
  ├── langgraph_graph.py  # Tool-based LangGraph logic
  ├── main.py             # CLI entrypoint
  ├── prompts.py
  ├── rag.py
  └── weather.py

utils/
  └── build_vectorstore.py
```

---

## 📚 References

- LangGraph: https://docs.langchain.com/langgraph
- LangChain: https://docs.langchain.com/docs
- OpenWeatherMap API: https://openweathermap.org/api

---

Happy hacking! 🔧🧠
