# LangGraph RAG Agent

A Retrieval-Augmented Generation (RAG) agent using [LangGraph](https://docs.langchain.com/langgraph/) and [LangChain](https://docs.langchain.com/docs), featuring modular tool invocation for document Q&A and weather APIs.

---

## 🚩 Features

- 🧩 LangGraph-based agent orchestration
- 📄 `DocumentQA`: Answers from internal docs (FAISS + HuggingFace embeddings)
- ☁️ `CityWeather`: Real-time weather via OpenWeather API
- 🔧 Modern `ToolExecutor` + `ToolNode` integration

---

## 🚀 Quickstart

### 1. Create & Activate Virtual Environment

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
# === OpenAI Official API ===
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# === OpenWeatherMap API ===
OPENWEATHER_API_KEY=your-openweather-api-key

# === IDC API Token (for MCP-integrated demos) ===
IDC_API_TOKEN=your-idc-token-if-applicable

```

---

## 🛠 Build VectorStore

```bash
make build-vectorstore
```

Make sure `docs/itac.txt` exists and contains content to embed.

---

## ▶️ Run the Agent

```bash
make run
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
