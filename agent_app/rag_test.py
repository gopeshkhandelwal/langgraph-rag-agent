from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

INDEX_DIR = "vectorstore/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(INDEX_DIR):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_DIR}. Run `build_vectorstore.py` first.")

vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
docs = retriever.invoke("Intel Tiber Developer Cloud")
for d in docs:
    print(d.page_content)
