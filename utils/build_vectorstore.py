import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(ROOT_DIR, "..", "docs", "itac.txt")

with open(DOC_PATH, "r") as f:
    content = f.read()

docs = [Document(page_content=chunk) for chunk in CharacterTextSplitter(
    chunk_size=500, chunk_overlap=50).split_text(content)]
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

INDEX_PATH = os.path.join(ROOT_DIR, "..", "vectorstore", "faiss_index")
vectorstore.save_local(INDEX_PATH)
print(f"✅ FAISS index saved to {INDEX_PATH}")
