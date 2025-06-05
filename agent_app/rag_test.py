from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

retriever = FAISS.load_local("vectorstore/faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True).as_retriever()
docs = retriever.invoke("What is Intel Tiber AI Cloud")
for d in docs:
    print(d.page_content)
