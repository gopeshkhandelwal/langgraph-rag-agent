from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

retriever = FAISS.load_local("vectorstore/faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"), allow_dangerous_deserialization=True).as_retriever()
docs = retriever.get_relevant_documents("What is Intel Tiber AI Cloud")
for d in docs:
    print(d.page_content)