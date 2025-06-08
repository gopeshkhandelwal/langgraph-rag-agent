from langchain.prompts import PromptTemplate

def get_custom_rag_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template=""""
Answer the question using only the provided context. If the answer is not in the context, say "I don't know."
Do not repeat the question. Make your response informative.

Context:
{context}

Question:
{question}

Answer in a concise and helpful way:
""".strip()
    )
