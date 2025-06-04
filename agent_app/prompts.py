from langchain.prompts import PromptTemplate

def get_custom_rag_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template=""""
If the context is short or ambiguous, do your best to explain the topic based on your knowledge. 
Do not repeat the question. Make your response informative.

Context:
{context}

Question:
{question}

Answer in a concise and helpful way:
""".strip()
    )
