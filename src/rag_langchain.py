from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough


# 1. Load the embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create an embedding function for LangChain
class SentenceTransformerEmbeddings:
    def embed_documents(self, texts):
        return embedding_model.encode(texts).tolist()

    def embed_query(self, text):
        return embedding_model.encode(text).tolist()


embeddings = SentenceTransformerEmbeddings()


# 3. Connect LangChain to our existing ChromaDB
vectorstore = Chroma(
    collection_name="real_document",
    persist_directory="data/chroma",
    embedding_function=embeddings,
)


# 4. Create a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 5. Create the prompt template
prompt = ChatPromptTemplate.from_template(
    """
You are a question-answering system for a software testing knowledge base.

IMPORTANT RULES:
1. Answer ONLY using information explicitly contained in the provided context.
2. Do NOT use your general knowledge or information from outside the context.
3. If the context does not contain the answer, respond exactly:
   "The answer is not available in the provided context."
4. Do not add explanations, assumptions, or recommendations based on outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)


# 6. Create the LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)


# 7. Build the LCEL RAG chain
def format_documents(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )


rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)


# 8. Get the user's question
question = input("\nEnter your question: ")


# 9. Run the complete RAG chain
response = rag_chain.invoke(question)


# 10. Display the answer
print("\n--- LLM Answer ---")
print(response.content)