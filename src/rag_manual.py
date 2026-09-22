import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama


# 1. Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to our existing ChromaDB
client = chromadb.PersistentClient(path="data/chroma")

collection = client.get_collection(
    name="real_document"
)


# 3. Create the LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)


# 4. Get user query
query = input("\nEnter your question: ")


# 5. Convert query into an embedding
query_embedding = embedding_model.encode(query).tolist()


# 6. Retrieve the top 3 relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)


# 7. Extract retrieved documents
retrieved_documents = results["documents"][0]


# 8. Combine retrieved chunks into context
context = "\n\n".join(retrieved_documents)


# 9. Build the prompt
prompt = f"""
You are a helpful assistant answering questions using the provided context.

IMPORTANT RULES:
1. Answer ONLY using information explicitly contained in the provided context.
2. Do NOT use your general knowledge or information from outside the context.
3. If the context does not contain the answer, respond exactly:
   "The answer is not available in the provided context."
4. Do not add explanations, assumptions, or recommendations based on outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""


# 10. Send the prompt to Llama 3.1
response = llm.invoke(prompt)


# 11. Display the result
print("\n--- Retrieved Context ---")

for i, document in enumerate(retrieved_documents):
    print(f"\n[Chunk {i + 1}]")
    print(document)

print("\n--- LLM Answer ---")
print(response.content)