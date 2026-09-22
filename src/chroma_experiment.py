import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Create a persistent Chroma client
client = chromadb.PersistentClient(path="data/chroma")

# Create or get our collection
collection = client.get_or_create_collection(
    name="rag_documents"
)

# Our test chunks
documents = [
    "Regression testing verifies that existing functionality still works after software changes.",
    "API testing validates application programming interfaces, including requests, responses, authentication, and schemas.",
    "Continuous integration allows teams to integrate code changes frequently and automatically execute validation activities."
]

# Generate embeddings
embeddings = embedding_model.encode(documents).tolist()

# Store the chunks in ChromaDB
collection.upsert(
    ids=["chunk_001", "chunk_002", "chunk_003"],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"source": "chroma_experiment", "chunk_id": 1},
        {"source": "chroma_experiment", "chunk_id": 2},
        {"source": "chroma_experiment", "chunk_id": 3},
    ]
)
# User query
query = "What is regression testing?"

# Convert the query into an embedding
query_embedding = embedding_model.encode(query).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nQuery:")
print(query)

print("\nSearch Results:")
print(results)

print(f"Collection: {collection.name}")
print(f"Number of documents: {collection.count()}")

print("\nStored documents:")
print(collection.get(include=["documents", "embeddings", "metadatas"]))