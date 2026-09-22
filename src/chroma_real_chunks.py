import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load the document
file_path = "data/documents/chunking_test.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

document = Document(
    page_content=text,
    metadata={"source": file_path},
)


# 2. Create the same splitter configuration
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512,
    chunk_overlap=50,
)


# 3. Create chunks
chunks = text_splitter.split_documents([document])

print(f"Number of chunks: {len(chunks)}")


# 4. Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 5. Extract chunk text
documents = [chunk.page_content for chunk in chunks]

# 6. Generate embeddings
embeddings = embedding_model.encode(documents).tolist()


# 7. Create persistent ChromaDB client
client = chromadb.PersistentClient(path="data/chroma")

collection = client.get_or_create_collection(
    name="real_document"
)


# 8. Create IDs
ids = [f"chunk_{i + 1:03d}" for i in range(len(chunks))]


# 9. Store chunks in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=[chunk.metadata for chunk in chunks],
)


print(f"Documents stored in ChromaDB: {collection.count()}")


# 10. Query ChromaDB
query = "What is regression testing?"

query_embedding = embedding_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
)


# 11. Display results
print("\nQuery:")
print(query)

print("\nTop 5 Results:")

for i in range(len(results["ids"][0])):
    print(f"\nRank {i + 1}")
    print(f"ID: {results['ids'][0][i]}")
    print(f"Distance: {results['distances'][0][i]:.4f}")
    print(f"Metadata: {results['metadatas'][0][i]}")
    print(f"Document: {results['documents'][0][i][:300]}...")