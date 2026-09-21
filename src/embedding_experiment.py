from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Text to embed
text = "Regression testing verifies that existing functionality still works after software changes."

# Generate embedding
embedding = model.encode(text)

# Inspect the result
print(f"Embedding type: {type(embedding)}")
print(f"Embedding dimensions: {embedding.shape}")
print(f"First 10 values: {embedding[:10]}")