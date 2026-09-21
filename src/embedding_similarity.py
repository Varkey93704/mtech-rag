from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Regression testing verifies existing functionality after software changes.",
    "Regression tests ensure that modifications do not break previously working features.",
    "The weather forecast predicts heavy rainfall tomorrow.",
]

embeddings = model.encode(sentences)

similarity_matrix = cosine_similarity(embeddings)

print("Similarity Matrix:")
print(similarity_matrix)

print("\nPairwise Similarities:")
print(f"A vs B: {similarity_matrix[0][1]:.4f}")
print(f"A vs C: {similarity_matrix[0][2]:.4f}")
print(f"B vs C: {similarity_matrix[1][2]:.4f}")