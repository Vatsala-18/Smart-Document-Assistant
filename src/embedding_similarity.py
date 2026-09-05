import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


sentences = [
    "RAG retrieves relevant information from documents.",
    "Retrieval augmented generation searches documents to find useful context.",
    "Chocolate cake is made using flour, eggs and sugar."
]


# Generate embeddings
vectors = embeddings.embed_documents(sentences)


# Calculate similarities
similarity_1_2 = cosine_similarity(
    [vectors[0]],
    [vectors[1]]
)[0][0]

similarity_1_3 = cosine_similarity(
    [vectors[0]],
    [vectors[2]]
)[0][0]

similarity_2_3 = cosine_similarity(
    [vectors[1]],
    [vectors[2]]
)[0][0]


# Print actual values
print(f"Sentence 1 ↔ Sentence 2: {similarity_1_2:.4f}")
print(f"Sentence 1 ↔ Sentence 3: {similarity_1_3:.4f}")
print(f"Sentence 2 ↔ Sentence 3: {similarity_2_3:.4f}")


# Create graph
labels = [
    "RAG ↔ Retrieval",
    "RAG ↔ Cake",
    "Retrieval ↔ Cake"
]

values = [
    similarity_1_2,
    similarity_1_3,
    similarity_2_3
]


plt.figure(figsize=(9, 5))

plt.bar(labels, values)

plt.ylabel("Cosine Similarity")
plt.xlabel("Sentence Pair")
plt.title("Semantic Similarity Using Gemini Embeddings")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()