import numpy as np
import faiss


# Create three simple vectors
vectors = np.array(
    [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ],
    dtype="float32"
)


print("Vectors:")
print(vectors)


# Vector dimension
dimension = vectors.shape[1]

print(f"Vector dimension: {dimension}")


# Create FAISS index
index = faiss.IndexFlatL2(dimension)


# Add vectors to FAISS
index.add(vectors)


print(f"Number of vectors in FAISS: {index.ntotal}")