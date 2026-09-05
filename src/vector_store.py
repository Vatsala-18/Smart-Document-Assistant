import os

import faiss
import numpy as np

from dotenv import load_dotenv

from src.embeddings import (
    embed_documents,
    embed_query,
    get_embedding_dimension
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Create FAISS Index
# ============================================================

def create_faiss_index(chunks):
    """
    Create a FAISS index from document chunks.
    """

    if not chunks:

        raise ValueError(
            "No chunks provided."
        )


    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    vectors = embed_documents(
        texts
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    vectors = np.array(
        vectors,
        dtype="float32"
    )


    print(
        f"Embedding shape: {vectors.shape}"
    )


    # --------------------------------------------------------
    # Configured dimension
    # --------------------------------------------------------

    configured_dimension = (
        get_embedding_dimension()
    )


    actual_dimension = (
        vectors.shape[1]
    )


    print(
        f"Configured dimension: "
        f"{configured_dimension}"
    )

    print(
        f"Actual dimension: "
        f"{actual_dimension}"
    )


    # --------------------------------------------------------
    # Validate dimension
    # --------------------------------------------------------

    if actual_dimension != configured_dimension:

        raise ValueError(
            "FAISS embedding dimension mismatch. "
            f"Expected {configured_dimension}, "
            f"got {actual_dimension}."
        )


    # --------------------------------------------------------
    # Normalize vectors
    # --------------------------------------------------------

    faiss.normalize_L2(
        vectors
    )


    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        configured_dimension
    )


    # --------------------------------------------------------
    # Add vectors
    # --------------------------------------------------------

    index.add(
        vectors
    )


    print(
        f"FAISS contains "
        f"{index.ntotal} vectors"
    )


    return index


# ============================================================
# Search FAISS
# ============================================================

def search_faiss(
    index,
    chunks,
    question,
    k=5,
    score_threshold=None
):
    """
    Search FAISS using a user question.
    """

    if index is None:

        raise ValueError(
            "FAISS index is None."
        )


    if not chunks:

        raise ValueError(
            "No chunks available."
        )


    if not question.strip():

        return []


    # --------------------------------------------------------
    # Limit k
    # --------------------------------------------------------

    k = min(
        k,
        index.ntotal
    )


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_vector = embed_query(
        question
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    query_vector = np.array(
        [query_vector],
        dtype="float32"
    )


    # --------------------------------------------------------
    # Validate query dimension
    # --------------------------------------------------------

    expected_dimension = (
        get_embedding_dimension()
    )


    actual_dimension = (
        query_vector.shape[1]
    )


    if actual_dimension != expected_dimension:

        raise ValueError(
            "Query embedding dimension mismatch. "
            f"Expected {expected_dimension}, "
            f"got {actual_dimension}."
        )


    # --------------------------------------------------------
    # Normalize query
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_vector
    )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    similarities, indices = index.search(
        query_vector,
        k
    )


    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    results = []


    for similarity, index_position in zip(
        similarities[0],
        indices[0]
    ):

        similarity = float(
            similarity
        )


        if index_position < 0:

            continue


        if (
            score_threshold is not None
            and similarity < score_threshold
        ):

            continue


        if index_position >= len(chunks):

            continue


        chunk = chunks[
            index_position
        ]


        results.append(
            {
                "text": chunk["text"],

                "metadata": chunk["metadata"],

                "score": similarity
            }
        )


    return results