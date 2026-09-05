import re

from rank_bm25 import BM25Okapi


# ============================================================
# Tokenizer
# ============================================================

def tokenize(text):
    """
    Convert text into tokens for BM25.

    Example:

    "What is an Embedding?"

    becomes:

    [
        "what",
        "is",
        "an",
        "embedding"
    ]
    """

    return re.findall(
        r"\b[\w-]+\b",
        text.lower()
    )


# ============================================================
# Create BM25 Index
# ============================================================

def create_bm25_index(chunks):
    """
    Create a BM25 index from document chunks.
    """

    if not chunks:
        raise ValueError(
            "No chunks provided for BM25."
        )


    tokenized_corpus = []


    for chunk in chunks:

        text = chunk["text"]

        tokens = tokenize(text)

        tokenized_corpus.append(
            tokens
        )


    bm25 = BM25Okapi(
        tokenized_corpus
    )


    return bm25


# ============================================================
# Search BM25
# ============================================================

def search_bm25(
    bm25,
    chunks,
    question,
    k=5
):
    """
    Search document chunks using BM25.
    """

    if bm25 is None:
        raise ValueError(
            "BM25 index is not available."
        )


    if not chunks:
        return []


    if not question.strip():
        return []


    query_tokens = tokenize(
        question
    )


    scores = bm25.get_scores(
        query_tokens
    )


    # Highest scores first
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )


    results = []


    for index_position in ranked_indices[:k]:

        chunk = chunks[
            index_position
        ]


        results.append(
            {
                "text": chunk["text"],

                "metadata": chunk["metadata"],

                "bm25_score": float(
                    scores[index_position]
                )
            }
        )


    return results