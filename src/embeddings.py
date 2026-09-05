import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Configuration
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

EMBEDDING_DIMENSION = int(
    os.getenv(
        "EMBEDDING_DIMENSION",
        "768"
    )
)


if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY is not set."
    )


# ============================================================
# Gemini Embedding Model
# ============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY,
    output_dimensionality=EMBEDDING_DIMENSION
)


# ============================================================
# Configuration Helper
# ============================================================

def get_embedding_dimension():
    """
    Return the configured embedding dimension.
    """

    return EMBEDDING_DIMENSION


# ============================================================
# Embed Documents
# ============================================================

def embed_documents(texts):
    """
    Generate embeddings for multiple documents.
    """

    vectors = embedding_model.embed_documents(
        texts
    )


    # --------------------------------------------------------
    # Validate dimension
    # --------------------------------------------------------

    for vector in vectors:

        if len(vector) != EMBEDDING_DIMENSION:

            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"got {len(vector)}."
            )


    return vectors


# ============================================================
# Embed Query
# ============================================================

def embed_query(text):
    """
    Generate an embedding for a user question.
    """

    vector = embedding_model.embed_query(
        text
    )


    # --------------------------------------------------------
    # Validate dimension
    # --------------------------------------------------------

    if len(vector) != EMBEDDING_DIMENSION:

        raise ValueError(
            "Query embedding dimension mismatch. "
            f"Expected {EMBEDDING_DIMENSION}, "
            f"got {len(vector)}."
        )


    return vector

# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    test_text = (
        "This is a test sentence for embeddings."
    )


    vector = embed_query(
        test_text
    )


    print(
        f"Configured dimension: "
        f"{EMBEDDING_DIMENSION}"
    )


    print(
        f"Actual dimension: "
        f"{len(vector)}"
    )


    print(
        "First 10 values:"
    )


    print(
        vector[:10]
    )