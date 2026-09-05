import json

from pathlib import Path

from src.vector_store import (
    create_faiss_index,
    search_faiss
)

from src.keyword_search import (
    create_bm25_index,
    search_bm25
)

from src.hybrid_search import (
    hybrid_search
)


# ============================================================
# Load evaluation questions
# ============================================================

evaluation_file = Path(
    "tests/evaluation_questions.json"
)


with open(
    evaluation_file,
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(
        file
    )


# ============================================================
# Load test chunks
# ============================================================

# For now we will reuse the PDF manually.
#
# We will improve this later so the evaluator
# automatically processes a PDF.


import fitz


PDF_PATH = "Smart Document Assistant.pdf"


document = fitz.open(
    PDF_PATH
)


from src.text_splitter import split_text


all_chunks = []


for page_number, page in enumerate(
    document
):

    page_text = page.get_text()


    if not page_text.strip():
        continue


    chunks = split_text(
        page_text,
        chunk_size=1000,
        chunk_overlap=200
    )


    for chunk in chunks:

        all_chunks.append(
            {
                "text": chunk,

                "metadata": {
                    "source": PDF_PATH,
                    "page": page_number + 1,
                    "chunk_id": len(all_chunks)
                }
            }
        )


# ============================================================
# Create indexes
# ============================================================

print(
    f"Loaded {len(all_chunks)} chunks."
)


print(
    "Creating FAISS index..."
)


faiss_index = create_faiss_index(
    all_chunks
)


print(
    "Creating BM25 index..."
)


bm25_index = create_bm25_index(
    all_chunks
)


# ============================================================
# Evaluation function
# ============================================================

def check_hit(
    results,
    expected_page
):

    for result in results:

        page = result[
            "metadata"
        ]["page"]


        if page == expected_page:

            return True


    return False


# ============================================================
# Counters
# ============================================================

faiss_hits = 0
bm25_hits = 0
hybrid_hits = 0


total = len(
    questions
)


# ============================================================
# Run evaluation
# ============================================================

for item in questions:

    question = item[
        "question"
    ]

    expected_page = item[
        "expected_page"
    ]


    print(
        "\n"
        + "=" * 70
    )


    print(
        f"Question: {question}"
    )


    print(
        f"Expected page: {expected_page}"
    )


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    faiss_results = search_faiss(
        faiss_index,
        all_chunks,
        question,
        k=5
    )


    faiss_hit = check_hit(
        faiss_results,
        expected_page
    )


    if faiss_hit:

        faiss_hits += 1


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = search_bm25(
        bm25_index,
        all_chunks,
        question,
        k=5
    )


    bm25_hit = check_hit(
        bm25_results,
        expected_page
    )


    if bm25_hit:

        bm25_hits += 1


    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_results = hybrid_search(
        faiss_results,
        bm25_results,
        k=5
    )


    hybrid_hit = check_hit(
        hybrid_results,
        expected_page
    )


    if hybrid_hit:

        hybrid_hits += 1


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"FAISS:  {'HIT' if faiss_hit else 'MISS'}"
    )

    print(
        f"BM25:   {'HIT' if bm25_hit else 'MISS'}"
    )

    print(
        f"Hybrid: {'HIT' if hybrid_hit else 'MISS'}"
    )


# ============================================================
# Final results
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "EVALUATION RESULTS"
)


print(
    "=" * 70
)


print(
    f"Total questions: {total}"
)


print(
    f"FAISS Hit@5: "
    f"{faiss_hits}/{total} "
    f"({faiss_hits / total * 100:.1f}%)"
)


print(
    f"BM25 Hit@5: "
    f"{bm25_hits}/{total} "
    f"({bm25_hits / total * 100:.1f}%)"
)


print(
    f"Hybrid Hit@5: "
    f"{hybrid_hits}/{total} "
    f"({hybrid_hits / total * 100:.1f}%)"
)