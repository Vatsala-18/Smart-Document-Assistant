import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf

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

from src.text_splitter import split_text


# ============================================================
# Load evaluation questions
# ============================================================

evaluation_file = PROJECT_ROOT / "tests" / "evaluation_questions.json"

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

PDF_NAME = "Smart Document Assistant.pdf"
PDF_PATH = PROJECT_ROOT / PDF_NAME

document = pymupdf.open(
    PDF_PATH
)

all_chunks = []

for page_number, page in enumerate(
    document,
    start=1
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
                    "source": PDF_NAME,
                    "page": page_number,
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
    expected_pages
):
    """
    Checks if any of the expected pages appear in the retrieval results.
    Returns True if at least one expected page was retrieved, False otherwise.
    """
    if not expected_pages:
        return False

    if isinstance(expected_pages, int):
        expected_pages = [expected_pages]

    retrieved_pages = {
        result["metadata"].get("page")
        for result in results
        if "metadata" in result
    }

    return any(
        page in retrieved_pages
        for page in expected_pages
    )


# ============================================================
# Counters
# ============================================================

faiss_hits = 0
bm25_hits = 0
hybrid_hits = 0

total = len(questions)
evaluable_total = 0


# ============================================================
# Run evaluation
# ============================================================

for item in questions:

    question = item[
        "question"
    ]

    # Support both 'expected_pages' (list) and legacy 'expected_page' (int or list)
    raw_expected = item.get("expected_pages")
    if raw_expected is None:
        raw_expected = item.get("expected_page", [])

    if isinstance(raw_expected, int):
        expected_pages = [raw_expected]
    else:
        expected_pages = list(raw_expected)

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Question: {question}"
    )

    if expected_pages:
        print(
            f"Expected page(s): {expected_pages}"
        )
    else:
        print(
            "Expected page(s): None (unanswerable question)"
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
        expected_pages
    )

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
        expected_pages
    )

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
        expected_pages
    )

    if expected_pages:
        evaluable_total += 1

        if faiss_hit:
            faiss_hits += 1

        if bm25_hit:
            bm25_hits += 1

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
    else:
        print(
            "FAISS:  N/A (unanswerable question)"
        )

        print(
            "BM25:   N/A (unanswerable question)"
        )

        print(
            "Hybrid: N/A (unanswerable question)"
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
    f"Total questions: {total} ({evaluable_total} evaluable with expected pages)"
)

denom = evaluable_total if evaluable_total > 0 else 1

print(
    f"FAISS Hit@5: "
    f"{faiss_hits}/{evaluable_total} "
    f"({faiss_hits / denom * 100:.1f}%)"
)

print(
    f"BM25 Hit@5: "
    f"{bm25_hits}/{evaluable_total} "
    f"({bm25_hits / denom * 100:.1f}%)"
)

print(
    f"Hybrid Hit@5: "
    f"{hybrid_hits}/{evaluable_total} "
    f"({hybrid_hits / denom * 100:.1f}%)"
)