import json
import re
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so 'src' imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import embed_documents
from src.hybrid_search import hybrid_search
from src.keyword_search import create_bm25_index, search_bm25
from src.llm import generate_answer
from src.text_splitter import split_text
from src.vector_store import create_faiss_index, search_faiss

# ============================================================
# Configuration
# ============================================================

PDF_PATH = PROJECT_ROOT / "Smart Document Assistant.pdf"
DATASET_PATH = PROJECT_ROOT / "tests" / "evaluation_questions.json"
OUTPUT_PATH = PROJECT_ROOT / "tests" / "answer_evaluation_results.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

FAISS_TOP_K = 8
BM25_TOP_K = 8
HYBRID_TOP_K = 5

# Semantic similarity threshold for an answer to be considered correct
SEMANTIC_SIMILARITY_THRESHOLD = 0.80

# Delay between LLM calls to respect API rate limits (seconds)
LLM_CALL_DELAY = 1.0

# Status Constants
STATUS_SUCCESS = "SUCCESS"
STATUS_REFUSAL = "REFUSAL"
STATUS_GENERATION_ERROR = "GENERATION_ERROR"
STATUS_INCORRECT = "INCORRECT"


# ============================================================
# Load evaluation dataset
# ============================================================

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH.absolute()}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Load and chunk PDF
# ============================================================

def load_chunks():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.absolute()}"
        )

    print(f"\nLoading PDF: {PDF_PATH}")

    document = pymupdf.open(PDF_PATH)

    chunks = []

    for page_number, page in enumerate(document, start=1):

        page_text = page.get_text()

        if not page_text.strip():
            continue

        page_chunks = split_text(
            page_text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        for chunk_text in page_chunks:

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": PDF_PATH.name,
                    "page": page_number,
                    "chunk_id": len(chunks)
                }
            })

    document.close()

    return chunks


# ============================================================
# Check whether expected page was retrieved
# ============================================================

def check_retrieval_hit(retrieved_results, expected_pages):
    """
    A question is considered a retrieval HIT if at least
    one expected page occurs in the hybrid top-k results.
    Returns None if expected_pages is empty.
    """
    if not expected_pages:
        return None

    retrieved_pages = {
        result["metadata"].get("page")
        for result in retrieved_results
    }

    return any(
        page in retrieved_pages
        for page in expected_pages
    )


# ============================================================
# Get unique retrieved pages
# ============================================================

def get_retrieved_pages(results):
    pages = []

    for result in results:

        page = result["metadata"].get("page")

        if page not in pages:
            pages.append(page)

    return pages


# ============================================================
# Normalize text
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    return " ".join(str(text).lower().split())


# ============================================================
# Check unanswerable response
# ============================================================

def check_unanswerable_response(answer):
    """
    The current LLM implementation is expected to return:
    "The answer is not available in the uploaded document."
    for questions whose answer is not present.
    """
    expected = "The answer is not available in the uploaded document."
    return normalize_text(answer) == normalize_text(expected)


# ============================================================
# Strip citation footer from generated answer
# ============================================================

def strip_citations(text):
    """
    Strips source and page citation footers (e.g., 'Source: ...' or 'Page 1')
    to isolate the core semantic answer content for evaluation.
    """
    if not text:
        return ""

    cleaned = re.split(
        r"(?i)\n+(?:source:|sources:|page\s*\d|pages\s*\d|\*\*source|\*\*page)",
        text
    )[0].strip()

    return cleaned if cleaned else text.strip()


# ============================================================
# Detect generation errors (API/Quota/Network failures)
# ============================================================

def is_generation_error(answer):
    """
    Checks if the generated answer string indicates an API error,
    quota exhaustion (429), connection failure, or empty response,
    rather than a genuine model response.
    """
    if not answer or not isinstance(answer, str):
        return True

    clean = answer.strip()
    return (
        clean.startswith("ERROR:")
        or clean.startswith("Traceback (most recent call last):")
        or "429 RESOURCE_EXHAUSTED" in clean
        or "GoogleRateLimitError" in clean
        or "Quota exceeded" in clean
        or "ResourceExhausted" in clean
    )


# ============================================================
# Classify answer: SUCCESS / REFUSAL / GENERATION_ERROR / INCORRECT
# ============================================================

def classify_answer(
    answerable,
    expected_answer,
    generated_answer,
    threshold=SEMANTIC_SIMILARITY_THRESHOLD
):
    """
    Distinguishes:
      - GENERATION_ERROR: API/quota/network failures (excluded from quality accuracy)
      - REFUSAL: Model returned unanswerable refusal on an answerable question (false refusal)
      - SUCCESS: Correct answer (semantically matched for answerable; refused for unanswerable)
      - INCORRECT: Semantic mismatch for answerable; hallucinated answer for unanswerable

    Returns:
      (status, answer_correct, similarity_score, evaluation_reason)
    """
    # 1. Generation error (429, quota, crash) -> Exclude from quality accuracy
    if is_generation_error(generated_answer):
        return (
            STATUS_GENERATION_ERROR,
            None,  # Neither PASS nor FAIL in quality metrics
            None,
            "API / generation error (429/quota/network) - excluded from answer quality metrics"
        )

    is_refusal = check_unanswerable_response(generated_answer)

    # 2. Answerable question evaluation
    if answerable:
        if is_refusal:
            return (
                STATUS_REFUSAL,
                False,
                0.0,
                "Model returned unanswerable refusal for an answerable question [FAIL]"
            )

        clean_gen = strip_citations(generated_answer)
        clean_exp = strip_citations(expected_answer)

        try:
            vectors = embed_documents([clean_exp, clean_gen])
            similarity = float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])
            is_correct = similarity >= threshold
            status = STATUS_SUCCESS if is_correct else STATUS_INCORRECT
            status_tag = "PASS" if is_correct else "FAIL"
            reason = (
                f"Semantic similarity {similarity:.4f} "
                f"{'>=' if is_correct else '<'} threshold {threshold:.2f} [{status_tag}]"
            )
            return status, is_correct, similarity, reason
        except Exception as exc:
            # Fallback to lexical token recall if embedding API is unreachable
            tokens_exp = set(re.findall(r"\w+", clean_exp.lower()))
            tokens_gen = set(re.findall(r"\w+", clean_gen.lower()))
            if tokens_exp:
                recall = len(tokens_exp & tokens_gen) / len(tokens_exp)
                is_correct = recall >= 0.60
                status = STATUS_SUCCESS if is_correct else STATUS_INCORRECT
                status_tag = "PASS" if is_correct else "FAIL"
                reason = f"Fallback token recall {recall:.4f} [{status_tag}] (Embeddings error: {exc})"
                return status, is_correct, recall, reason
            return STATUS_INCORRECT, False, 0.0, f"Error calculating similarity: {exc}"

    # 3. Unanswerable question evaluation
    else:
        if is_refusal:
            return (
                STATUS_SUCCESS,
                True,
                None,
                "PASS (Correctly returned standard refusal response)"
            )
        else:
            return (
                STATUS_INCORRECT,
                False,
                None,
                "FAIL (Hallucinated answer instead of refusing unanswerable question)"
            )


# ============================================================
# Print retrieved results
# ============================================================

def print_retrieved_results(results):

    for index, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")

        rrf_score = result.get("rrf_score", 0.0)
        faiss_score = result.get("faiss_score")
        bm25_score = result.get("bm25_score")

        print(
            f"  {index}. "
            f"Page={page}, "
            f"Chunk={chunk_id}, "
            f"RRF={rrf_score:.6f}, "
            f"FAISS={faiss_score}, "
            f"BM25={bm25_score}"
        )


# ============================================================
# Main evaluation
# ============================================================

def main():
    # Check CLI options
    eval_only = "--eval-only" in sys.argv or "--cached" in sys.argv

    print("=" * 80)
    print("SMART DOCUMENT ASSISTANT")
    print("ANSWER-LEVEL RAG EVALUATION")
    if eval_only:
        print("MODE: Evaluating cached answers from output file")
    else:
        print("MODE: Full RAG pipeline (Retrieval + LLM Answer Generation)")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_dataset()
    total_questions = len(dataset)

    # Load existing results if available (used for cached evaluation or fallback on 429)
    cached_answers = {}
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
                prev_results = json.load(file)
                for res in prev_results:
                    qid = res.get("id")
                    ans = res.get("generated_answer")
                    if qid and ans and not ans.startswith("ERROR:"):
                        cached_answers[qid] = ans
        except Exception:
            pass

    print(f"\nEvaluation questions loaded: {total_questions}")

    # --------------------------------------------------------
    # Load document and create chunks
    # --------------------------------------------------------

    chunks = load_chunks()
    print(f"Document chunks created: {len(chunks)}")

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    print("\nCreating FAISS index...")
    faiss_index = create_faiss_index(chunks)

    # --------------------------------------------------------
    # Create BM25 index
    # --------------------------------------------------------

    print("\nCreating BM25 index...")
    bm25_index = create_bm25_index(chunks)

    # --------------------------------------------------------
    # Evaluation counters
    # --------------------------------------------------------

    # Retrieval denominator: count ONLY questions that have expected_pages (23/23)
    retrieval_total = sum(1 for item in dataset if item.get("expected_pages"))
    retrieval_hits = 0

    total_evaluated_successfully = 0
    total_generation_errors = 0

    answerable_total = sum(1 for item in dataset if item.get("answerable", True))
    answerable_evaluated = 0
    answerable_success = 0
    answerable_refusal = 0
    answerable_incorrect = 0
    answerable_generation_errors = 0

    unanswerable_total = total_questions - answerable_total
    unanswerable_evaluated = 0
    unanswerable_success = 0
    unanswerable_incorrect = 0
    unanswerable_generation_errors = 0

    results = []

    # --------------------------------------------------------
    # Evaluate each question
    # --------------------------------------------------------

    for position, item in enumerate(dataset, start=1):

        question_id = item.get("id", position)
        question = item["question"]
        expected_pages = item.get("expected_pages", [])
        answerable = item.get("answerable", True)
        expected_answer = item.get("expected_answer", "")

        print("\n")
        print("=" * 80)
        print(f"QUESTION {position}/{total_questions}")
        print("=" * 80)

        print(f"\nID: {question_id}")
        print(f"Question: {question}")
        print(f"Expected pages: {expected_pages}")
        print(f"Answerable: {answerable}")

        # ----------------------------------------------------
        # FAISS retrieval
        # ----------------------------------------------------

        faiss_results = search_faiss(
            faiss_index,
            chunks,
            question,
            k=FAISS_TOP_K
        )

        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        bm25_results = search_bm25(
            bm25_index,
            chunks,
            question,
            k=BM25_TOP_K
        )

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        hybrid_results = hybrid_search(
            faiss_results,
            bm25_results,
            k=HYBRID_TOP_K
        )

        retrieved_pages = get_retrieved_pages(hybrid_results)
        print("\nRetrieved pages:")
        print(retrieved_pages)

        print("\nHybrid retrieval results:")
        print_retrieved_results(hybrid_results)

        # ----------------------------------------------------
        # Retrieval evaluation (23 denominator)
        # ----------------------------------------------------

        if expected_pages:
            retrieval_hit = check_retrieval_hit(
                hybrid_results,
                expected_pages
            )
            if retrieval_hit:
                retrieval_hits += 1
            retrieval_display = "HIT" if retrieval_hit else "MISS"
        else:
            # Unanswerable question has no expected pages in document
            retrieval_hit = None
            retrieval_display = "N/A (unanswerable question - no expected pages)"

        print(f"\nRetrieval result: {retrieval_display}")

        # ----------------------------------------------------
        # Generate or reuse answer
        # ----------------------------------------------------

        if eval_only and question_id in cached_answers:
            print("\nUsing cached answer...")
            generated_answer = cached_answers[question_id]
        else:
            print("\nGenerating answer using Gemini...")
            try:
                generated_answer = generate_answer(
                    question,
                    hybrid_results
                )
            except Exception as error:
                print(f"\nERROR generating answer: {error}")
                # Fallback to cached answer if available
                if question_id in cached_answers:
                    print("Falling back to previously cached valid answer.")
                    generated_answer = cached_answers[question_id]
                else:
                    generated_answer = f"ERROR: {error}"

            # Small delay between LLM calls to prevent bursting rate limits
            if LLM_CALL_DELAY > 0:
                time.sleep(LLM_CALL_DELAY)

        print("\nGenerated answer:")
        print(generated_answer)

        # ----------------------------------------------------
        # Answer evaluation (Distinguishing SUCCESS / REFUSAL / GENERATION_ERROR)
        # ----------------------------------------------------

        status, answer_correct, semantic_similarity, evaluation_reason = classify_answer(
            answerable=answerable,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            threshold=SEMANTIC_SIMILARITY_THRESHOLD
        )

        print(f"\nStatus: {status}")

        if status == STATUS_GENERATION_ERROR:
            total_generation_errors += 1
            if answerable:
                answerable_generation_errors += 1
            else:
                unanswerable_generation_errors += 1

            print(f"Evaluation: {evaluation_reason}")
            print("Answer correctness: EXCLUDED (GENERATION_ERROR)")

        else:
            total_evaluated_successfully += 1

            if answerable:
                answerable_evaluated += 1
                if status == STATUS_SUCCESS:
                    answerable_success += 1
                elif status == STATUS_REFUSAL:
                    answerable_refusal += 1
                else:
                    answerable_incorrect += 1

                print(f"Expected answer: {expected_answer}")
                if semantic_similarity is not None:
                    print(f"Semantic similarity: {semantic_similarity:.4f}")
                print(f"Evaluation: {evaluation_reason}")
                print("Answer correctness:", "PASS" if answer_correct else "FAIL")

            else:
                unanswerable_evaluated += 1
                if status == STATUS_SUCCESS:
                    unanswerable_success += 1
                else:
                    unanswerable_incorrect += 1

                print(f"Evaluation: {evaluation_reason}")
                print("Unanswerable handling:", "PASS" if answer_correct else "FAIL")

        # ----------------------------------------------------
        # Store evaluation result
        # ----------------------------------------------------

        result = {
            "id": question_id,
            "question": question,
            "answerable": answerable,
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages,
            "retrieval_hit": retrieval_hit,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "status": status,
            "semantic_similarity": round(semantic_similarity, 4) if semantic_similarity is not None else None,
            "answer_correct": answer_correct,
            "evaluation_reason": evaluation_reason
        }

        results.append(result)

    # ========================================================
    # Calculate final metrics (Excluding GENERATION_ERROR)
    # ========================================================

    retrieval_accuracy = (
        retrieval_hits / retrieval_total * 100
        if retrieval_total > 0
        else 0.0
    )

    answerable_accuracy = (
        answerable_success / answerable_evaluated * 100
        if answerable_evaluated > 0
        else 0.0
    )

    unanswerable_accuracy = (
        unanswerable_success / unanswerable_evaluated * 100
        if unanswerable_evaluated > 0
        else 0.0
    )

    total_evaluated = answerable_evaluated + unanswerable_evaluated
    total_success = answerable_success + unanswerable_success

    overall_quality_accuracy = (
        total_success / total_evaluated * 100
        if total_evaluated > 0
        else 0.0
    )

    # ========================================================
    # Final report
    # ========================================================

    print("\n\n")
    print("=" * 80)
    print("FINAL EVALUATION REPORT")
    print("=" * 80)

    print(f"\nTotal questions in dataset: {total_questions}")
    print(
        f"Questions evaluated successfully: "
        f"{total_evaluated_successfully}/{total_questions} "
        f"({total_evaluated_successfully / total_questions * 100:.1f}%)"
        if total_questions > 0 else ""
    )
    print(
        f"Generation errors (API/Quota excluded): "
        f"{total_generation_errors}/{total_questions} "
        f"({total_generation_errors / total_questions * 100:.1f}%)"
        if total_questions > 0 else ""
    )

    print("\n" + "-" * 80)
    print("RETRIEVAL PERFORMANCE (Hybrid FAISS + BM25)")
    print("-" * 80)
    print(f"Evaluated retrieval questions: {retrieval_total}")
    print(
        f"Hybrid Retrieval Hit@{HYBRID_TOP_K}: "
        f"{retrieval_hits}/{retrieval_total} "
        f"({retrieval_accuracy:.1f}%)"
    )

    print("\n" + "-" * 80)
    print("ANSWER-QUALITY PERFORMANCE (Excluding Generation Errors)")
    print("-" * 80)

    print(f"Answerable questions evaluated: {answerable_evaluated}/{answerable_total}")
    if answerable_generation_errors > 0:
        print(f"  (Excluded {answerable_generation_errors} generation error(s))")
    if answerable_evaluated > 0:
        print(f"  - SUCCESS (Semantic Match): {answerable_success} ({answerable_success / answerable_evaluated * 100:.1f}%)")
        print(f"  - REFUSAL (False Refusal):  {answerable_refusal} ({answerable_refusal / answerable_evaluated * 100:.1f}%)")
        print(f"  - INCORRECT (Semantic Mismatch): {answerable_incorrect} ({answerable_incorrect / answerable_evaluated * 100:.1f}%)")
        print(f"  Answerable Accuracy: {answerable_success}/{answerable_evaluated} ({answerable_accuracy:.1f}%) [Threshold: {SEMANTIC_SIMILARITY_THRESHOLD:.2f}]")
    else:
        print("  Answerable Accuracy: N/A (all answerable questions encountered generation errors)")

    print(f"\nUnanswerable questions evaluated: {unanswerable_evaluated}/{unanswerable_total}")
    if unanswerable_generation_errors > 0:
        print(f"  (Excluded {unanswerable_generation_errors} generation error(s))")
    if unanswerable_evaluated > 0:
        print(f"  - SUCCESS (Correct Refusal): {unanswerable_success} ({unanswerable_success / unanswerable_evaluated * 100:.1f}%)")
        print(f"  - INCORRECT (Hallucination): {unanswerable_incorrect} ({unanswerable_incorrect / unanswerable_evaluated * 100:.1f}%)")
        print(f"  Unanswerable Accuracy: {unanswerable_success}/{unanswerable_evaluated} ({unanswerable_accuracy:.1f}%)")
    else:
        print("  Unanswerable Accuracy: N/A (all unanswerable questions encountered generation errors)")

    print("\n" + "-" * 80)
    print("OVERALL ANSWER-QUALITY SUMMARY")
    print("-" * 80)
    if total_evaluated > 0:
        print(
            f"Overall Quality Accuracy (on {total_evaluated} evaluated): "
            f"{total_success}/{total_evaluated} "
            f"({overall_quality_accuracy:.1f}%)"
        )
    else:
        print("Overall Quality Accuracy: N/A (all questions encountered generation errors)")

    if total_generation_errors > 0:
        print(f"\nNOTE: {total_generation_errors} question(s) were excluded from answer-quality accuracy")
        print("due to LLM generation/quota errors (429 RESOURCE_EXHAUSTED).")
    print("=" * 80)

    # ========================================================
    # Save detailed results
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nDetailed results saved to:\n{OUTPUT_PATH}")
    print("\nEvaluation completed.")
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
