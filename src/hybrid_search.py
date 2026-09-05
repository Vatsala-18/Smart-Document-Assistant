def hybrid_search(
    faiss_results,
    bm25_results,
    k=5,
    rrf_constant=60
):
    """
    Combine FAISS and BM25 rankings using
    Reciprocal Rank Fusion (RRF).
    """

    combined = {}


    # ========================================================
    # FAISS Results
    # ========================================================

    for rank, result in enumerate(
        faiss_results,
        start=1
    ):

        chunk_id = result[
            "metadata"
        ]["chunk_id"]


        if chunk_id not in combined:

            combined[chunk_id] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "faiss_score": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }


        combined[
            chunk_id
        ]["faiss_score"] = result[
            "score"
        ]


        combined[
            chunk_id
        ]["rrf_score"] += (
            1.0
            /
            (rrf_constant + rank)
        )


    # ========================================================
    # BM25 Results
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = result[
            "metadata"
        ]["chunk_id"]


        if chunk_id not in combined:

            combined[chunk_id] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "faiss_score": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }


        combined[
            chunk_id
        ]["bm25_score"] = result[
            "bm25_score"
        ]


        combined[
            chunk_id
        ]["rrf_score"] += (
            1.0
            /
            (rrf_constant + rank)
        )


    # ========================================================
    # Sort combined results
    # ========================================================

    results = list(
        combined.values()
    )


    results.sort(
        key=lambda result: result[
            "rrf_score"
        ],
        reverse=True
    )


    return results[:k]