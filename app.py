import streamlit as st
import fitz

from src.text_splitter import split_text

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

from src.llm import (
    generate_answer
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# Application Title
# ============================================================

st.title(
    "📄 Smart Document Assistant"
)

st.write(
    "Upload a PDF and ask questions about its contents."
)


# ============================================================
# Upload PDF
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ============================================================
# Process Uploaded PDF
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )


    st.write(
        f"**Number of pages:** {len(document)}"
    )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    all_chunks = []


    for page_number, page in enumerate(
        document
    ):

        page_text = page.get_text()


        # Skip empty pages
        if not page_text.strip():
            continue


        # ----------------------------------------------------
        # Recursive chunking
        # ----------------------------------------------------

        chunks = split_text(
            page_text,
            chunk_size=1000,
            chunk_overlap=200
        )


        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        for chunk in chunks:

            chunk_data = {

                "text": chunk,

                "metadata": {

                    "source":
                        uploaded_file.name,

                    "page":
                        page_number + 1,

                    "chunk_id":
                        len(all_chunks)
                }
            }


            all_chunks.append(
                chunk_data
            )


    # --------------------------------------------------------
    # Display chunk count
    # --------------------------------------------------------

    st.write(
        f"**Number of chunks:** "
        f"{len(all_chunks)}"
    )


    # ========================================================
    # View Document Chunks
    # ========================================================

    with st.expander(
        "🔍 View document chunks"
    ):

        for chunk in all_chunks:

            st.write(
                f"### Chunk "
                f"{chunk['metadata']['chunk_id']}"
            )


            st.write(
                f"**Page:** "
                f"{chunk['metadata']['page']}"
            )


            st.write(
                f"**Source:** "
                f"{chunk['metadata']['source']}"
            )


            st.write(
                chunk["text"]
            )


            st.divider()


    # ========================================================
    # Create Search Indexes
    # ========================================================

    st.divider()


    if st.button(
        "🔨 Create Search Index",
        type="primary"
    ):

        with st.spinner(
            "Creating FAISS and BM25 indexes..."
        ):

            # -----------------------------------------------
            # Dense vector index
            # -----------------------------------------------

            faiss_index = create_faiss_index(
                all_chunks
            )


            # -----------------------------------------------
            # Sparse keyword index
            # -----------------------------------------------

            bm25_index = create_bm25_index(
                all_chunks
            )


        # ----------------------------------------------------
        # Store indexes in Streamlit session
        # ----------------------------------------------------

        st.session_state[
            "faiss_index"
        ] = faiss_index


        st.session_state[
            "bm25_index"
        ] = bm25_index


        st.session_state[
            "chunks"
        ] = all_chunks


        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        st.success(
            f"Search indexes created successfully! "
            f"FAISS vectors: {faiss_index.ntotal} | "
            f"BM25 documents: {len(all_chunks)}"
        )


    # ========================================================
    # Question Answering
    # ========================================================

    if (
        "faiss_index" in st.session_state
        and
        "bm25_index" in st.session_state
    ):

        st.divider()


        st.subheader(
            "💬 Ask a question"
        )


        question = st.text_input(
            "What do you want to know?",
            placeholder=(
                "Example: What is RAG?"
            )
        )


        # ====================================================
        # Process Question
        # ====================================================

        if question:

            # =================================================
            # STEP 1 — FAISS Search
            # =================================================

            with st.spinner(
                "Running semantic search..."
            ):

                faiss_results = search_faiss(

                    st.session_state[
                        "faiss_index"
                    ],

                    st.session_state[
                        "chunks"
                    ],

                    question,

                    k=8,

                    score_threshold=None
                )


            # =================================================
            # STEP 2 — BM25 Search
            # =================================================

            with st.spinner(
                "Running keyword search..."
            ):

                bm25_results = search_bm25(

                    st.session_state[
                        "bm25_index"
                    ],

                    st.session_state[
                        "chunks"
                    ],

                    question,

                    k=8
                )


            # =================================================
            # STEP 3 — Hybrid Fusion
            # =================================================

            retrieved_results = hybrid_search(

                faiss_results,

                bm25_results,

                k=5
            )


            # =================================================
            # FAISS Results
            # =================================================

            with st.expander(
                "🧠 Semantic Search Results (FAISS)"
            ):

                if faiss_results:

                    for rank, result in enumerate(
                        faiss_results
                    ):

                        st.write(
                            f"### FAISS Result "
                            f"{rank + 1}"
                        )


                        st.write(
                            f"**Similarity:** "
                            f"{result['score']:.4f}"
                        )


                        st.write(
                            f"**Page:** "
                            f"{result['metadata']['page']}"
                        )


                        st.write(
                            f"**Chunk ID:** "
                            f"{result['metadata']['chunk_id']}"
                        )


                        st.write(
                            result["text"]
                        )


                        st.divider()


                else:

                    st.info(
                        "No FAISS results."
                    )


            # =================================================
            # BM25 Results
            # =================================================

            with st.expander(
                "🔤 Keyword Search Results (BM25)"
            ):

                if bm25_results:

                    for rank, result in enumerate(
                        bm25_results
                    ):

                        st.write(
                            f"### BM25 Result "
                            f"{rank + 1}"
                        )


                        st.write(
                            f"**BM25 Score:** "
                            f"{result['bm25_score']:.4f}"
                        )


                        st.write(
                            f"**Page:** "
                            f"{result['metadata']['page']}"
                        )


                        st.write(
                            f"**Chunk ID:** "
                            f"{result['metadata']['chunk_id']}"
                        )


                        st.write(
                            result["text"]
                        )


                        st.divider()


                else:

                    st.info(
                        "No BM25 results."
                    )


            # =================================================
            # Hybrid Results
            # =================================================

            st.subheader(
                "🔎 Hybrid Retrieved Context"
            )


            st.info(
                "Hybrid search combines semantic "
                "FAISS retrieval and keyword-based "
                "BM25 retrieval using Reciprocal "
                "Rank Fusion (RRF)."
            )


            if retrieved_results:

                for rank, result in enumerate(
                    retrieved_results
                ):

                    st.write(
                        f"### Hybrid Result "
                        f"{rank + 1}"
                    )


                    st.write(
                        f"**RRF Score:** "
                        f"{result['rrf_score']:.6f}"
                    )


                    if result[
                        "faiss_score"
                    ] is not None:

                        st.write(
                            f"**FAISS Similarity:** "
                            f"{result['faiss_score']:.4f}"
                        )

                    else:

                        st.write(
                            "**FAISS Similarity:** "
                            "Not retrieved"
                        )


                    if result[
                        "bm25_score"
                    ] is not None:

                        st.write(
                            f"**BM25 Score:** "
                            f"{result['bm25_score']:.4f}"
                        )

                    else:

                        st.write(
                            "**BM25 Score:** "
                            "Not retrieved"
                        )


                    st.write(
                        f"**Source:** "
                        f"{result['metadata']['source']}"
                    )


                    st.write(
                        f"**Page:** "
                        f"{result['metadata']['page']}"
                    )


                    st.write(
                        f"**Chunk ID:** "
                        f"{result['metadata']['chunk_id']}"
                    )


                    st.write(
                        result["text"]
                    )


                    st.divider()


            else:

                st.warning(
                    "No relevant context was found."
                )


            # =================================================
            # STEP 4 — Generate Final Answer
            # =================================================

            st.subheader(
                "💡 Answer"
            )


            if retrieved_results:

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        retrieved_results
                    )


                st.write(
                    answer
                )


                # =============================================
                # Sources
                # =============================================

                st.subheader(
                    "📚 Sources"
                )


                sources = set()


                for result in retrieved_results:

                    source = result[
                        "metadata"
                    ]["source"]

                    page = result[
                        "metadata"
                    ]["page"]


                    sources.add(
                        (
                            source,
                            page
                        )
                    )


                for source, page in sorted(
                    sources
                ):

                    st.write(
                        f"📄 **{source}** — "
                        f"Page {page}"
                    )


            else:

                st.warning(
                    "The answer is not available "
                    "in the uploaded document."
                )