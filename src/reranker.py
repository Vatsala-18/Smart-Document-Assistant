import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


if not os.getenv("GEMINI_API_KEY"):
    raise ValueError(
        "GEMINI_API_KEY is not set"
    )


# ============================================================
# Gemini model used for reranking
# ============================================================

reranker_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# ============================================================
# Rerank retrieved chunks
# ============================================================

def rerank_chunks(
    question,
    retrieved_results,
    top_n=3
):
    """
    Use Gemini to evaluate the retrieved chunks
    and select the most useful ones.
    """

    if not retrieved_results:
        return []


    # --------------------------------------------------------
    # Build candidate text
    # --------------------------------------------------------

    candidates = []


    for i, result in enumerate(
        retrieved_results
    ):

        candidates.append(
            f"""
Candidate {i}

Source:
{result['metadata']['source']}

Page:
{result['metadata']['page']}

FAISS Similarity:
{result['score']:.4f}

Text:
{result['text']}
"""
        )


    candidates_text = "\n\n---\n\n".join(
        candidates
    )


    # --------------------------------------------------------
    # Reranking prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document retrieval evaluator.

Your job is to determine which retrieved document
chunks are actually useful for answering the user's
question.

USER QUESTION:

{question}


RETRIEVED CANDIDATES:

{candidates_text}


IMPORTANT RULES:

1. Select only chunks that contain information
   useful for answering the question.

2. A chunk that merely mentions the question but
   does not provide the answer should NOT be
   considered highly relevant.

3. Do not invent information.

4. If none of the chunks contain useful information,
   return an empty list.

5. Return ONLY valid JSON.

Use this exact format:

[
    {{
        "candidate": 0,
        "relevance": 0.95,
        "reason": "Contains the answer."
    }}
]

The relevance score must be between 0 and 1.

Return at most {top_n} candidates.
"""


    # --------------------------------------------------------
    # Ask Gemini
    # --------------------------------------------------------

    response = reranker_llm.invoke(
        prompt
    )


    content = response.content


    # --------------------------------------------------------
    # Extract text safely
    # --------------------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        content = "\n".join(
            text_parts
        )


    content = content.strip()


    # --------------------------------------------------------
    # Remove Markdown JSON fences if Gemini
    # returns them.
    # --------------------------------------------------------

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        rankings = json.loads(
            content
        )

    except json.JSONDecodeError:

        print(
            "Could not parse reranker response:"
        )

        print(content)

        return retrieved_results[
            :top_n
        ]


    # --------------------------------------------------------
    # Build reranked results
    # --------------------------------------------------------

    reranked_results = []


    for ranking in rankings:

        candidate_index = ranking.get(
            "candidate"
        )

        relevance = ranking.get(
            "relevance",
            0
        )


        if not isinstance(
            candidate_index,
            int
        ):
            continue


        if (
            candidate_index < 0
            or candidate_index >= len(
                retrieved_results
            )
        ):
            continue


        original_result = (
            retrieved_results[
                candidate_index
            ]
        )


        reranked_results.append(
            {
                "text": original_result[
                    "text"
                ],

                "metadata": original_result[
                    "metadata"
                ],

                "score": original_result[
                    "score"
                ],

                "rerank_score": float(
                    relevance
                ),

                "rerank_reason": ranking.get(
                    "reason",
                    ""
                )
            }
        )


    # --------------------------------------------------------
    # Sort by reranker score
    # --------------------------------------------------------

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )


    return reranked_results[
        :top_n
    ]