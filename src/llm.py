import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# Environment
# ============================================================

load_dotenv()


if not os.getenv("GEMINI_API_KEY"):

    raise ValueError(
        "GEMINI_API_KEY is not set."
    )


# ============================================================
# Gemini LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# ============================================================
# Generate Answer
# ============================================================

def generate_answer(
    question,
    retrieved_results
):
    """
    Generate a grounded answer using ONLY
    the retrieved document context.
    """


    # ========================================================
    # No Context
    # ========================================================

    if not retrieved_results:

        return (
            "The answer is not available "
            "in the uploaded document."
        )


    # ========================================================
    # Build Context
    # ========================================================

    context_parts = []


    for result in retrieved_results:

        source = result[
            "metadata"
        ]["source"]


        page = result[
            "metadata"
        ]["page"]


        text = result[
            "text"
        ]


        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{text}
"""
        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    # ========================================================
    # Grounded RAG Prompt
    # ========================================================

    prompt = f"""
You are a Smart Document Assistant.

Your job is to answer the user's question using
ONLY the information provided in the document context.

STRICT RULES:

1. Use ONLY the provided document context.

2. Do NOT use your general knowledge.

3. Do NOT guess, assume, or invent information.

4. If the answer cannot be found in the provided
   context, respond exactly:

"The answer is not available in the uploaded document."

5. If the document explicitly states that an answer
   is unavailable, do not attempt to infer or guess it.

6. A chunk that merely contains or mentions a question
   is NOT evidence that the answer is present.

7. Only answer when the provided context contains
   information that supports the answer.

8. Keep the answer concise but informative.

9. After the answer, provide the supporting source
   page or pages.

10. Only cite pages that actually support the answer.

11. Never invent page numbers or sources.


DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    # ========================================================
    # Gemini Call
    # ========================================================

    response = llm.invoke(
        prompt
    )


    # ========================================================
    # Extract Response
    # ========================================================

    content = response.content


    # --------------------------------------------------------
    # Normal string response
    # --------------------------------------------------------

    if isinstance(
        content,
        str
    ):

        return content.strip()


    # --------------------------------------------------------
    # Structured response
    # --------------------------------------------------------

    if isinstance(
        content,
        list
    ):

        text_parts = []


        for item in content:

            if isinstance(
                item,
                dict
            ):

                if item.get(
                    "type"
                ) == "text":

                    text_parts.append(
                        item.get(
                            "text",
                            ""
                        )
                    )


            elif isinstance(
                item,
                str
            ):

                text_parts.append(
                    item
                )


        return "\n".join(
            text_parts
        ).strip()


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return str(
        content
    )