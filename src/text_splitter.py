from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def split_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split text into semantically reasonable chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)