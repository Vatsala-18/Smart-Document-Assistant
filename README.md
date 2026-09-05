# Smart Document Assistant

A **Retrieval-Augmented Generation (RAG)** based document assistant that allows users to upload PDF documents, ask questions about their content, and receive context-grounded answers.

This project is being developed as a practical learning project to understand **Python, embeddings, vector search, hybrid retrieval, RAG, LLMs, and RAG evaluation**.

---

## 🚀 Features

* Upload and process PDF documents
* Extract text from PDF pages
* Split documents into smaller chunks
* Generate embeddings using Google Gemini
* Semantic search using FAISS
* Keyword search using BM25
* Hybrid retrieval using Reciprocal Rank Fusion (RRF)
* Generate answers using Google Gemini
* Preserve document source and page metadata
* Evaluate retrieval and answer quality
* Handle questions whose answers are not available in the document

---

## 🏗️ Architecture

```text
                         PDF Document
                              │
                              ▼
                           PyMuPDF
                              │
                              ▼
                        Text Extraction
                              │
                              ▼
                         Text Chunking
                              │
                              ▼
                     Gemini Embeddings
                              │
                              ▼
                            FAISS
                     Semantic Retrieval
                              │
                              │
User Question ────────────────┤
                              │
                              ▼
                            BM25
                     Keyword Retrieval
                              │
                              ▼
                     Hybrid Retrieval
                            RRF
                              │
                              ▼
                       Top-K Chunks
                              │
                              ▼
                       Gemini LLM
                              │
                              ▼
                     Grounded Answer
```

---

## 🔄 RAG Pipeline

The current pipeline works as follows:

1. Upload a PDF document.
2. Extract text page by page using PyMuPDF.
3. Split the text into smaller chunks.
4. Attach metadata such as source, page number, and chunk ID.
5. Generate embeddings for document chunks using Gemini.
6. Store the embeddings in FAISS.
7. Create a BM25 index for keyword-based retrieval.
8. Convert the user's question into an embedding.
9. Retrieve semantically similar chunks using FAISS.
10. Retrieve keyword-relevant chunks using BM25.
11. Combine the retrieval rankings using Reciprocal Rank Fusion (RRF).
12. Select the top retrieved chunks.
13. Send the retrieved context to Gemini.
14. Generate an answer grounded in the retrieved document content.

---

## 🔍 Retrieval

### FAISS Semantic Search

FAISS is used for vector similarity search.

Document chunks and user questions are converted into embeddings. FAISS then retrieves chunks that are semantically similar to the question.

This allows queries to find relevant information even when the exact words used in the question are different from the words in the document.

### BM25 Keyword Search

BM25 provides keyword-based retrieval.

It is useful for queries containing:

* Exact terms
* Technical terminology
* Names
* Identifiers
* Error messages

### Hybrid Search

The project combines FAISS and BM25 results using **Reciprocal Rank Fusion (RRF)**.

```text
FAISS Semantic Search
          +
BM25 Keyword Search
          │
          ▼
          RRF
          │
          ▼
   Hybrid Ranked Results
```

Using both approaches helps combine semantic understanding with exact keyword matching.

---

## 🧠 Embeddings

The project currently uses:

```text
Embedding Model: gemini-embedding-001
Embedding Dimension: 768
```

Document chunks and queries use the same embedding model and vector dimension to ensure compatibility during similarity search.

---

## 📊 Evaluation

The project includes an evaluation dataset containing **26 questions** covering:

* Basic questions
* Semantic search
* Reasoning
* RAG concepts
* Specific information
* Multi-document scenarios
* Unanswerable questions

---

## 🛠️ Technology Stack

| Component            | Technology                       |
| -------------------- | -------------------------------- |
| Programming Language | Python                           |
| User Interface       | Streamlit                        |
| PDF Processing       | PyMuPDF                          |
| Embeddings           | Google Gemini                    |
| Vector Search        | FAISS                            |
| Keyword Search       | BM25                             |
| Hybrid Retrieval     | Reciprocal Rank Fusion           |
| LLM                  | Google Gemini                    |
| Evaluation           | Custom Python Evaluation Scripts |

---

## 🎯 Learning Objectives

This project is designed to provide practical experience with:

* Python
* Document processing
* Text chunking
* Embeddings
* Vector databases
* Semantic search
* Keyword search
* Hybrid retrieval
* Reciprocal Rank Fusion
* Retrieval-Augmented Generation
* Prompt engineering
* Hallucination prevention
* RAG evaluation
* Answer correctness evaluation

---

## 🔮 Planned Improvements

Future improvements include:

* Improve answer correctness
* Improve handling of unanswerable questions
* Add retrieval reranking
* Add answerability detection
* Experiment with different chunking strategies
* Improve evaluation methodology
* Support multiple documents
* Add conversational memory
* Optimize RAG latency
* Improve retrieval quality
* Explore production-oriented RAG architecture

---

## 📌 Current Status

**Status: RAG baseline implementation and evaluation**

The core RAG pipeline is currently implemented with:

```text
PDF
 ↓
Chunking
 ↓
Gemini Embeddings
 ↓
FAISS + BM25
 ↓
RRF Hybrid Retrieval
 ↓
Gemini
 ↓
Grounded Answer
```


## 📚 Project Purpose

This repository is primarily a hands-on learning project focused on understanding how modern RAG systems work internally, from document ingestion and embeddings to retrieval, generation, and evaluation.

The implementation will evolve as additional RAG techniques and optimizations are explored.
