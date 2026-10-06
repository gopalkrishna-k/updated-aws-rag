# AWS Overview RAG System

> A production-grade **Retrieval-Augmented Generation (RAG)** microservice built on the official *"Overview of Amazon Web Services"* whitepaper. It combines PostgreSQL hybrid search, hierarchical parent-child chunking, Cross-Encoder reranking, and Gemini LLM generation into a fully-evaluated, API-ready system.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [REST API](#-rest-api)
- [Evaluation System](#-evaluation-system)
- [Retrieval Pipeline](#-retrieval-pipeline)
- [Key Design Decisions](#-key-design-decisions)

---

## 🔍 Overview

This project answers natural language questions about AWS services by grounding every answer strictly in the AWS Overview whitepaper. It uses a multi-stage retrieval pipeline to fetch the most relevant content, then uses Google Gemini to synthesize a concise, cited response. The system also features a full evaluation harness with 6 distinct metrics covering both retrieval quality and generation quality.

**What it can do:**
- Answer questions about 200+ AWS services with inline citations
- Detect service comparisons and expand queries automatically
- Refuse questions not covered by the document (hallucination prevention)
- Evaluate itself against a labeled benchmark dataset
- Serve answers via a REST API with Swagger documentation

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Stage 1: Query Pre-Processing           │
│  • Acronym Expansion (ec2 → "Amazon EC2")            │
│  • Intent Detection (comparison / technical)         │
│  • Query Variation Generation                        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Stage 2: Parallel Hybrid Search             │
│                                                      │
│   Dense Vector Search          Sparse BM25 Search    │
│   (BGE-large-en-v1.5)         (PostgreSQL tsvector)  │
│   top 20 child chunks          top 20 child chunks   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Stage 3A: RRF Fusion (top 15)               │
│  Reciprocal Rank Fusion merges Dense + Sparse lists  │
│  with intent-adaptive weighting                      │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Stage 3B: Cross-Encoder Reranking (top 5)   │
│  ms-marco-MiniLM-L-6-v2 re-scores relevance          │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Stage 4: Parent Context Resolution          │
│  Fetch full parent chunks from PostgreSQL            │
│  by parent_id → assemble prompt context              │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Gemini LLM Generation                   │
│  gemini-3.6-flash with grounded prompt template      │
│  → Cited answer or "Not covered in the document."    │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AWS-RAG/
├── api.py                          # FastAPI REST server
├── main.py                         # CLI entrypoint (REPL, eval, batch eval)
├── datasets_shim.py                # pyarrow DLL shim for Windows environments
├── config.yaml                     # Chunking, model, and DB configuration
├── requirements.txt
│
├── src/
│   ├── config.py                   # AppConfig dataclass + load_config()
│   ├── cli.py                      # Interactive REPL CLI
│   │
│   ├── ingest/                     # PDF ingestion pipeline
│   │   ├── parse_pdf.py            # PDF → Markdown (PyMuPDF)
│   │   ├── ingestion.py            # Ingestion orchestrator
│   │   ├── extract_images.py       # Image extraction & captioning
│   │   ├── extract_tables.py       # Table extraction
│   │   └── validate_text_quality.py
│   │
│   ├── chunking/                   # Parent-Child chunking
│   │   ├── parent_child.py         # Hierarchical chunking strategy
│   │   ├── chunking.py
│   │   └── build_chunks.py
│   │
│   ├── indexing/                   # Vector indexing
│   │   ├── build_vectorstore.py    # BGEEmbeddings + PostgreSQL indexing
│   │   └── embed_store.py
│   │
│   ├── retrieval/
│   │   ├── retrieval_service.py    # ★ Multi-stage hybrid retrieval pipeline
│   │   ├── postgres_retriever.py   # PostgreSQL + pgvector retriever
│   │   └── retriever.py            # LangChain retriever wrapper
│   │
│   └── generation/
│       └── rag_chain.py            # ★ LCEL RAG chain + Gemini LLM generation
│
├── backend/
│   ├── services/
│   │   ├── eval_service.py         # ★ 6-metric evaluation engine
│   │   └── retrieval_service.py    # Backend retrieval alias
│   └── utils/
│       └── key_rotator.py          # ★ Thread-safe round-robin API key rotator
│
├── data/
│   ├── raw/aws-overview.pdf        # Source document
│   ├── processed/aws-overview.md   # Parsed Markdown
│   ├── chunks/                     # parent_chunks.jsonl, child_chunks.jsonl
│   └── evaluation_dataset.json     # Benchmark dataset (with expected_parent_ids)
│
├── ground truth/
│   └── evaluation_dataset.json     # ★ Primary ground truth file (user-editable)
│
└── vectorstore/                    # Chroma + BM25 index artifacts
```

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| **LLM** | Google Gemini (`gemini-3.6-flash`) via `langchain-google-genai` |
| **Embeddings** | `BAAI/bge-large-en-v1.5` via `sentence-transformers` |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` (Cross-Encoder) |
| **Vector DB** | PostgreSQL + `pgvector` extension |
| **Sparse Search** | PostgreSQL `tsvector` (BM25-style full-text search) |
| **RAG Framework** | LangChain (LCEL chain) |
| **API Framework** | FastAPI + Uvicorn |
| **Chunking** | Hierarchical Parent-Child (`RecursiveCharacterTextSplitter`) |
| **PDF Parsing** | PyMuPDF / pymupdf4llm |
| **Evaluation** | Custom LLM-as-a-Judge + Deterministic IR metrics |

---

## ✅ Prerequisites

- Python 3.11+
- PostgreSQL 14+ with the **`pgvector`** extension installed
- Google Gemini API Key(s) — set in `.env`
- (Optional) Groq API Key — for fallback LLM evaluation

---

## ⚙️ Setup & Installation

### 1. Clone and create virtual environment

```powershell
git clone https://github.com/gopalkrishna-k/updated-aws-rag.git
cd AWS-RAG
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Comma-separated list of Gemini API keys (rotated automatically)
GEMINI_API_KEYS=your_key_1,your_key_2,your_key_3

# Optional: Groq API key for fallback
GROQ_API_KEYS=your_groq_key
```

### 4. Set up PostgreSQL

```sql
-- Run in psql
CREATE DATABASE aws_rag_db;
\c aws_rag_db
CREATE EXTENSION IF NOT EXISTS vector;
```

Update `config.yaml` with your database credentials (see [Configuration](#-configuration)).

### 5. Ingest the PDF and build the vector index

```powershell
# Parse PDF → Markdown
python -m src.ingest.parse_pdf

# Build parent-child chunks and store to PostgreSQL
python -m src.indexing.build_vectorstore
```

---

## 🔧 Configuration

All settings live in [`config.yaml`](config.yaml):

```yaml
# LLM & Embeddings
embedding_model: BAAI/bge-large-en-v1.5
llm_model: gemini-3.6-flash

# Parent-Child Chunking
parent_chunk_size: 2000      # Characters per parent chunk
parent_chunk_overlap: 100
child_chunk_size: 400        # Characters per child chunk (indexed for search)
child_chunk_overlap: 50

# PostgreSQL
postgres_host: localhost
postgres_port: 5432
postgres_db: aws_rag_db
postgres_user: postgres
postgres_password: "your_password"

# Retrieval
top_k: 5           # Final chunks sent to LLM after reranking
rerank_top_n: 5    # Cross-Encoder top-N selection
```

---

## 🚀 Running the Application

### Interactive Q&A REPL

```powershell
python main.py
```

Type any AWS question at the prompt. Type `exit` or `quit` to stop.

### Single-Query Evaluation

```powershell
python main.py --eval
```

Type a question — if it matches a question in the ground truth dataset, the system evaluates all 6 metrics and prints a scorecard.

### Batch Evaluation (all ground truth questions)

```powershell
python main.py --eval-batch

# With a custom dataset path:
python main.py --eval-batch --dataset data/my_dataset.json
```

### REST API Server

```powershell
python api.py
```

Server starts at `http://127.0.0.1:8000`. Swagger UI at `http://127.0.0.1:8000/docs`.

---

## 🌐 REST API

### `GET /health`

Quick liveness check.

**Response:**
```json
{ "status": "healthy" }
```

### `POST /query`

Run a question through the full RAG pipeline.

**Request body:**
```json
{
  "question": "What is Amazon S3 and what storage classes does it offer?"
}
```

**Response:**
```json
{
  "question": "What is Amazon S3 and what storage classes does it offer?",
  "answer": "Amazon S3 is an object storage service... (Amazon S3, page 41)",
  "status": "success"
}
```

**Swagger UI:** `http://127.0.0.1:8000/docs`

---

## 📊 Evaluation System

The system evaluates RAG quality across **6 metrics** using both deterministic algorithms and an LLM-as-a-Judge approach (Gemini):

| # | Metric | Type | What it measures |
|---|---|---|---|
| 1 | **MRR** (Mean Reciprocal Rank) | Deterministic | Position of first relevant parent chunk |
| 2 | **NDCG@5** | Deterministic | Graded relevance across top-5 retrieved chunks |
| 3 | **Context Precision** | LLM Judge | Signal-to-noise ratio of retrieved context |
| 4 | **Context Recall** | LLM Judge | Coverage of all ground-truth facts in context |
| 5 | **Faithfulness** | LLM Judge | Absence of hallucinations vs. retrieved context |
| 6 | **Answer Correctness** | LLM Judge | Factual alignment with reference ground truth |

**Scoring:** Arithmetic mean across all metrics, displayed at 4-decimal precision.

**Pass Threshold:** ≥ `0.8500 / 1.0000` → `PASSED`

**Batch scorecard example:**
```
╔══════════════════════════════════════════════════╗
║      SYSTEM-WIDE BATCH EVALUATION RESULTS        ║
╠══════════════════════════════════════════════════╣
║  MRR                    :  0.8750 / 1.0000       ║
║  NDCG@5                 :  0.8320 / 1.0000       ║
║  Context Precision      :  0.8100 / 1.0000       ║
║  Context Recall         :  0.7900 / 1.0000       ║
║  Faithfulness           :  0.9100 / 1.0000       ║
║  Answer Correctness     :  0.8450 / 1.0000       ║
╠══════════════════════════════════════════════════╣
║  OVERALL SYSTEM SCORE   :  0.8437 / 1.0000       ║
║  VERDICT                :  PARTIAL PASS          ║
╚══════════════════════════════════════════════════╝
```

### Ground Truth Dataset

The benchmark lives at `ground truth/evaluation_dataset.json`. Each record has:

```json
{
  "question": "What is Amazon ECS?",
  "ground_truth": "Amazon ECS is a fully managed container orchestration service...",
  "expected_parent_ids": ["uuid-of-relevant-parent-chunk"]
}
```

> You can manually add/remove questions from the ground truth file. The system always reads from `ground truth/evaluation_dataset.json` as the source of truth.

---

## 🔑 Key Design Decisions

### 1. Parent-Child Chunking
Small child chunks (400 chars) are indexed for precise retrieval. When a child matches, its full **parent chunk** (2000 chars) is fetched and sent to the LLM — giving the LLM rich context without polluting the search index with long passages.

### 2. Hybrid Search (Dense + Sparse)
- **Dense search** (BGE embeddings + pgvector) captures semantic similarity.
- **Sparse search** (tsvector BM25) captures exact keyword matches — critical for service names like "Amazon Kinesis" or acronyms like "MWAA".
- **RRF fusion** combines both ranked lists optimally.

### 3. API Key Rotation
Multiple Gemini API keys are cycled via round-robin (`backend/utils/key_rotator.py`). On `429 RESOURCE_EXHAUSTED` or `403 PERMISSION_DENIED`, the rotator automatically tries the next key with a 1.5s backoff — preventing evaluation job failures.

### 4. Hallucination Prevention
The LLM prompt enforces closed-corpus answering. If the retrieved context doesn't cover the question, the model must respond with exactly: `"This isn't covered in the document."` — no guessing or hedging allowed.
