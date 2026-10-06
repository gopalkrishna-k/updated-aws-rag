# AWS Overview RAG — Technical Documentation

> **Version:** 1.0 | **Last Updated:** October 2026  
> **GitHub:** https://github.com/gopalkrishna-k/updated-aws-rag

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Reference](#2-module-reference)
   - 2.1 [Entry Points](#21-entry-points)
   - 2.2 [Ingestion Pipeline](#22-ingestion-pipeline)
   - 2.3 [Chunking Module](#23-chunking-module)
   - 2.4 [Indexing Module](#24-indexing-module)
   - 2.5 [Retrieval Pipeline](#25-retrieval-pipeline)
   - 2.6 [Generation Chain](#26-generation-chain)
   - 2.7 [Evaluation Service](#27-evaluation-service)
   - 2.8 [API Key Rotator](#28-api-key-rotator)
   - 2.9 [REST API](#29-rest-api)
3. [Database Schema](#3-database-schema)
4. [Retrieval Pipeline — Deep Dive](#4-retrieval-pipeline--deep-dive)
5. [Evaluation System — Deep Dive](#5-evaluation-system--deep-dive)
6. [Configuration Reference](#6-configuration-reference)
7. [Environment Variables](#7-environment-variables)
8. [Data Formats](#8-data-formats)
9. [Error Handling & Known Issues](#9-error-handling--known-issues)
10. [Extending the System](#10-extending-the-system)

---

## 1. System Overview

The AWS Overview RAG System is a **Retrieval-Augmented Generation** application that answers questions strictly from the AWS Overview whitepaper (`data/raw/aws-overview.pdf`). It is designed as a production-ready microservice with a REST API, CLI interface, and a full evaluation harness.

### Data Flow (End-to-End)

```
PDF Document
  │
  ▼  [src/ingest/parse_pdf.py]
Markdown Text
  │
  ▼  [src/chunking/parent_child.py]
Parent Chunks (2000 chars) + Child Chunks (400 chars)
  │
  ▼  [src/indexing/build_vectorstore.py]
PostgreSQL: parent_chunks table + child_chunks table
           (child_chunks include dense vector column via pgvector)
  │
  ▼ ── At Query Time ──────────────────────────────────
  │
  ▼  [src/retrieval/retrieval_service.py]
Stage 1: Query preprocessing + intent routing
Stage 2: Dense vector search + BM25 sparse search → top 20+20 child chunks
Stage 3A: RRF Fusion → top 15 child chunks
Stage 3B: Cross-Encoder Reranking → top 5 child chunks
Stage 4: parent_id resolution → full parent context text
  │
  ▼  [src/generation/rag_chain.py]
Gemini LLM (gemini-3.6-flash) + Grounded Prompt Template
  │
  ▼
Cited Answer (or refusal: "This isn't covered in the document.")
```

---

## 2. Module Reference

### 2.1 Entry Points

#### `main.py` — CLI Entrypoint

The primary entry point supporting three run modes via `argparse`.

**CLI Flags:**

| Flag | Description |
|---|---|
| *(none)* | Launch interactive Q&A REPL |
| `--eval` | Launch single-query evaluation mode (matches against ground truth) |
| `--eval-batch` | Run batch evaluation across all ground truth records |
| `--dataset <path>` | Override the default ground truth dataset path |

**Key Functions:**

| Function | Description |
|---|---|
| `format_scorecard(eval_res)` | Formats single-query evaluation results into a terminal scorecard |
| `format_batch_scorecard(results)` | Formats batch aggregate scores into a double-box scorecard |
| `run_batch_evaluation(dataset_path)` | Orchestrates the full batch evaluation loop |

---

#### `api.py` — FastAPI REST Server

Exposes the RAG pipeline as a REST microservice.

**Starting the server:**
```powershell
python api.py
# Server: http://127.0.0.1:8000
# Docs:   http://127.0.0.1:8000/docs
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check → `{"status": "healthy"}` |
| `POST` | `/query` | Run RAG pipeline for a question |

**`POST /query` Request Schema:**
```json
{ "question": "string" }
```

**`POST /query` Response Schema:**
```json
{
  "question": "string",
  "answer": "string",
  "status": "success"
}
```

> **Note:** `api.py` imports `datasets_shim` at the top before any other project imports. This is required on Windows machines where pyarrow DLLs may be blocked by Application Control policies.

---

#### `datasets_shim.py` — pyarrow DLL Workaround

A compatibility shim that must be imported before `sentence_transformers` in environments where Windows Application Control blocks `pyarrow._compute.pyd`.

**What it does:**
- Probes whether `pyarrow` can be imported.
- If it fails, injects a stub `datasets` module into `sys.modules` with `__version__ = "0.0.0"`.
- `sentence_transformers` uses `datasets` only for training helpers (sampler/trainer). Inference (`encode()`, `cross_encode()`) works correctly with the stub.

**Usage:**
```python
import datasets_shim  # Must be first import in main.py and api.py
```

---

### 2.2 Ingestion Pipeline

**Location:** `src/ingest/`

| File | Purpose |
|---|---|
| `parse_pdf.py` | Converts `aws-overview.pdf` to Markdown using PyMuPDF + pymupdf4llm |
| `ingestion.py` | Orchestrates the full ingestion pipeline |
| `extract_images.py` | Extracts and optionally captions images from the PDF |
| `extract_tables.py` | Extracts table content |
| `validate_text_quality.py` | Sanity-checks extracted text for quality |

**Output:** `data/processed/aws-overview.md`

---

### 2.3 Chunking Module

**Location:** `src/chunking/`

#### `parent_child.py` — Hierarchical Chunking

Implements a **Parent-Child** chunking strategy:

- **Parent Chunks** — Larger, semantically complete passages (default: 2000 chars, 100 overlap). These are what the LLM reads.
- **Child Chunks** — Smaller, search-optimized segments derived from parents (default: 400 chars, 50 overlap). These are what gets indexed and searched.

Every child chunk stores the UUID of its parent (`parent_id`) so the retrieval pipeline can look up the full parent after a child matches.

**Chunk types created:**
- `service` — Individual AWS service descriptions
- `concept` — Core AWS concepts and definitions
- `table` — Tabular comparisons extracted from the whitepaper
- `notice` — Important notices and callout boxes
- `image_caption` — Captions from figures and diagrams

**Key functions:**

| Function | Description |
|---|---|
| `build_parent_child_chunks(markdown_path)` | Main entry — returns `(parent_chunks, child_chunks)` |
| `create_parent_chunks(text)` | Splits text into parent-sized chunks |
| `create_child_chunks(parent_chunk)` | Splits a parent into child-sized chunks with `parent_id` reference |

---

### 2.4 Indexing Module

**Location:** `src/indexing/`

#### `build_vectorstore.py`

- Loads parent and child chunks from `data/chunks/`.
- Generates dense embeddings for all child chunks using `BGEEmbeddings` (`BAAI/bge-large-en-v1.5`).
- Stores parent chunks in PostgreSQL `parent_chunks` table.
- Stores child chunks + embeddings in PostgreSQL `child_chunks` table (with pgvector column).
- Also builds tsvector index for BM25 sparse search.

**`BGEEmbeddings` class:**
- Wraps `sentence_transformers.SentenceTransformer('BAAI/bge-large-en-v1.5')`.
- Embedding dimension: **1024**.
- Prefixes queries with `"Represent this sentence: "` for improved retrieval performance.

---

### 2.5 Retrieval Pipeline

**Location:** `src/retrieval/retrieval_service.py`

See [Section 4](#4-retrieval-pipeline--deep-dive) for a complete deep-dive.

**Key class:** `RetrievalService`

**Key method:** `retrieve(user_query: str, top_k: int = 5) -> dict`

**Returns:**
```python
{
    "user_query": str,
    "intent": dict,                    # Intent flags and weights
    "dense_candidates_count": int,
    "sparse_candidates_count": int,
    "fused_candidates_count": int,
    "top_5_children": list[dict],      # Child chunks after reranking
    "resolved_parents": list[dict],    # Full parent chunks
    "formatted_context": str,          # Assembled prompt context
}
```

---

### 2.6 Generation Chain

**Location:** `src/generation/rag_chain.py`

#### `RAGChain` class

Wraps the LCEL (LangChain Expression Language) pipeline:

```
RetrievalService.retrieve()
    → Grounded PromptTemplate
    → ChatGoogleGenerativeAI (gemini-3.6-flash)
    → StrOutputParser
```

**Key method:** `generate(query: str) -> dict`

**Returns:**
```python
{
    "answer": str,          # LLM-generated cited answer
    "citations": list[str], # Extracted citation labels
    "chunks_used": int,     # Number of parent chunks in context
}
```

#### `generate_answer(query: str) -> dict`

Module-level convenience function that instantiates `RAGChain` and calls `generate()`. Used by `api.py`.

#### Grounded Prompt Template

The LLM is instructed with four hard rules:
1. **Closed-corpus**: Only use information from the retrieved context.
2. **Strict refusal**: If the context doesn't cover the question, respond with exactly: `"This isn't covered in the document."`
3. **Inline citations**: Every factual claim must be followed by its citation label (e.g., `(Amazon S3, page 41)`).
4. **Conciseness**: Direct, factual, no hedging.

#### API Key Rotation in Generation

`rag_chain.py` calls `key_rotator.get_next_key("gemini")` each time it instantiates `ChatGoogleGenerativeAI`, rotating through all configured keys to distribute load and avoid `429` quota errors.

---

### 2.7 Evaluation Service

**Location:** `backend/services/eval_service.py`

See [Section 5](#5-evaluation-system--deep-dive) for a complete deep-dive.

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `evaluate_single_query` | `(question, ground_truth, top_k) -> dict` | Evaluate one Q&A pair across all 6 metrics |
| `evaluate_batch_dataset` | `(dataset_path) -> dict` | Run batch evaluation and return aggregate scores |
| `load_ground_truth_dataset` | `(path) -> list[dict]` | Load and validate the evaluation dataset |
| `mrr_by_parent_id` | `(retrieved_ids, expected_ids) -> float` | Deterministic MRR calculation |
| `ndcg_at_k_by_parent_id` | `(retrieved_ids, expected_ids, k) -> float` | Deterministic NDCG@k calculation |

---

### 2.8 API Key Rotator

**Location:** `backend/utils/key_rotator.py`

#### `APIKeyRotator` class

Thread-safe round-robin key rotator for Gemini and Groq API keys.

**Key methods:**

| Method | Description |
|---|---|
| `get_keys(provider)` | Returns list of keys from environment for `"gemini"` or `"groq"` |
| `get_next_key(provider)` | Returns the next key in round-robin order (thread-safe) |
| `execute_with_retry(fn, provider, max_retries)` | Calls `fn(api_key)`, rotating keys and retrying on `429`/`403` errors |

**Retry Behavior:**
- `429 RESOURCE_EXHAUSTED` → rotate key + sleep 1.5s
- `403 PERMISSION_DENIED` → rotate key + retry immediately
- `401 UNAUTHORIZED` → rotate key + retry immediately
- Max retries: `max(len(keys) × 3, 5)`

**Singleton:** `key_rotator = APIKeyRotator()` — imported and shared across all modules.

**Reading keys from `.env`:**
```
GEMINI_API_KEYS=key1,key2,key3,key4,key5
GROQ_API_KEYS=groq_key1
```

---

### 2.9 REST API

**Location:** `api.py`

**Framework:** FastAPI + Uvicorn

**Auto-generated docs:** Swagger UI at `/docs`, ReDoc at `/redoc`.

**Request/Response models** (Pydantic):

```python
class QueryRequest(BaseModel):
    question: str

# Response (inline dict):
{
    "question": str,
    "answer": str,
    "status": "success"
}
```

**Error responses:**
- `400 Bad Request` — empty question string
- `500 Internal Server Error` — pipeline failure (logged server-side)

---

## 3. Database Schema

**Database:** PostgreSQL (`aws_rag_db`)  
**Extension:** `pgvector`

### `parent_chunks` table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID PRIMARY KEY` | Unique parent chunk identifier |
| `doc_id` | `TEXT` | Source document identifier |
| `category` | `TEXT` | Chunk type (`service`, `concept`, `table`, `notice`, `image_caption`) |
| `service_name` | `TEXT` | AWS service name (if applicable) |
| `text_content` | `TEXT` | Full parent chunk text (sent to LLM) |
| `metadata` | `JSONB` | Additional metadata (page numbers, etc.) |

### `child_chunks` table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID PRIMARY KEY` | Unique child chunk identifier |
| `parent_id` | `UUID REFERENCES parent_chunks(id)` | Parent chunk reference |
| `doc_id` | `TEXT` | Source document identifier |
| `text_content` | `TEXT` | Child chunk text (indexed for search) |
| `embedding` | `VECTOR(1024)` | BGE-large-en-v1.5 dense embedding |
| `ts_content` | `TSVECTOR` | Full-text search index (BM25) |
| `metadata` | `JSONB` | Chunk position and source metadata |

**Key indexes:**
- `child_chunks.embedding` — HNSW index via pgvector for ANN search
- `child_chunks.ts_content` — GIN index for full-text search

---

## 4. Retrieval Pipeline — Deep Dive

**File:** `src/retrieval/retrieval_service.py`

### Stage 1: Query Pre-Processing & Intent Routing

**Function:** `preprocess_query(user_query: str) -> dict`

Three intent types are detected:

| Intent Flag | Detection Logic | Effect |
|---|---|---|
| `is_comparison` | Keywords: `vs`, `compare`, `difference`, `versus` | Increases top_k to 30; balanced vector/BM25 weights |
| `is_technical` | Keywords: `api`, `cli`, `parameter`, `limit`, code patterns | Increases BM25 weight (exact keyword matching) |
| `is_summary` | Keywords: `list`, `all`, `overview`, `types of` | Increases vector weight (semantic understanding) |

**Acronym expansion:** Common AWS acronyms are expanded before search (e.g., `ec2` → `"Amazon Elastic Compute Cloud (EC2)"`). This dramatically improves recall for abbreviated queries.

**Query variations:** Up to 2 alternative phrasings are generated and searched in parallel, with results merged before RRF.

### Stage 2: Parallel Hybrid Search

**Function:** `hybrid_search_child_chunks(queries, dense_limit, sparse_limit) -> (dense_list, sparse_list)`

**Dense Vector Search (PostgreSQL + pgvector):**
- Query is embedded with `SentenceTransformer('BAAI/bge-large-en-v1.5')`.
- ANN search using HNSW index against `child_chunks.embedding` (cosine similarity).
- Returns top 20 child chunks per query (30 for comparison queries).

**Sparse BM25 Search (PostgreSQL tsvector):**
- Query is tokenized and searched against `child_chunks.ts_content` using `plainto_tsquery`.
- Ranked by `ts_rank_cd` (cover density ranking).
- Returns top 20 child chunks per query (30 for comparison queries).

**Chunk counts at each stage:**

```
Dense Search:   up to 20 child chunks  ─┐
                                         ├→ Combined pool (up to 40+ unique chunks)
Sparse Search:  up to 20 child chunks  ─┘
```

### Stage 3A: Reciprocal Rank Fusion (RRF)

**Function:** `reciprocal_rank_fusion(dense_list, sparse_list, dense_weight, sparse_weight, top_n=15) -> list`

RRF formula for each chunk:

```
score(chunk) = dense_weight × (1 / (k + rank_dense))
             + sparse_weight × (1 / (k + rank_sparse))
```

Where `k = 60` (standard RRF constant).

**Intent-adaptive weights:**

| Query Type | Dense Weight | BM25 Weight |
|---|---|---|
| Technical | 0.4 | 0.6 |
| Comparison | 0.5 | 0.5 |
| Summary/default | 0.6 | 0.4 |

**Output:** Top 15 child chunks by RRF score.

### Stage 3B: Cross-Encoder Reranking

**Function:** `rerank_cross_encoder(query, candidates, top_n=5) -> list`

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

The Cross-Encoder jointly encodes the query and each candidate chunk, producing a relevance score that is more accurate than embedding similarity alone.

- **Input:** 15 child chunks from RRF
- **Output:** Top 5 child chunks, re-scored by cross-encoder relevance

### Stage 4: Parent Context Resolution

**Function:** `resolve_parent_contexts(top_children) -> (resolved_parents, formatted_context)`

1. Extracts `parent_id` from each of the top 5 child chunks.
2. Deduplicates `parent_id` values (multiple children can share a parent).
3. Fetches full `text_content` from `parent_chunks` table for each unique `parent_id`.
4. Assembles a formatted context string with citation headers.

**Formatted context example:**
```
[Source 1: Amazon S3, page 41]
Amazon S3 is an object storage service offering industry-leading scalability...

[Source 2: Amazon S3 Storage Classes, page 42]
Amazon S3 offers a range of storage classes designed for different use cases...
```

---

## 5. Evaluation System — Deep Dive

**File:** `backend/services/eval_service.py`

### Ground Truth Dataset Format

Located at `ground truth/evaluation_dataset.json`:

```json
[
  {
    "question": "What is Amazon ECS?",
    "ground_truth": "Amazon ECS is a fully managed container orchestration service...",
    "expected_parent_ids": ["3f2a4b1c-...", "7d8e9f0a-..."]
  }
]
```

- `question` — Natural language question
- `ground_truth` — Reference answer (used for Answer Correctness + Context Recall)
- `expected_parent_ids` — List of PostgreSQL UUIDs of parent chunks that should be retrieved (used for MRR and NDCG@5)

### Metric Implementations

#### 1. Reciprocal Rank (MRR) — Deterministic

```python
def mrr_by_parent_id(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in expected_ids:
            return 1.0 / rank
    return 0.0
```

- `retrieved_ids` = ordered list of `parent_id` values from Stage 4
- `expected_ids` = `expected_parent_ids` from ground truth
- Returns `1/rank` of the first hit, or `0.0` if no hit

#### 2. NDCG@5 — Deterministic

```python
def ndcg_at_k_by_parent_id(retrieved_ids, expected_ids, k=5) -> float:
    # Binary relevance: 1 if parent_id in expected_ids, else 0
    gains = [1.0 if pid in expected_ids else 0.0 for pid in retrieved_ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    # Ideal DCG: all relevant items at top positions
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

#### 3–6. LLM-as-a-Judge Metrics

**Function:** `LLMJudgeEvaluator._evaluate_prompt(prompt: str) -> float`

A custom LLM judge sends structured prompts to Gemini and parses a numeric score between 0.0 and 1.0.

**Context Precision Prompt Structure:**
- Input: Retrieved context + Question
- Asks: "What fraction of the retrieved context is relevant to answering this question?"
- Output: Float score 0.0–1.0

**Context Recall Prompt Structure:**
- Input: Retrieved context + Ground truth answer
- Asks: "What fraction of the ground truth statements are covered by the retrieved context?"
- Output: Float score 0.0–1.0

**Faithfulness Prompt Structure:**
- Input: Generated answer + Retrieved context
- Asks: "What fraction of claims in the generated answer are directly supported by the context?"
- Output: Float score 0.0–1.0

**Answer Correctness Prompt Structure:**
- Input: Generated answer + Ground truth answer
- Asks: "How semantically similar and factually consistent is the generated answer with the reference?"
- Output: Float score 0.0–1.0

All LLM judge calls go through `key_rotator.execute_with_retry()` for automatic key rotation on rate limits.

### Batch Evaluation Output

```python
{
    "total_queries": int,
    "mean_mrr": float,
    "mean_ndcg": float,
    "mean_precision": float,
    "mean_recall": float,
    "mean_faithfulness": float,
    "mean_correctness": float,
    "overall_score": float,      # Arithmetic mean of all 6
    "verdict": str,              # "PASSED" / "PARTIAL PASS" / "NEEDS IMPROVEMENT"
    "per_query_results": list,   # Detailed results for each question
}
```

**Verdict thresholds:**
- `overall_score >= 0.85` → `PASSED`
- `0.70 <= overall_score < 0.85` → `PARTIAL PASS`
- `overall_score < 0.70` → `NEEDS IMPROVEMENT`

---

## 6. Configuration Reference

**File:** `config.yaml`

| Key | Default | Description |
|---|---|---|
| `embedding_model` | `BAAI/bge-large-en-v1.5` | HuggingFace model for dense embeddings |
| `llm_model` | `gemini-3.6-flash` | Gemini model identifier for generation |
| `parent_chunk_size` | `2000` | Max characters per parent chunk |
| `parent_chunk_overlap` | `100` | Character overlap between parent chunks |
| `child_chunk_size` | `400` | Max characters per child chunk |
| `child_chunk_overlap` | `50` | Character overlap between child chunks |
| `postgres_host` | `localhost` | PostgreSQL host |
| `postgres_port` | `5432` | PostgreSQL port |
| `postgres_db` | `aws_rag_db` | PostgreSQL database name |
| `postgres_user` | `postgres` | PostgreSQL username |
| `postgres_password` | — | PostgreSQL password |
| `vectorstore_path` | `vectorstore` | Path for Chroma/BM25 artifacts |
| `top_k` | `5` | Final number of parent chunks sent to LLM |
| `rerank_top_n` | `5` | Number of chunks selected by Cross-Encoder |

---

## 7. Environment Variables

**File:** `.env` (project root)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEYS` | ✅ Yes | Comma-separated Gemini API keys |
| `GEMINI_API_KEY` | Fallback | Single Gemini API key (if not using rotation) |
| `GOOGLE_API_KEY` | Fallback | Alternative Google API key env var |
| `GROQ_API_KEYS` | Optional | Comma-separated Groq API keys (fallback LLM) |
| `GROQ_API_KEY` | Fallback | Single Groq API key |

**Example `.env`:**
```env
GEMINI_API_KEYS=AIza..._key1,AIza..._key2,AIza..._key3
GROQ_API_KEYS=gsk_...your_key
```

> **Security:** Never commit `.env` to version control. It is listed in `.gitignore`.

---

## 8. Data Formats

### `data/chunks/parent_chunks.jsonl`

JSONL file, one JSON object per line:

```json
{
  "id": "3f2a4b1c-9d8e-4f7a-b2c1-e5f6a7b8c9d0",
  "doc_id": "aws-overview",
  "category": "service",
  "service_name": "Amazon S3",
  "text_content": "Amazon S3 is an object storage service...",
  "metadata": { "page": 41, "section": "Storage" }
}
```

### `data/chunks/child_chunks.jsonl`

```json
{
  "id": "a1b2c3d4-...",
  "parent_id": "3f2a4b1c-...",
  "doc_id": "aws-overview",
  "text_content": "Amazon S3 stores data as objects within buckets...",
  "metadata": { "chunk_index": 0 }
}
```

### `ground truth/evaluation_dataset.json`

```json
[
  {
    "question": "What is Amazon S3?",
    "ground_truth": "Amazon S3 is a scalable object storage service...",
    "expected_parent_ids": ["3f2a4b1c-...", "7d8e9f0a-..."]
  }
]
```

---

## 9. Error Handling & Known Issues

### pyarrow DLL Block (Windows Application Control)

**Symptom:**
```
ImportError: DLL load failed while importing _compute: 
An Application Control policy has blocked this file.
```

**Cause:** `sentence_transformers` → `datasets` → `pyarrow._compute.pyd` — blocked by enterprise Windows policy.

**Fix:** `datasets_shim.py` intercepts the import chain and stubs out `datasets` before `pyarrow` is loaded. Inference (embedding + reranking) is unaffected.

**Both `main.py` and `api.py` import `datasets_shim` at the top.**

---

### Gemini API Rate Limits (429)

**Symptom:** `Error calling model 'gemini-3.6-flash': RESOURCE_EXHAUSTED`

**Fix:** Configure multiple Gemini API keys in `.env` as a comma-separated list under `GEMINI_API_KEYS`. The key rotator cycles through them automatically, sleeping 1.5s between retries on `429` errors.

---

### Gemini API 403 (Permission Denied)

**Symptom:** `403 PERMISSION_DENIED` on specific keys.

**Cause:** The Gemini API key has been disabled or doesn't have access to the requested model.

**Fix:** The key rotator automatically skips 403 keys and moves to the next one. Remove disabled keys from `GEMINI_API_KEYS`.

---

### No Results from Retrieval

**Symptom:** LLM returns `"This isn't covered in the document."` for questions that should be answerable.

**Possible Causes:**
1. The database is empty — re-run indexing: `python -m src.indexing.build_vectorstore`
2. The query uses an unsupported language (only English is supported)
3. The topic is genuinely not covered in the AWS Overview whitepaper

---

## 10. Extending the System

### Adding a New AWS Service to the Knowledge Base

1. Update `data/raw/aws-overview.pdf` with the new version of the whitepaper.
2. Re-run the ingestion and indexing pipeline:
   ```powershell
   python -m src.ingest.parse_pdf
   python -m src.indexing.build_vectorstore
   ```

### Adding a New Evaluation Question

Edit `ground truth/evaluation_dataset.json` directly — add a new record:
```json
{
  "question": "Your new question",
  "ground_truth": "The expected reference answer",
  "expected_parent_ids": []
}
```

If you don't know the `expected_parent_ids`, leave the list empty. MRR and NDCG@5 will score `0.0` for that record, but Context Precision/Recall, Faithfulness, and Answer Correctness will still be evaluated.

### Adding a New LLM Provider

1. Add new API keys to `.env` (e.g., `ANTHROPIC_API_KEYS=...`)
2. Extend `APIKeyRotator.get_keys()` in `backend/utils/key_rotator.py` to handle the new provider string.
3. Update `rag_chain.py` to use the new LangChain LLM class.

### Changing the Embedding Model

1. Update `embedding_model` in `config.yaml`.
2. Re-run indexing to rebuild all embeddings:
   ```powershell
   python -m src.indexing.build_vectorstore
   ```
3. Update the pgvector column dimension if the new model has a different output size (default BGE-large: 1024 dims).

