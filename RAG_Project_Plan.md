# AWS Overview RAG — Project Plan & Codex Prompt Pack

Source document confirmed by direct inspection: `aws-overview.pdf`, 162 pages, AWS Whitepaper
"Overview of Amazon Web Services." Structure: Category → Service → description, ~15 real
multi-page comparison tables, ~10 large hero infographics per category, many decorative
72×72px icons (no info value), heavy hyperlinking on service names.

---

## 0. Design decisions and WHY (read this before running any prompt)

### Chunking strategy: **Hierarchy-aware structural chunking**, not fixed-size splitting

**What:** Parse the PDF into a structure tree (Category → Service → paragraphs) using heading
detection (font size / boldness, not just text). Each **service subsection becomes one chunk**
(most are 100–300 words — naturally under any reasonable token budget). Only if a section
exceeds ~800 tokens, recursively split it with `RecursiveCharacterTextSplitter` *inside* that
section boundary (never merge two services into one chunk, never split a service in half).

**Why not naive fixed-size (e.g. 500-char sliding window) chunking:** This document is a
glossary of ~160 independent entities. A fixed-size splitter, applied blindly, will frequently
straddle the boundary between two unrelated services (e.g. end of "Amazon MQ" + start of
"Blockchain" in one chunk), which:
- Pollutes the embedding for both services (semantic centroid shifts toward neither), hurting
  retrieval precision.
- Causes the LLM to answer a question about Service A using text that's actually about Service B.
- Produces citations that point to page X for content that's actually about a different service
  on the same page.

Since the natural unit of meaning here **is** the service description, aligning chunk boundaries
to it is strictly better than any generic splitter, and it's cheap to do because the PDF's
heading structure is clean and detectable.

**Every chunk gets rich metadata** (this matters more than the splitting itself):
`category`, `service_name`, `page_start`, `page_end`, `chunk_type` (`text` / `table` / `image_caption`),
`source_urls` (hyperlinks in that section).

### Tables: extracted and chunked separately, never flattened into prose

Tables are parsed with their row/column structure preserved (converted to Markdown table
syntax) and reassembled across page breaks (detecting repeated header rows like
`Category | AWS service`). Each table is its **own chunk** (or split by logical row-group if it's
very long), tagged `chunk_type=table`. This preserves the ability to answer comparison questions
("what's the difference between EC2 and Fargate for compute?") accurately, which would be
impossible if the table text got mixed into the surrounding paragraph flow or chopped mid-row.

### Images: hero infographics captioned, decorative icons discarded

- Filter by pixel size: anything under ~150px in both dimensions is a decorative badge → **drop it**.
- Anything larger (the category hero diagrams) → extract, send to a vision-capable LLM for a
  structured caption that captures the *relationships* shown (e.g. "pipeline order: ingestion →
  processing → analysis → warehousing"), and index that caption as a `chunk_type=image_caption`
  chunk tied to its category. This captures information that is genuinely only in the diagram
  (ordering/flow), which plain OCR or the surrounding bullet list doesn't fully convey.

### Retrieval strategy: **Hybrid (BM25 + dense) with Reciprocal Rank Fusion, then reranking**

Query patterns this document needs to support, and why single-method retrieval fails each one:

| Query type | Example | Why pure dense fails | Why pure BM25 fails |
|---|---|---|---|
| Direct entity lookup | "What is Amazon EMR?" | usually fine actually | usually fine actually |
| Acronym / exact name | "What does MWAA stand for?" | embeddings can conflate similar service names/acronyms | fine — this is where BM25 shines |
| Cross-service comparison | "Compare EC2 and Fargate" | may retrieve only one service's chunk if phrasing is asymmetric | may miss if exact terms don't co-occur |
| Broad category enumeration | "What analytics services does AWS offer?" | top-k similarity naturally favors a few dominant matches, not full-category recall | keyword match on "analytics" catches more of them |

**Chosen setup:**
1. **Hybrid retrieval**: LangChain `EnsembleRetriever` combining a dense vector retriever
   (Chroma) and `BM25Retriever`, merged via Reciprocal Rank Fusion. This covers both semantic
   paraphrase queries and exact-name/acronym queries.
2. **Metadata filtering / self-query** for category-enumeration queries: LangChain
   `SelfQueryRetriever` lets the LLM translate "what analytics services exist" into a structured
   filter (`category == "Analytics"`) and pull *all* matching chunks rather than relying on
   top-k similarity, which structurally under-recalls for "list everything in X" queries.
3. **Cross-encoder reranking** on the fused candidate set (e.g. a local `bge-reranker` or
   Cohere Rerank) before handing the final top-N to the LLM — cheap precision boost since
   first-stage retrieval is optimized for recall, not precision.
4. **Parent-document expansion**: retrieve at the fine-grained chunk level for precision, but
   pass the LLM the *full service section* (or full table) as context, not just the matched
   fragment — since these sections are already short, this costs little context budget and
   guarantees the LLM never sees a truncated answer.

### Embeddings & vector store
- Embedding model: pluggable via LangChain's model abstraction (default suggestion:
  `text-embedding-3-large` if using OpenAI, or `bge-large-en-v1.5` locally if you want a
  no-API-key option). Don't hardcode this — put it behind a config value.
- Vector store: **Chroma**, local/persistent, terminal-friendly, supports metadata filters
  natively (needed for the self-query retriever) — no hosted DB needed for this scope.

### LLM for generation
Left pluggable behind LangChain's `BaseChatModel` interface — pick whichever provider you have
an API key for (OpenAI, Anthropic, or local via Ollama). The prompt template forces the model to
(a) answer only from retrieved context, (b) cite `service_name (page X)` for every claim, and
(c) explicitly say "not found in document" rather than guessing, since this is a closed-corpus
factual QA task, not open-ended chat.

---

## 1. Repo layout (fix this before Phase 1 so every later prompt agrees)

```
aws-rag/
├── config.yaml                  # embedding model, chunk size, vector store path, top_k, etc.
├── data/
│   ├── raw/aws-overview.pdf
│   ├── processed/               # structured JSON after Phase 1-2
│   └── chunks/                  # final chunk objects after Phase 3
├── src/
│   ├── ingest/
│   │   ├── parse_pdf.py         # Phase 1
│   │   ├── extract_tables.py    # Phase 1
│   │   └── extract_images.py    # Phase 2
│   ├── chunking/
│   │   └── build_chunks.py      # Phase 3
│   ├── indexing/
│   │   └── build_vectorstore.py # Phase 4
│   ├── retrieval/
│   │   └── retriever.py         # Phase 5
│   ├── generation/
│   │   └── rag_chain.py         # Phase 6
│   └── cli.py                   # Phase 8
├── eval/
│   ├── test_questions.jsonl     # Phase 7
│   └── run_eval.py              # Phase 7
├── requirements.txt
└── README.md
```

---

## 2. Phases, in the order to run them with Codex

Run **one phase per Codex session**. Don't paste the whole plan at once — see the "Working
with Codex" section at the bottom for why. After each phase: run the verification command
yourself, look at real output, then move on.

### Phase 0 — Scaffolding
**Goal:** repo skeleton, venv, dependencies, config loader, empty stub files matching the layout above.
**Verify:** `python -m src.cli --help` runs without import errors.

### Phase 1 — PDF parsing into structured JSON (text + tables)
**Goal:** Walk the PDF, detect heading levels via font-size/boldness heuristics (or use
`unstructured.partition_pdf(strategy="hi_res")` which classifies Title/NarrativeText/Table
elements automatically — recommended, don't hand-roll heading detection if this library is
available), and emit one JSON record per Category and per Service, plus separate Table records
with rows preserved and multi-page tables merged.
**Verify:** spot-check 5 known services (e.g. Amazon EMR, AWS Lambda, Amazon Aurora) and 2 known
tables (compute, database comparisons) appear correctly and completely in the output JSON.

### Phase 2 — Image extraction and captioning
**Goal:** extract embedded images, drop anything under 150×150px, send the rest to a
vision-capable LLM with a prompt asking specifically for the *relationships/flow* shown, attach
resulting captions to their category record.
**Verify:** print the caption generated for the Analytics hero image and confirm it mentions the
pipeline ordering, not just a generic description.

### Phase 3 — Chunk construction
**Goal:** build final chunk objects per the hierarchy rules in section 0 above (one chunk per
service, tables as their own chunks, image captions as their own chunks), all metadata attached.
**Verify:** print total chunk count, distribution by `chunk_type`, and confirm no chunk mixes two
`service_name` values.

### Phase 4 — Embedding + vector store indexing
**Goal:** embed all chunks, persist to Chroma with metadata, build the BM25 index over the same
chunk set (kept in sync — same chunk IDs).
**Verify:** query the vector store directly for "serverless container orchestration" and confirm
Fargate/ECS-related chunks come back.

### Phase 5 — Hybrid retrieval + self-query + reranking
**Goal:** `EnsembleRetriever` (dense + BM25, RRF-merged), `SelfQueryRetriever` for
category-filtered queries, reranker as a final pass, parent-section expansion before returning.
**Verify:** run the 4 example query types from the table in section 0 and manually confirm the
retrieved chunks are correct BEFORE moving to generation — this is the step where silent bugs
hide, since a fluent wrong answer in Phase 6 will look fine unless you already checked retrieval.

### Phase 6 — Generation chain
**Goal:** LCEL chain: retriever → prompt (grounded-answer + citation instructions + "say not
found if absent") → LLM → parse citations out for display.
**Verify:** ask 3 questions with known answers and 1 question about something NOT in the document
(e.g. "What is Google BigQuery?") and confirm it correctly refuses instead of hallucinating.

### Phase 7 — Evaluation set
**Goal:** ~30 hand-written questions covering all 4 query types + table questions + image/diagram
questions, with expected key facts (not full expected strings). Score faithfulness/relevance,
ideally with RAGAS if available, otherwise a simple LLM-graded rubric.
**Verify:** run once, read every failure by hand, don't just trust an aggregate score.

### Phase 8 — Terminal CLI
**Goal:** simple REPL: `> ask your question`, prints answer + citations. No frontend needed.
**Verify:** interactive session, ask 5 questions live.

### Phase 9 (optional/stretch)
Conversational follow-up handling (query rewriting using chat history), response caching,
adding query rewriting/HyDE for especially short user queries.

---

## 3. Ready-to-paste Codex prompts, one per phase

Paste `PROJECT_CONTEXT` once at the start of a new Codex session, then the phase prompt.

### PROJECT_CONTEXT (paste first, every session)
```
We're building a RAG system over a single AWS whitepaper PDF (162 pages, "Overview of Amazon
Web Services") using LangChain and Python. Full design spec and repo layout is in
RAG_Project_Plan.md at the repo root — read it before writing any code, and follow its chunking
and retrieval design decisions exactly; don't substitute your own default RAG pattern (e.g. don't
default to naive fixed-size chunking — the plan explains why that's wrong for this doc).
We build this in small phases, one per session. Only implement the phase I specify below. Don't
implement later phases even partially. Stub any interface you need from a later phase with a
clearly marked TODO and a docstring describing the expected contract.
Before writing code, output a short plan (files you'll touch/create, function signatures,
sample input/output) and wait for me to confirm before proceeding.
When done, tell me exactly which command to run to verify your work, and what output I should
expect to see.
```

### Phase 0 prompt
```
Phase 0: Repo scaffolding.
Create the exact folder/file layout from RAG_Project_Plan.md section 1. requirements.txt should
include langchain, langchain-community, chromadb, rank-bm25, pypdf or unstructured (your call,
justify briefly), and a config loader (config.yaml -> pydantic or simple dataclass) exposing:
embedding_model, chunk_max_tokens, vectorstore_path, top_k, rerank_top_n. src/cli.py should be a
stub that argparses --help successfully. Give me the verification command.
```

### Phase 1 prompt
```
Phase 1: PDF parsing into structured JSON.
Input: data/raw/aws-overview.pdf. Implement src/ingest/parse_pdf.py to walk the document and
emit data/processed/structure.json containing a list of Category objects, each with a name,
page range, and a list of Service objects (name, description text, page range, source_urls
found in that section). Implement src/ingest/extract_tables.py to detect the ~15 known
comparison tables (2-column: Category | AWS service, with bullet-list cell content), preserve
row structure as markdown tables, and correctly merge tables that continue across a page break
(detect by repeated header row). Emit these as a separate list of Table objects in the same
structure.json (or a sibling tables.json — your call, tell me which and why).
Before coding: tell me which parsing library you'll use (unstructured hi-res partitioning is
preferred if available; otherwise pdfplumber/PyMuPDF with font-size heuristics — justify
whichever you pick) and show me your planned JSON schema.
Verification: I will manually check that Amazon EMR, AWS Lambda, Amazon Aurora, the "Compare AWS
compute services" table, and the "Compare AWS database services" table are all present and
complete (not truncated at a page boundary) in the output. Tell me the exact command to
regenerate and inspect the output.
```

### Phase 2 prompt
```
Phase 2: Image extraction and captioning.
Implement src/ingest/extract_images.py: extract embedded images per page from the PDF (pymupdf
or pdfimages via subprocess, your call), discard any image under 150x150px (decorative icons),
and for the remainder, call a vision-capable LLM (make the provider pluggable/configurable —
don't hardcode one) with a prompt that specifically asks it to describe any process flow,
ordering, or grouping relationships shown in the diagram, not just a generic visual description.
Attach the resulting caption to the relevant Category record in structure.json as an
image_caption field, tagged with its page number.
Verification: print the caption generated for the image on the Analytics category page and
confirm it mentions the pipeline order (ingestion -> processing -> analytics -> warehousing),
not just "a diagram of AWS analytics services."
```

### Phase 3 prompt
```
Phase 3: Chunk construction.
Implement src/chunking/build_chunks.py, reading structure.json (+ tables.json), producing
data/chunks/chunks.jsonl where each line is one chunk: {id, text, chunk_type
(text/table/image_caption), category, service_name (nullable for table/image chunks), page_start,
page_end, source_urls}.
Rule: one chunk per Service by default. Only if a service's description exceeds
config.chunk_max_tokens, split it further using RecursiveCharacterTextSplitter *within* that
service's text only — never merge across service boundaries. Tables and image captions are
always their own chunk(s) regardless of size (split a very long table by logical row-group if
needed, keeping the header row repeated in each split for standalone readability).
Verification: print total chunk count, a breakdown by chunk_type, and assert (print a check) that
no single chunk's text contains two different service names from structure.json's service name
list.
```

### Phase 4 prompt
```
Phase 4: Embedding + vector store indexing.
Implement src/indexing/build_vectorstore.py: embed every chunk from chunks.jsonl using the
configured embedding model, persist to a local Chroma collection at config.vectorstore_path with
full metadata attached (category, service_name, chunk_type, page_start, page_end). Separately
build and persist a BM25Retriever over the same chunk set, keyed by the same chunk ids, so the
two retrievers stay in sync if chunks.jsonl changes.
Verification: run a direct similarity query for "serverless container orchestration" against the
Chroma collection and print the top 5 results with their service_name metadata — I expect
Fargate/ECS/EKS-related chunks near the top.
```

### Phase 5 prompt
```
Phase 5: Hybrid retrieval, self-query filtering, reranking, parent-section expansion.
Implement src/retrieval/retriever.py exposing a single retrieve(query: str) -> List[Document]
function that:
1. Runs an EnsembleRetriever combining the Chroma dense retriever and the BM25Retriever, merged
   via reciprocal rank fusion.
2. Also supports a SelfQueryRetriever path for category-enumeration style queries (LLM
   translates the query into a metadata filter over `category`), and decides which path to use
   (or combine both) — tell me your decision logic before implementing.
3. Applies a cross-encoder reranker (pick one that doesn't require an extra paid API if possible,
   e.g. a local bge-reranker via sentence-transformers/FlagEmbedding; otherwise Cohere Rerank
   behind a config flag) to the fused candidates, keeping config.rerank_top_n.
4. Expands each surviving chunk to its full parent section (full service description, or full
   table) before returning, using structure.json/tables.json as the source of truth for parent
   content.
Verification: run these four queries and print the retrieved chunks' service_name/category for
each, so I can manually confirm correctness before we build generation on top:
  - "What is Amazon EMR?"
  - "What does MWAA stand for?"
  - "Compare EC2 and Fargate for running containers"
  - "What analytics services does AWS offer?"
```

### Phase 6 prompt
```
Phase 6: Generation chain.
Implement src/generation/rag_chain.py as a LangChain LCEL chain: retriever (from Phase 5) ->
prompt -> chat model (pluggable via config, don't hardcode a provider) -> output parser.
Prompt must instruct the model to: answer only using the provided context, cite each claim as
"(Service Name, page N)", and explicitly respond "This isn't covered in the document" if the
retrieved context doesn't answer the question rather than using general knowledge.
Verification: ask 3 questions with known answers from the document (I'll supply them) and 1
question about something NOT in the document (e.g. "What is Google BigQuery?") — I expect the
first 3 to be correctly answered with citations, and the 4th to be correctly refused rather than
hallucinated.
```

### Phase 7 prompt
```
Phase 7: Evaluation harness.
Create eval/test_questions.jsonl with ~30 questions I'll help curate, covering: direct lookup,
acronym-based, comparison, category-enumeration, table-based, and image/diagram-based questions,
each with a list of expected key facts (not exact strings). Implement eval/run_eval.py to run
each question through the Phase 6 chain and score faithfulness (is every claim supported by
retrieved context) and relevance (were the expected key facts present). Use RAGAS if it's already
a reasonable fit, otherwise implement a simple LLM-graded rubric — tell me which and why.
Verification: run it and show me the full per-question breakdown, not just an aggregate score —
I want to read the failures myself.
```

### Phase 8 prompt
```
Phase 8: Terminal CLI.
Implement src/cli.py as a simple REPL: prompt "> ", read a question, run it through the Phase 6
chain, print the answer with citations, loop until the user types "exit". No web/frontend
component needed.
Verification: I'll run it interactively and ask a few questions myself.
```

---

## 4. Working with Codex (or any coding agent) so it actually does what you want

This is the part that was probably going wrong before, based on what you described:

1. **One phase per session, not the whole project in one prompt.** A single giant prompt
   ("build me a full RAG app") gives the agent too much freedom to invent its own architecture,
   skip steps silently, or produce glue code that looks complete but wasn't verified anywhere.
   Small, scoped phases with an explicit "don't implement later phases" instruction keep it
   honest about what's actually done.

2. **Freeze decisions in a written spec the agent must read, not just describe in chat.**
   That's what `RAG_Project_Plan.md` is for. Chat context resets or gets summarized between
   sessions; a file on disk doesn't. Every phase prompt above tells Codex to read it first.

3. **Ask for a plan before code, every time**, and actually read it before saying "go." This
   catches misunderstandings (wrong library, wrong schema, wrong chunk boundary) while they're
   one sentence to fix, instead of after 200 lines of code are already built on the wrong
   assumption.

4. **Always demand a concrete verification command + expected output**, and actually run it
   yourself. RAG failures are silent — a wrong-context answer still reads fluently. "It works" from
   the agent is not evidence; a specific retrieved chunk or printed diff is.

5. **Give errors, not vibes, when something's wrong.** "The retrieval isn't good" produces vague
   fixes. Pasting the actual query, the actual retrieved chunks, and what you expected instead
   produces a targeted fix.

6. **Commit after each verified phase.** If phase 5 breaks something in phase 3's output, you
   want a clean rollback point, not a half-mixed state.

7. **Don't let the agent silently change earlier decisions.** If Codex decides mid-phase-5 that
   it wants to redo chunking, that's a sign the phase prompt wasn't scoped tightly enough — stop
   it, and go fix the actual root cause instead of letting scope creep backward.
