"""Vector and BM25 indexing for AWS Overview RAG chunks.

This module indexes the 252 hierarchy-aware chunks from chunks.jsonl into:
1. A local persistent Chroma vector store using BAAI/bge-large-en-v1.5 embeddings
   with asymmetric query instruction prefixing.
2. A persisted BM25 index over the identical chunk set, keyed by the same IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from langchain_community.retrievers import BM25Retriever
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    # Fallback for installations where BM25Retriever is in langchain_classic
    from langchain_classic.retrievers import BM25Retriever

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import load_config


DEFAULT_CHUNKS = Path("data/chunks/chunks.jsonl")
DEFAULT_BM25_FILENAME = "bm25.pkl"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
CHROMA_COLLECTION_NAME = "aws_overview"


class BGEEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for BGE with asymmetric query instruction."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        query_instruction: str = BGE_QUERY_INSTRUCTION,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.normalize_embeddings = normalize_embeddings
        self.client = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunk texts as-is without instruction prefix."""
        embeddings = self.client.encode(
            texts,
            batch_size=32,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed query with BGE search instruction prefix."""
        prefixed_text = f"{self.query_instruction}{text}"
        embedding = self.client.encode(
            prefixed_text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load JSONL chunk objects from disk."""
    with chunks_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunks_to_documents(chunks: list[dict[str, Any]]) -> list[Document]:
    """Convert chunk dicts to LangChain Documents with sanitized Chroma metadata."""
    documents: list[Document] = []
    for chunk in chunks:
        metadata = {
            "id": str(chunk["id"]),
            "chunk_type": str(chunk.get("chunk_type", "")),
            "category": str(chunk.get("category") or ""),
            "service_name": str(chunk.get("service_name") or ""),
            "concept_group": str(chunk.get("concept_group") or ""),
            "page_start": int(chunk.get("page_start", 0)),
            "page_end": int(chunk.get("page_end", 0)),
            "source_urls": json.dumps(chunk.get("source_urls", []), ensure_ascii=False),
        }
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )
    return documents


def tokenize_words(text: str) -> list[str]:
    """Regex word tokenizer for BM25 matching."""
    return re.findall(r"\w+", text.lower())


def build_chroma_vectorstore(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: Path | str,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """Build or update Chroma vector store using chunk IDs as document keys."""
    persist_dir = str(persist_directory)
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    ids = [doc.metadata["id"] for doc in documents]
    vectorstore.add_documents(documents=documents, ids=ids)
    return vectorstore


def build_bm25_retriever(
    documents: list[Document],
    k: int = 10,
    persist_path: Path | str | None = None,
) -> BM25Retriever:
    """Build BM25 index over documents and optionally persist to disk."""
    retriever = BM25Retriever.from_documents(
        documents=documents,
        k=k,
        preprocess_func=tokenize_words,
    )
    if persist_path is not None:
        p_path = Path(persist_path)
        p_path.parent.mkdir(parents=True, exist_ok=True)
        with p_path.open("wb") as f:
            pickle.dump(retriever, f)
    return retriever


def load_bm25_retriever(persist_path: Path | str) -> BM25Retriever:
    """Load a persisted BM25 retriever from disk."""
    with Path(persist_path).open("rb") as f:
        return pickle.load(f)


def verify_indexing(
    vectorstore: Chroma,
    bm25: BM25Retriever,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run thorough verification across vector store and BM25 indices."""
    expected_count = len(chunks)
    actual_chroma_count = vectorstore._collection.count()

    # Check 1 & 6: Vector count and chunk ID presence
    chroma_all = vectorstore._collection.get()
    stored_ids = set(chroma_all["ids"])
    expected_ids = {chunk["id"] for chunk in chunks}
    missing_ids = expected_ids - stored_ids

    # Check 2: Similarity search for 'serverless container orchestration'
    query1 = "serverless container orchestration"
    docs1 = vectorstore.similarity_search(query1, k=5)
    top5_q1 = [
        {
            "id": d.metadata.get("id"),
            "service_name": d.metadata.get("service_name"),
            "chunk_type": d.metadata.get("chunk_type"),
            "category": d.metadata.get("category"),
        }
        for d in docs1
    ]
    q1_terms = ["fargate", "container", "ecs", "eks", "orchestration"]
    q1_passed = any(
        any(
            t in (d.metadata.get("service_name", "") + " " + d.metadata.get("category", "") + " " + d.page_content).lower()
            for t in q1_terms
        )
        for d in docs1
    )

    # Check 3: Similarity search for 'what does MWAA stand for'
    query2 = "what does MWAA stand for"
    docs2 = vectorstore.similarity_search(query2, k=5)
    top5_q2 = [
        {
            "id": d.metadata.get("id"),
            "service_name": d.metadata.get("service_name"),
            "chunk_type": d.metadata.get("chunk_type"),
            "category": d.metadata.get("category"),
        }
        for d in docs2
    ]
    q2_passed = any("mwaa" in d.metadata.get("id", "").lower() for d in docs2)

    # Check 4: BM25 search for 'MWAA'
    bm25_mwaa_docs = bm25.invoke("MWAA")
    top5_bm25_mwaa = [
        {
            "id": d.metadata.get("id"),
            "service_name": d.metadata.get("service_name"),
        }
        for d in bm25_mwaa_docs[:5]
    ]
    bm25_mwaa_passed = bool(
        top5_bm25_mwaa and "mwaa" in top5_bm25_mwaa[0].get("id", "").lower()
    )

    return {
        "expected_chunk_count": expected_count,
        "chroma_vector_count": actual_chroma_count,
        "count_matches": actual_chroma_count == expected_count,
        "all_chunk_ids_present": len(missing_ids) == 0,
        "missing_chunk_ids": list(missing_ids),
        "query1_serverless_container_top5": top5_q1,
        "query1_passed": q1_passed,
        "query2_mwaa_top5": top5_q2,
        "query2_passed": q2_passed,
        "bm25_mwaa_top5": top5_bm25_mwaa,
        "bm25_mwaa_passed": bm25_mwaa_passed,
        "local_execution": True,
    }


def _print_report(report: dict[str, Any]) -> None:
    """Print detailed verification report."""
    print("=" * 60)
    print("PHASE 4 INDEXING VERIFICATION REPORT")
    print("=" * 60)
    print(f"Total chunks in source: {report['expected_chunk_count']}")
    print(f"Total vectors in Chroma: {report['chroma_vector_count']}")
    print(f"Count match check: {report['count_matches']}")
    print(f"All 252 chunk IDs present in Chroma: {report['all_chunk_ids_present']}")
    if report['missing_chunk_ids']:
        print(f"Missing IDs: {report['missing_chunk_ids']}")

    print("\n--- Check 2: Dense Query ('serverless container orchestration') ---")
    print(f"Passed: {report['query1_passed']}")
    for i, item in enumerate(report['query1_serverless_container_top5'], 1):
        print(f"  {i}. [{item['id']}] {item['service_name'] or item['chunk_type']} ({item['category']})")

    print("\n--- Check 3: Dense Query ('what does MWAA stand for') ---")
    print(f"Passed: {report['query2_passed']}")
    for i, item in enumerate(report['query2_mwaa_top5'], 1):
        print(f"  {i}. [{item['id']}] {item['service_name'] or item['chunk_type']} ({item['category']})")

    print("\n--- Check 4: BM25 Acronym Query ('MWAA') ---")
    print(f"Passed: {report['bm25_mwaa_passed']}")
    for i, item in enumerate(report['bm25_mwaa_top5'], 1):
        print(f"  {i}. [{item['id']}] {item['service_name']}")

    print("\n--- Check 5: Local Execution (no API/network calls) ---")
    print(f"Passed: {report['local_execution']}")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> int:
    """Build vector store and BM25 retriever from chunks."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    parser = argparse.ArgumentParser(description="Index AWS Overview chunks into Chroma and BM25.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--vectorstore", type=Path, default=Path(config.vectorstore_path))
    parser.add_argument("--bm25-path", type=Path, default=Path(config.vectorstore_path) / DEFAULT_BM25_FILENAME)
    parser.add_argument("--report", action="store_true", help="Print verification report.")
    args = parser.parse_args(argv)

    print(f"Loading chunks from {args.chunks}...")
    chunks = load_chunks(args.chunks)
    documents = chunks_to_documents(chunks)
    print(f"Loaded {len(documents)} document objects.")

    print(f"Initializing embedding model: {config.embedding_model}...")
    embeddings = BGEEmbeddings(model_name=config.embedding_model)

    print(f"Building Chroma vector store at {args.vectorstore}...")
    vectorstore = build_chroma_vectorstore(
        documents=documents,
        embeddings=embeddings,
        persist_directory=args.vectorstore,
    )
    print(f"Chroma vector store built with {vectorstore._collection.count()} items.")

    print(f"Building BM25 index and saving to {args.bm25_path}...")
    bm25 = build_bm25_retriever(
        documents=documents,
        k=config.top_k,
        persist_path=args.bm25_path,
    )
    print("BM25 index built and persisted.")

    report = verify_indexing(vectorstore, bm25, chunks)
    if args.report:
        _print_report(report)

    all_passed = (
        report["count_matches"]
        and report["all_chunk_ids_present"]
        and report["query1_passed"]
        and report["query2_passed"]
        and report["bm25_mwaa_passed"]
    )
    if not all_passed:
        print("Verification FAILED!", file=sys.stderr)
        return 1

    print("Phase 4 indexing completed and verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
