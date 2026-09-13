"""
Command-line entry point for the AWS Overview RAG system.

Phase 8 implements an interactive REPL that:
- Welcomes the user and reminds them this is a closed-corpus system.
- Reads questions from stdin.
- Calls src.generation.rag_chain.generate_answer for non-empty queries.
- Prints the answer and a nicely formatted list of citations.
- Handles exit/quit commands and transient rate-limit errors.
- Tracks how many queries have been asked in the current session.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from typing import Any

# Suppress specific known-harmless library warnings before the heavy imports
# (langchain_google_genai, langchain_community, sentence_transformers) are
# loaded, so they never appear in the REPL's console output.
#
# 1. langchain_google_genai: "Model '...' uses fixed sampling defaults;
#    the sampling parameter(s) temperature will be ignored."
warnings.filterwarnings(
    "ignore",
    message=r".*fixed sampling defaults.*",
    category=UserWarning,
    module=r"langchain_google_genai",
)
# 2. LangChain community: deprecation warning about the `Chroma` class being
#    moved to `langchain-chroma`.
warnings.filterwarnings(
    "ignore",
    message=r".*Chroma.*deprecated.*",
    category=DeprecationWarning,
)
# Also catch it if emitted as LangChainDeprecationWarning (a UserWarning subclass).
warnings.filterwarnings(
    "ignore",
    message=r".*Chroma.*deprecated.*",
    category=UserWarning,
)
# 3. Hugging Face Hub: "You are sending unauthenticated requests to the HF Hub."
warnings.filterwarnings(
    "ignore",
    message=r".*unauthenticated requests.*HF Hub.*",
    category=UserWarning,
)

from src.generation.rag_chain import generate_answer


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=(
            "AWS Overview RAG CLI & REPL. "
            "Supports PDF ingestion, hierarchical chunking, PostgreSQL embedding, and QA."
        ),
    )
    parser.add_argument("--ingest", action="store_true", help="Extract raw PDF into clean Markdown (data/processed/aws-overview.md).")
    parser.add_argument("--chunk", action="store_true", help="Generate hierarchical parent and child chunks from Markdown.")
    parser.add_argument("--embed-store", action="store_true", help="Embed child chunks and persist parent/child chunks into PostgreSQL.")
    return parser


def _format_citations(citations: list[dict[str, Any]]) -> str:
    """Format structured citation dicts as a readable string.

    Each dict has a ``citation_label`` key such as ``"Amazon EMR, page 22"``
    or ``"Table: Compare AWS compute services, pages 42-44"``.  Labels are
    joined with ``'; '`` and prefixed with ``'Sources: '``.  Returns an empty
    string if the list is empty.
    """
    if not citations:
        return ""
    labels: list[str] = []
    for c in citations:
        label = c.get("citation_label", "").strip()
        if label:
            labels.append(label)
    if not labels:
        return ""
    return "Sources: " + "; ".join(labels)


def _run_repl() -> None:
    """Run the interactive question-answering REPL."""
    print(
        "\nWelcome to the AWS Overview RAG REPL!\n"
        "This is a closed-corpus system — answers are sourced exclusively from\n"
        "the 'Overview of Amazon Web Services' whitepaper.\n"
        "Type a question and press Enter.  Type 'exit' or 'quit' to stop.\n"
    )

    query_counter = 0

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue  # empty input — reprompt without a Gemini call

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Non-empty query — invoke the generation chain.
        try:
            result = generate_answer(user_input)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "resource_exhausted" in msg.lower() or "rate limit" in msg.lower() or "429" in msg:
                print(
                    "Rate limit hit — try again in a moment or check your Gemini quota."
                )
            else:
                print(f"An error occurred: {exc}")
            continue

        answer: str = result.get("answer", "")
        citations: list[dict[str, Any]] = result.get("citations", [])
        retrieved_chunks: list[dict[str, Any]] = result.get("retrieved_chunks", [])
        timings: dict[str, float] | None = result.get("timings")

        print("\n" + "=" * 70)
        print("ANSWER:\n")
        print(answer)

        citation_str = _format_citations(citations)
        if citation_str:
            print(f"\n{citation_str}")

        if retrieved_chunks:
            print("\n" + "-" * 70)
            print(f"RETRIEVED PARENT CHUNKS SENT TO LLM ({len(retrieved_chunks)} parent chunks):\n")
            for i, chunk in enumerate(retrieved_chunks, start=1):
                cid = chunk.get("id", "")
                ctype = chunk.get("chunk_type", "parent")
                cat = chunk.get("category", "")
                sname = chunk.get("service_name", "")
                p_start = chunk.get("page_start", 0)
                p_end = chunk.get("page_end", 0)
                p_str = f"page {p_start}" if (p_start and p_start == p_end) else (f"pages {p_start}-{p_end}" if p_start else "")

                header_parts = [f"Parent Chunk {i}: [{cid}]", f"Type: {ctype}"]
                if sname:
                    header_parts.append(f"Service: {sname}")
                if cat:
                    header_parts.append(f"Category: {cat}")
                if p_str:
                    header_parts.append(p_str)

                print(f"[{' | '.join(header_parts)}]")
                print(chunk.get("text", "").strip())
                print()

        query_counter += 1
        print("-" * 70)
        suffix = ""
        if timings and "total_s" in timings:
            suffix = f"  [Total Duration: {timings['total_s']:.1f}s]"
        print(f"[Query {query_counter} this session]{suffix}")
        print("=" * 70 + "\n")





def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    # Use the factory to automatically select the PostgreSQL‑backed retriever when configured
    from .retrieval import make_retriever
    retriever = make_retriever()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.ingest:
        from src.ingest.ingestion import process_pdf_to_markdown
        process_pdf_to_markdown()
        return 0

    if args.chunk:
        from src.chunking.chunking import DEFAULT_INPUT_MD, generate_hierarchical_chunks
        import json
        markdown_text = DEFAULT_INPUT_MD.read_text(encoding="utf-8")
        parents, children = generate_hierarchical_chunks(markdown_text)
        print(f"Generated {len(parents)} Parent Chunks and {len(children)} Child Chunks.")
        return 0

    if args.embed_store:
        from src.indexing.embed_store import DEFAULT_CHILDREN_FILE, DEFAULT_PARENTS_FILE, embed_and_store, load_jsonl
        parents = load_jsonl(DEFAULT_PARENTS_FILE)
        children = load_jsonl(DEFAULT_CHILDREN_FILE)
        embed_and_store(parents, children)
        return 0

    from src.config import load_config
    cfg = load_config()
    print(f"Retrieval top_k = {cfg.top_k} per technique (interleaved deduplication)")

    _run_repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

