"""chunking.py — Refactored Hierarchical Two-Tier Chunking.

Transforms Markdown text into a two-level hierarchy of Parent Chunks (Context Payload)
and Child Chunks (Search Payload) using MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter.
NO vector embeddings or database storage are performed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_MD = Path("data/processed/aws-overview.md")
DEFAULT_PARENTS_OUTPUT = Path("data/chunks/parent_chunks.jsonl")
DEFAULT_CHILDREN_OUTPUT = Path("data/chunks/child_chunks.jsonl")

# Configuration constants as per specifications
PARENT_MAX_THRESHOLD = 2500
PARENT_SIZE = 2000
PARENT_OVERLAP = 200

CHILD_SIZE = 900
CHILD_OVERLAP = 120
CHILD_SEPARATORS = ["\n\n", "\n|", "\n- ", "\n", ". ", " "]


def extract_header_metadata(metadata: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract category (Header 2), service_name (Header 3), and header_path."""
    category = metadata.get("Header 2") or metadata.get("Category") or ""
    service_name = metadata.get("Header 3") or metadata.get("Service") or metadata.get("Header 4") or ""
    h1 = metadata.get("Header 1") or ""

    category = category.strip()
    service_name = service_name.strip()

    path_parts = []
    if category:
        path_parts.append(category)
    if service_name:
        path_parts.append(service_name)
    if not path_parts and h1.strip():
        path_parts.append(h1.strip())

    header_path = " > ".join(path_parts) if path_parts else "General"
    return category, service_name, header_path


def generate_hierarchical_chunks(
    markdown_content: str,
    source_path: str = "data/processed/aws-overview.md",
    parent_max_threshold: int = PARENT_MAX_THRESHOLD,
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate refined Parent Chunks (Context Payload) and Child Chunks (Search Payload) with lineage linking."""
    logger.info("Parsing Markdown text with MarkdownHeaderTextSplitter (#, ##, ###)...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_docs = markdown_splitter.split_text(markdown_content)

    # Fallback splitter for parent header blocks exceeding 2,500 characters
    parent_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Child chunk splitter preserving tables and lists
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=CHILD_SEPARATORS,
    )

    raw_parent_docs: List[Document] = []
    for doc in header_docs:
        if len(doc.page_content) > parent_max_threshold:
            sub_docs = parent_text_splitter.split_documents([doc])
            raw_parent_docs.extend(sub_docs)
        else:
            raw_parent_docs.append(doc)

    parent_chunks: List[Dict[str, Any]] = []
    child_chunks: List[Dict[str, Any]] = []

    current_page = 1  # Track page state across chunks

    for doc in raw_parent_docs:
        text = doc.page_content

        page_markers = re.findall(r"<!--\s*PAGE_(\d+)\s*-->", text)
        if page_markers:
            current_page = int(page_markers[0])
        chunk_page = current_page

        clean_text = re.sub(r"<!--\s*PAGE_\d+\s*-->", "", text).strip()
        if not clean_text:
            continue

        parent_doc_id = str(uuid.uuid4())
        category, service_name, header_path = extract_header_metadata(doc.metadata)

        parent_obj = {
            "doc_id": parent_doc_id,
            "id": parent_doc_id,  # backward compatibility
            "source": source_path,
            "category": category,
            "service_name": service_name,
            "service": service_name,  # backward compatibility
            "header_path": header_path,
            "chunk_type": "parent",
            "text_content": clean_text,
            "page": chunk_page,
        }
        parent_chunks.append(parent_obj)

        sub_child_texts = child_text_splitter.split_text(clean_text)

        for child_text in sub_child_texts:
            child_text_clean = child_text.strip()
            if not child_text_clean:
                continue

            child_doc_id = str(uuid.uuid4())
            context_prefix = f"[Context: {header_path}]\n\n" if header_path else ""
            prepended_text = f"{context_prefix}{child_text_clean}"

            child_obj = {
                "doc_id": child_doc_id,
                "id": child_doc_id,  # backward compatibility
                "parent_id": parent_doc_id,
                "source": source_path,
                "category": category,
                "service_name": service_name,
                "service": service_name,  # backward compatibility
                "header_path": header_path,
                "chunk_type": "child",
                "text_content": prepended_text,
                "page": chunk_page,
            }
            child_chunks.append(child_obj)

    logger.info("Successfully generated %d Parent Chunks and %d Child Chunks.", len(parent_chunks), len(child_chunks))
    return parent_chunks, child_chunks


def print_chunk_summary(parents: List[Dict[str, Any]], children: List[Dict[str, Any]]) -> None:
    """Print a structured summary and sample preview of Parent and Child Chunks."""
    print("\n" + "=" * 70)
    print("REFINED HIERARCHICAL CHUNKING SUMMARY & INSPECTION")
    print("=" * 70)
    print(f"Total Parent Chunks (Context Payload): {len(parents)}")
    print(f"Total Child Chunks  (Search Payload):  {len(children)}")

    if parents:
        print("\n--- SAMPLE PARENT CHUNK ---")
        sample_p = parents[0]
        print(f"doc_id:      {sample_p['doc_id']}")
        print(f"header_path: {sample_p['header_path']}")
        print(f"category:    {sample_p['category']}")
        print(f"service:     {sample_p['service_name']}")
        print(f"chunk_type:  {sample_p['chunk_type']}")
        print(f"text_content preview:\n{sample_p['text_content'][:200]}...")

    if children:
        print("\n--- SAMPLE CHILD CHUNK (WITH CONTEXT PREPENDED) ---")
        sample_c = children[0]
        print(f"doc_id:      {sample_c['doc_id']}")
        print(f"parent_id:   {sample_c['parent_id']}")
        print(f"header_path: {sample_c['header_path']}")
        print(f"category:    {sample_c['category']}")
        print(f"service:     {sample_c['service_name']}")
        print(f"chunk_type:  {sample_c['chunk_type']}")
        print(f"text_content preview:\n{sample_c['text_content'][:250]}...")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate hierarchical parent-child chunks from Markdown.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_MD)
    parser.add_argument("--parents-output", type=Path, default=DEFAULT_PARENTS_OUTPUT)
    parser.add_argument("--children-output", type=Path, default=DEFAULT_CHILDREN_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input markdown file not found: %s", args.input)
        raise FileNotFoundError(f"Input markdown file not found: {args.input}")

    markdown_text = args.input.read_text(encoding="utf-8")
    parents, children = generate_hierarchical_chunks(markdown_text, source_path=str(args.input))

    args.parents_output.parent.mkdir(parents=True, exist_ok=True)
    with args.parents_output.open("w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with args.children_output.open("w", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("Wrote parent chunks to %s", args.parents_output)
    logger.info("Wrote child chunks to %s", args.children_output)

    print_chunk_summary(parents, children)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


