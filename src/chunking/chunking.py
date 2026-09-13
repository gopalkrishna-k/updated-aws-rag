"""chunking.py — Refactored Hierarchical Two-Tier Chunking.

Transforms Markdown text into a two-level hierarchy of Parent Chunks (Context Payload)
and Child Chunks (Search Payload) using MarkdownHeaderTextSplitter (H1-H5) and RecursiveCharacterTextSplitter.
Includes atomic Markdown table protection, pre-processing cleanup, and metadata consolidation.
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


def clean_markdown_text(raw_text: str) -> str:
    """Pre-process and clean raw Markdown text by removing page markers, stray headers/footers, and <u> tags."""
    # 1. Strip HTML tags like <u> and </u> from text
    text = re.sub(r"<\/?u>", "", raw_text)

    # 2. Strip HTML page comment markers (e.g. <!-- PAGE_11 -->)
    text = re.sub(r"<!--\s*PAGE_\d+\s*-->", "", text)

    # 3. Filter out stray running headers/footers line-by-line
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip recurring header/footer titles
        if stripped in (
            "Overview of Amazon Web Services",
            "AWS Whitepaper",
            "Overview of Amazon Web Services AWS Whitepaper",
            "Overview of Amazon Web Services  AWS Whitepaper",
        ):
            continue
        # Skip copyright disclaimers
        if re.search(r"Copyright\s+©\s+\d{4}.*Amazon Web Services", stripped, re.IGNORECASE):
            continue
        if stripped.startswith("Copyright ©") or stripped.startswith("©"):
            continue
        # Skip standalone page numbers (e.g. "14")
        if re.fullmatch(r"\d+", stripped):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_header_str(val: str | None) -> str:
    """Sanitize header text by stripping Markdown bold (**), italics (* or _), and HTML tags (<u>)."""
    if not val:
        return ""
    cleaned = re.sub(r"<\/?u>", "", val)
    cleaned = re.sub(r"[\*_]{1,3}", "", cleaned)
    return cleaned.strip()


def extract_header_metadata(metadata: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract clean category (H3/H2), service_name (H4/H5 or fallback to H3/H2), and header_path."""
    h1 = clean_header_str(metadata.get("title") or metadata.get("Header 1"))
    h2 = clean_header_str(metadata.get("section") or metadata.get("Header 2"))
    h3 = clean_header_str(metadata.get("category") or metadata.get("Category") or metadata.get("Header 3"))
    h4 = clean_header_str(metadata.get("service_name") or metadata.get("Service") or metadata.get("Header 4"))
    h5 = clean_header_str(metadata.get("subservice_name") or metadata.get("Header 5"))

    # Category: H3 (or H2 if under section without H3)
    category = h3 if h3 else (h2 if h2 else "General")

    # Service name: H4 (or H5) if present; fallback to active category/section if no H4/H5 exists
    if h4:
        service_name = f"{h4} > {h5}" if h5 else h4
    elif h5:
        service_name = h5
    else:
        service_name = category

    # Construct clean breadcrumb header_path (e.g. "Analytics > Amazon Athena")
    path_parts = []
    if category and category != "General":
        path_parts.append(category)
    if service_name and service_name != category:
        path_parts.append(service_name)

    if not path_parts:
        if h2:
            path_parts.append(h2)
        elif h1:
            path_parts.append(h1)

    header_path = " > ".join(path_parts) if path_parts else "General"
    return category, service_name, header_path


def preserve_markdown_tables(parent_text: str, child_texts: List[str]) -> List[str]:
    """Ensure child chunks containing Markdown table rows retain their table header rows atomically."""
    table_header_match = re.search(r"(\|[^\n]+\|\n\|\s*[-:]+[-|\s:]*\|)", parent_text)
    if not table_header_match:
        return child_texts

    table_header = table_header_match.group(1)
    refined_children: List[str] = []

    for child in child_texts:
        lines = child.strip().splitlines()
        table_rows = [l for l in lines if l.strip().startswith("|") and l.strip().endswith("|")]
        if table_rows:
            has_header = any(re.match(r"^\|\s*[-:]+[-|\s:]*\|$", l.strip()) for l in lines)
            if not has_header:
                child = f"{table_header}\n{child}"
        refined_children.append(child)

    return refined_children


def generate_hierarchical_chunks(
    markdown_content: str,
    source_path: str = "aws-overview.md",
    parent_max_threshold: int = PARENT_MAX_THRESHOLD,
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate Parent Chunks and Child Chunks with H1-H5 heading support and atomic table protection."""
    logger.info("Pre-processing Markdown text (cleaning page markers, noise & HTML tags)...")
    cleaned_markdown = clean_markdown_text(markdown_content)

    logger.info("Parsing Markdown text with MarkdownHeaderTextSplitter (H1-H5)...")
    headers_to_split_on = [
        ("#", "title"),
        ("##", "section"),
        ("###", "category"),
        ("####", "service_name"),
        ("#####", "subservice_name"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_docs = markdown_splitter.split_text(cleaned_markdown)

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

    for doc in raw_parent_docs:
        clean_text = doc.page_content.strip()
        if not clean_text:
            continue

        parent_doc_id = str(uuid.uuid4())
        category, service_name, header_path = extract_header_metadata(doc.metadata)

        parent_obj = {
            "doc_id": parent_doc_id,
            "id": parent_doc_id,  # backward compatibility
            "source": Path(source_path).name,
            "category": category,
            "service_name": service_name,
            "service": service_name,  # consolidated mirror
            "header_path": header_path,
            "chunk_type": "parent",
            "text_content": clean_text,
        }
        parent_chunks.append(parent_obj)

        raw_sub_child_texts = child_text_splitter.split_text(clean_text)
        sub_child_texts = preserve_markdown_tables(clean_text, raw_sub_child_texts)

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
                "source": Path(source_path).name,
                "category": category,
                "service_name": service_name,
                "service": service_name,  # consolidated mirror
                "header_path": header_path,
                "chunk_type": "child",
                "text_content": prepended_text,
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
        print("\n--- SAMPLE PARENT CHUNKS ---")
        for sample_p in parents[:3]:
            print(f"  doc_id:       {sample_p['doc_id']}")
            print(f"  header_path:  {sample_p['header_path']}")
            print(f"  category:     {sample_p['category']}")
            print(f"  service_name: {sample_p['service_name']}")
            print(f"  service:      {sample_p['service']}")
            print(f"  chunk_type:   {sample_p['chunk_type']}")
            print(f"  text preview: {sample_p['text_content'][:150]}...")
            print()

    if children:
        print("--- SAMPLE CHILD CHUNKS (WITH CONTEXT PREPENDED) ---")
        for sample_c in children[:3]:
            print(f"  doc_id:       {sample_c['doc_id']}")
            print(f"  parent_id:    {sample_c['parent_id']}")
            print(f"  header_path:  {sample_c['header_path']}")
            print(f"  category:     {sample_c['category']}")
            print(f"  service_name: {sample_c['service_name']}")
            print(f"  service:      {sample_c['service']}")
            print(f"  chunk_type:   {sample_c['chunk_type']}")
            print(f"  text preview: {sample_c['text_content'][:180]}...")
            print()
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




