# src/chunking/parent_child.py
"""Parent‑child chunking implementation for AWS‑RAG.

This module provides a **parent‑child** chunking strategy where larger
*parent* chunks (service, concept, table, notice, image‑caption) are first
created and then split into smaller *child* chunks that are indexed for
retrieval. Child chunks carry a ``parent_id`` field that references the
corresponding parent chunk, enabling the retrieval pipeline to fetch the
relevant parent context after a child matches.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_config

# ---------------------------------------------------------------------------
# Utility helpers (same as original implementation)
# ---------------------------------------------------------------------------
def _slug(value: str) -> str:
    """Return a deterministic identifier component."""
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold())
    return value.strip("-")


def _normalise_name(value: str) -> str:
    """Normalize PDF ligatures before comparing category labels."""
    return unicodedata.normalize("NFKC", value)


def _token_count(text: str) -> int:
    """Count whitespace‑delimited tokens (used for chunk size limits)."""
    return len(re.findall(r"\\S+", text))


def _service_text(record: dict[str, Any]) -> str:
    """Render a service or concept record with its name and description."""
    return f"{record['name']}\n\n{record['description'].strip()}".strip()


def split_service_text(text: str, chunk_max_tokens: int) -> List[str]:
    """Split an oversized service/concept text without breaking the record.

    If the text fits within ``chunk_max_tokens`` it is returned unchanged.
    Otherwise a :class:`RecursiveCharacterTextSplitter` is used.
    """
    if _token_count(text) <= chunk_max_tokens:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_max_tokens,
        chunk_overlap=0,
        length_function=_token_count,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _chunk(
    *,
    chunk_id: str,
    text: str,
    chunk_type: str,
    category: str | None,
    service_name: str | None,
    concept_group: str | None,
    page_start: int,
    page_end: int,
    source_urls: List[str],
    parent_id: str | None = None,
) -> dict[str, Any]:
    """Create a JSON‑serialisable chunk dictionary.

    ``parent_id`` is ``None`` for top‑level parent chunks and set to the
    containing parent’s ``id`` for child chunks.
    """
    return {
        "id": chunk_id,
        "text": text,
        "chunk_type": chunk_type,
        "category": category,
        "service_name": service_name,
        "concept_group": concept_group,
        "page_start": page_start,
        "page_end": page_end,
        "source_urls": source_urls,
        "parent_id": parent_id,
    }

# ---------------------------------------------------------------------------
# Parent chunk generators (large logical pieces)
# ---------------------------------------------------------------------------
def _service_parent_chunks(category: dict[str, Any]) -> List[dict[str, Any]]:
    """Create parent chunks for each service in a category.

    Each service may be split into multiple parent chunks if its full text
    exceeds the *parent* token limit.
    """
    chunks: List[dict[str, Any]] = []
    for service in category["services"]:
        parts = split_service_text(
            _service_text(service), load_config().parent_max_tokens
        )
        base_id = f"service:{_slug(category['name'])}:{_slug(service['name'])}"
        for idx, txt in enumerate(parts, start=1):
            chunk_id = base_id if len(parts) == 1 else f"{base_id}:part-{idx}"
            chunks.append(
                _chunk(
                    chunk_id=chunk_id,
                    text=txt,
                    chunk_type="service",
                    category=category["name"],
                    service_name=service["name"],
                    concept_group=None,
                    page_start=service["page_start"],
                    page_end=service["page_end"],
                    source_urls=service["source_urls"],
                    parent_id=None,
                )
            )
    return chunks


def _concept_parent_chunks(concepts: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    """Create one parent chunk per conceptual record."""
    chunks: List[dict[str, Any]] = []
    for concept in concepts:
        chunk_id = f"concept:{_slug(concept['concept_group'])}:{_slug(concept['name'])}"
        chunks.append(
            _chunk(
                chunk_id=chunk_id,
                text=_service_text(concept),
                chunk_type="concept",
                category=None,
                service_name=None,
                concept_group=concept["concept_group"],
                page_start=concept["page_start"],
                page_end=concept["page_end"],
                source_urls=concept["source_urls"],
                parent_id=None,
            )
        )
    return chunks


def _table_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Render a markdown table preserving rows and adding a repeated header."""
    def clean(cell: object) -> str:
        return re.sub(r"\\s*\\n\\s*", "<br>", str(cell).strip()).replace("|", "\\|")

    header = "| " + " | ".join(clean(c) for c in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(c) for c in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _table_parent_chunks(tables: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    """Create a parent chunk for each ``Compare AWS`` table."""
    chunks: List[dict[str, Any]] = []
    for table in tables:
        if not re.fullmatch(r"Compare AWS .+ services", table["title"].strip()):
            continue
        txt = f"{table['title'].strip()}\n\n{_table_markdown(table['headers'], table['rows'])}"
        chunk_id = f"table:{_slug(table['title'])}:p{table['page_start']}"
        chunks.append(
            _chunk(
                chunk_id=chunk_id,
                text=txt,
                chunk_type="table",
                category=None,
                service_name=None,
                concept_group=None,
                page_start=table["page_start"],
                page_end=table["page_end"],
                source_urls=[],
                parent_id=None,
            )
        )
    return chunks


def _notice_text(table: dict[str, Any]) -> str:
    """Flatten non‑empty notice‑box cells into readable plain text."""
    cells = [
        re.sub(r"\\s+", " ", str(cell)).strip()
        for row in table["rows"]
        for cell in row
        if str(cell).strip()
    ]
    return "\n\n".join(cells)


def _notice_service_context(
    notice_text: str,
    page: int,
    categories: Iterable[dict[str, Any]],
) -> tuple[str | None, str | None, List[str]]:
    """If the notice references exactly one service on ``page``, return its info.
    """
    normalized = _normalise_name(notice_text).casefold()
    candidates = [
        (cat["name"], svc)
        for cat in categories
        for svc in cat["services"]
        if svc["page_start"] <= page <= svc["page_end"]
        and _normalise_name(svc["name"]).casefold() in normalized
    ]
    if len(candidates) == 1:
        cat_name, svc = candidates[0]
        return cat_name, svc["name"], svc["source_urls"]
    return None, None, []


def _notice_parent_chunks(
    tables: Iterable[dict[str, Any]],
    categories: Iterable[dict[str, Any]],
    skipped_matter: Iterable[dict[str, Any]],
) -> List[dict[str, Any]]:
    """Create parent chunks for notice‑box content (non‑table)."""
    cat_list = list(categories)
    skipped = [(e["page_start"], e["page_end"]) for e in skipped_matter]
    chunks: List[dict[str, Any]] = []
    for table in tables:
        title_blank = not table["title"].strip(" |\t\n")
        if table.get("shape") != "unrecognized_table" or not title_blank:
            continue
        if any(start <= table["page_start"] <= end for start, end in skipped):
            continue
        txt = _notice_text(table)
        if not txt:
            continue
        cat, svc_name, src = _notice_service_context(txt, table["page_start"], cat_list)
        chunk_id = f"notice:p{table['page_start']}"
        chunks.append(
            _chunk(
                chunk_id=chunk_id,
                text=txt,
                chunk_type="notice",
                category=cat,
                service_name=svc_name,
                concept_group=None,
                page_start=table["page_start"],
                page_end=table["page_end"],
                source_urls=src,
                parent_id=None,
            )
        )
    return chunks


def _image_caption_parent_chunks(categories: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    """Create a parent chunk for each image‑caption in a category."""
    chunks: List[dict[str, Any]] = []
    for cat in categories:
        for idx, img in enumerate(cat.get("image_caption", []), start=1):
            chunk_id = f"image-caption:{_slug(cat['name'])}:p{img['page']}:part-{idx}"
            chunks.append(
                _chunk(
                    chunk_id=chunk_id,
                    text=img["caption"].strip(),
                    chunk_type="image_caption",
                    category=cat["name"],
                    service_name=None,
                    concept_group=None,
                    page_start=img["page"],
                    page_end=img["page"],
                    source_urls=[],
                    parent_id=None,
                )
            )
    return chunks

# ---------------------------------------------------------------------------
# Child chunk splitter
# ---------------------------------------------------------------------------
def _split_into_children(
    parent: dict[str, Any], child_max_tokens: int, child_overlap: int
) -> List[dict[str, Any]]:
    """Split a parent chunk into child chunks.

    Child chunks inherit all metadata from the parent and receive a ``parent_id``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_max_tokens,
        chunk_overlap=child_overlap,
        length_function=_token_count,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parts = splitter.split_text(parent["text"])
    children: List[dict[str, Any]] = []
    for idx, part in enumerate(parts, start=1):
        child_id = f"{parent['id']}:c{idx}" if len(parts) > 1 else parent["id"]
        child = parent.copy()
        child["id"] = child_id
        child["text"] = part
        child["parent_id"] = parent["id"]
        children.append(child)
    return children

# ---------------------------------------------------------------------------
# Public builder – returns only child chunks (parents are kept only in memory)
# ---------------------------------------------------------------------------
def build_chunks(structure: dict[str, Any], chunk_max_tokens: int) -> List[dict[str, Any]]:
    """Build the complete set of child chunks for the given ``structure``.

    ``chunk_max_tokens`` is interpreted as the *child* token limit. The *parent*
    token limit is taken from ``load_config().parent_max_tokens``. Overlap between
    child chunks defaults to ``load_config().child_overlap`` (or 25 tokens).
    """
    cfg = load_config()
    child_max = chunk_max_tokens
    child_overlap = cfg.child_overlap if hasattr(cfg, "child_overlap") else 25

    categories = structure["categories"]
    # Generate parent chunks for every logical section.
    parent_chunks: List[dict[str, Any]] = []
    for cat in categories:
        parent_chunks.extend(_service_parent_chunks(cat))
    parent_chunks.extend(_concept_parent_chunks(structure.get("concepts", [])))
    parent_chunks.extend(_table_parent_chunks(structure.get("tables", [])))
    parent_chunks.extend(
        _notice_parent_chunks(
            structure.get("tables", []),
            categories,
            structure.get("parsing_report", {}).get("skipped_front_back_matter", []),
        )
    )
    parent_chunks.extend(_image_caption_parent_chunks(categories))

    # Split each parent into child chunks.
    child_chunks: List[dict[str, Any]] = []
    for parent in parent_chunks:
        child_chunks.extend(_split_into_children(parent, child_max, child_overlap))

    # Validate uniqueness of IDs.
    ids = [c["id"] for c in child_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique after child splitting.")
    return child_chunks

# ---------------------------------------------------------------------------
# Verification utilities (unchanged from original module, now operating on child chunks)
# ---------------------------------------------------------------------------
def verify_chunks(chunks: List[dict[str, Any]], structure: dict[str, Any]) -> dict[str, Any]:
    """Return Phase 3 verification data for the generated chunks."""
    services = {
        svc["name"]
        for cat in structure["categories"]
        for svc in cat["services"]
    }
    boundary_failures = [
        c["id"]
        for c in chunks
        if c["chunk_type"] == "service" and c["service_name"] not in services
    ]
    image_categories = {c["category"] for c in chunks if c["chunk_type"] == "image_caption"}
    category_names = {cat["name"] for cat in structure["categories"]}
    return {
        "total_chunks": len(chunks),
        "chunk_type_counts": dict(sorted(Counter(c["chunk_type"] for c in chunks).items())),
        "service_boundary_check_passed": not boundary_failures,
        "service_boundary_failures": boundary_failures,
        "image_caption_categories": sorted(image_categories),
        "categories_without_image_caption": sorted(category_names - image_categories),
        "expected_image_categories_match": {
            _normalise_name(name) for name in image_categories
        } == EXPECTED_IMAGE_CATEGORIES,
        "expected_no_image_categories_match": {
            _normalise_name(name) for name in category_names - image_categories
        } == EXPECTED_NO_IMAGE_CATEGORIES,
    }

