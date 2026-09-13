"""Build hierarchy-aware chunks from the parsed AWS Overview structure."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_config


DEFAULT_INPUT = Path("data/processed/structure.json")
DEFAULT_OUTPUT = Path("data/chunks/chunks.jsonl")
EXPECTED_IMAGE_CATEGORIES = {
    "Analytics",
    "Application integration",
    "Cloud Financial Management",
    "Compute",
    "Containers",
    "Databases",
    "Frontend web and mobile services",
    "Internet of Things (IoT)",
    "Machine Learning (ML) and Artificial Intelligence (AI)",
    "Migration and transfer",
    "Networking and content delivery",
    "Security, identity, and compliance",
    "Storage",
}
EXPECTED_NO_IMAGE_CATEGORIES = {
    "Business applications",
    "Developer tools",
    "Management and governance",
    "Media",
}


def _slug(value: str) -> str:
    """Return a deterministic identifier component."""
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold())
    return value.strip("-")


def _normalise_name(value: str) -> str:
    """Normalize PDF ligatures before comparing category labels."""
    return unicodedata.normalize("NFKC", value)


def _token_count(text: str) -> int:
    """Use whitespace-delimited tokens for the configured structural limit."""
    return len(re.findall(r"\S+", text))


def _service_text(record: dict[str, Any]) -> str:
    """Keep a record's name attached to each standalone service/concept chunk."""
    return f"{record['name']}\n\n{record['description'].strip()}".strip()


def split_service_text(text: str, chunk_max_tokens: int) -> list[str]:
    """Split an oversized single-record text without crossing its boundary."""
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
    source_urls: list[str],
) -> dict[str, Any]:
    """Create one JSONL-compatible chunk object."""
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
    }


def _service_chunks(category: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    """Create one or more chunks per service, never combining service records."""
    chunks: list[dict[str, Any]] = []
    for service in category["services"]:
        text_parts = split_service_text(_service_text(service), max_tokens)
        base_id = f"service:{_slug(category['name'])}:{_slug(service['name'])}"
        for index, text in enumerate(text_parts, start=1):
            chunk_id = base_id if len(text_parts) == 1 else f"{base_id}:part-{index}"
            chunks.append(
                _chunk(
                    chunk_id=chunk_id,
                    text=text,
                    chunk_type="service",
                    category=category["name"],
                    service_name=service["name"],
                    concept_group=None,
                    page_start=service["page_start"],
                    page_end=service["page_end"],
                    source_urls=service["source_urls"],
                )
            )
    return chunks


def _concept_chunks(concepts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create one non-product chunk per conceptual record."""
    return [
        _chunk(
            chunk_id=f"concept:{_slug(concept['concept_group'])}:{_slug(concept['name'])}",
            text=_service_text(concept),
            chunk_type="concept",
            category=None,
            service_name=None,
            concept_group=concept["concept_group"],
            page_start=concept["page_start"],
            page_end=concept["page_end"],
            source_urls=concept["source_urls"],
        )
        for concept in concepts
    ]


def _table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Render preserved table rows with a repeated standalone header."""
    def clean(cell: object) -> str:
        return re.sub(r"\s*\n\s*", "<br>", str(cell).strip()).replace("|", "\\|")

    header = "| " + " | ".join(clean(cell) for cell in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def split_table_rows(table: dict[str, Any], chunk_max_tokens: int) -> list[str]:
    """Split a table only between logical rows, repeating its header per piece."""
    title = table["title"].strip()
    headers = table["headers"]
    groups: list[list[list[str]]] = []
    current_rows: list[list[str]] = []
    for row in table["rows"]:
        candidate = current_rows + [row]
        candidate_text = f"{title}\n\n{_table_markdown(headers, candidate)}"
        if current_rows and _token_count(candidate_text) > chunk_max_tokens:
            groups.append(current_rows)
            current_rows = [row]
        else:
            current_rows = candidate
    if current_rows:
        groups.append(current_rows)
    return [f"{title}\n\n{_table_markdown(headers, rows)}" for rows in groups]


def _table_chunks(tables: Iterable[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Create row-bounded chunks only for genuine ``Compare AWS`` tables."""
    chunks: list[dict[str, Any]] = []
    for table in tables:
        if not re.fullmatch(r"Compare AWS .+ services", table["title"].strip()):
            continue
        parts = split_table_rows(table, max_tokens)
        base_id = f"table:{_slug(table['title'])}:p{table['page_start']}"
        for index, text in enumerate(parts, start=1):
            chunk_id = base_id if len(parts) == 1 else f"{base_id}:part-{index}"
            chunks.append(
                _chunk(
                    chunk_id=chunk_id,
                    text=text,
                    chunk_type="table",
                    category=None,
                    service_name=None,
                    concept_group=None,
                    page_start=table["page_start"],
                    page_end=table["page_end"],
                    source_urls=[],
                )
            )
    return chunks


def _notice_text(table: dict[str, Any]) -> str:
    """Flatten nonempty notice-box cells into readable text without table syntax."""
    cells = [
        re.sub(r"\s+", " ", str(cell)).strip()
        for row in table["rows"]
        for cell in row
        if str(cell).strip()
    ]
    return "\n\n".join(cells)


def _notice_service_context(
    notice_text: str,
    page: int,
    categories: Iterable[dict[str, Any]],
) -> tuple[str | None, str | None, list[str]]:
    """Associate a notice only when its text explicitly names a service on that page."""
    normalized_notice = _normalise_name(notice_text).casefold()
    candidates = [
        (category["name"], service)
        for category in categories
        for service in category["services"]
        if service["page_start"] <= page <= service["page_end"]
        and _normalise_name(service["name"]).casefold() in normalized_notice
    ]
    if len(candidates) == 1:
        category, service = candidates[0]
        return category, service["name"], service["source_urls"]
    return None, None, []


def _notice_chunks(
    tables: Iterable[dict[str, Any]],
    categories: Iterable[dict[str, Any]],
    skipped_matter: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve in-corpus notice boxes while excluding reported front/back matter."""
    category_list = list(categories)
    skipped_ranges = [
        (entry["page_start"], entry["page_end"])
        for entry in skipped_matter
    ]
    chunks: list[dict[str, Any]] = []
    for table in tables:
        title_is_blank = not table["title"].strip(" |\t\n")
        if table.get("shape") != "unrecognized_table" or not title_is_blank:
            continue
        if any(start <= table["page_start"] <= end for start, end in skipped_ranges):
            continue
        text = _notice_text(table)
        if not text:
            continue
        category, service_name, source_urls = _notice_service_context(
            text, table["page_start"], category_list
        )
        chunks.append(
            _chunk(
                chunk_id=f"notice:p{table['page_start']}",
                text=text,
                chunk_type="notice",
                category=category,
                service_name=service_name,
                concept_group=None,
                page_start=table["page_start"],
                page_end=table["page_end"],
                source_urls=source_urls,
            )
        )
    return chunks


def _image_caption_chunks(categories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create one chunk per category hero-image caption."""
    chunks: list[dict[str, Any]] = []
    for category in categories:
        for index, image_caption in enumerate(category.get("image_caption", []), start=1):
            chunks.append(
                _chunk(
                    chunk_id=f"image-caption:{_slug(category['name'])}:p{image_caption['page']}:part-{index}",
                    text=image_caption["caption"].strip(),
                    chunk_type="image_caption",
                    category=category["name"],
                    service_name=None,
                    concept_group=None,
                    page_start=image_caption["page"],
                    page_end=image_caption["page"],
                    source_urls=[],
                )
            )
    return chunks


def build_chunks(structure: dict[str, Any], chunk_max_tokens: int) -> list[dict[str, Any]]:
    """Build parent‑child chunks."""
    from .parent_child import build_chunks as parent_child_build_chunks
    return parent_child_build_chunks(structure, chunk_max_tokens)                                                           


def _service_chunks_from_categories(categories: Iterable[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Apply service chunking to all catalog categories."""
    return [chunk for category in categories for chunk in _service_chunks(category, max_tokens)]


def write_chunks(chunks: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write one chunk object per UTF-8 JSONL line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def verify_chunks(chunks: list[dict[str, Any]], structure: dict[str, Any]) -> dict[str, Any]:
    """Return Phase 3 boundary, type, and image-coverage verification data."""
    services = {
        service["name"]
        for category in structure["categories"]
        for service in category["services"]
    }
    boundary_failures = [
        chunk["id"]
        for chunk in chunks
        if chunk["chunk_type"] == "service"
        and chunk["service_name"] not in services
    ]
    image_categories = {
        chunk["category"] for chunk in chunks if chunk["chunk_type"] == "image_caption"
    }
    category_names = {category["name"] for category in structure["categories"]}
    return {
        "total_chunks": len(chunks),
        "chunk_type_counts": dict(sorted(Counter(chunk["chunk_type"] for chunk in chunks).items())),
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


def _print_examples(chunks: list[dict[str, Any]]) -> None:
    """Print one complete representative object per emitted chunk type."""
    for chunk_type in ("service", "concept", "table", "notice", "image_caption"):
        example = next(chunk for chunk in chunks if chunk["chunk_type"] == chunk_type)
        print(f"Example {chunk_type} chunk:")
        print(json.dumps(example, ensure_ascii=False, indent=2))


def _print_report(report: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
    """Print the required Phase 3 verification report."""
    print(f"Total chunks: {report['total_chunks']}")
    print(f"Chunk type breakdown: {report['chunk_type_counts']}")
    print(f"Service boundary check passed: {report['service_boundary_check_passed']}")
    print(f"Image-caption categories ({len(report['image_caption_categories'])}): {report['image_caption_categories']}")
    print(f"No-image categories ({len(report['categories_without_image_caption'])}): {report['categories_without_image_caption']}")
    print(f"Expected image categories match: {report['expected_image_categories_match']}")
    print(f"Expected no-image categories match: {report['expected_no_image_categories_match']}")
    _print_examples(chunks)


def main(argv: Sequence[str] | None = None) -> int:
    """Read structured content, create JSONL chunks, and print validation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build hierarchy-aware AWS Overview chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", action="store_true", help="Print Phase 3 verification details.")
    args = parser.parse_args(argv)
    structure = json.loads(args.input.read_text(encoding="utf-8"))
    chunks = build_chunks(structure, load_config().chunk_max_tokens)
    write_chunks(chunks, args.output)
    if args.report:
        _print_report(verify_chunks(chunks, structure), chunks)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
