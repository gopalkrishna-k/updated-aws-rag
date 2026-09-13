"""Parse the AWS Overview PDF into hierarchy-aware structured JSON.

This module deliberately stops at structural extraction. Chunk construction,
image captioning, indexing, retrieval, and generation belong to later phases.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fitz

from src.ingest.extract_tables import extract_tables


DEFAULT_PDF = Path("data/raw/aws-overview.pdf")
DEFAULT_OUTPUT = Path("data/processed/structure.json")
CONCEPT_GROUP_NAMES = {"Deployment models", "Security"}


def clean_extracted_text(value: object) -> str:
    """Normalize PDF text into searchable Unicode without mojibake artifacts.

    PyMuPDF correctly returns Unicode ligatures such as ``ﬃ``. NFKC expands
    those compatibility characters into ordinary ASCII letter sequences. A
    conservative CP1252-to-UTF-8 repair also handles text that has already
    passed through a mojibake-prone display or extraction path.
    """
    text = str(value or "")
    if any(marker in text for marker in ("â", "Ã", "Â", "ï¬")):
        try:
            text = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return unicodedata.normalize("NFKC", text)


def _join_line_spans(spans: list[dict[str, Any]]) -> str:
    """Join adjacent PDF spans while restoring real horizontal word gaps."""
    pieces: list[str] = []
    previous: dict[str, Any] | None = None
    for span in spans:
        text = clean_extracted_text(span.get("text", ""))
        if not text:
            continue
        if previous is not None:
            previous_text = clean_extracted_text(previous.get("text", ""))
            gap = float(span["bbox"][0]) - float(previous["bbox"][2])
            if (
                previous_text
                and not previous_text[-1].isspace()
                and not text[0].isspace()
                and gap >= 0.75
            ):
                pieces.append(" ")
        pieces.append(text)
        previous = span
    return "".join(pieces)


def _is_bold(span: dict[str, Any]) -> bool:
    return bool(span.get("flags", 0) & 16) or "bold" in span.get("font", "").lower()


def _strip_running_matter(lines: list[dict[str, Any]], page_height: float) -> list[dict[str, Any]]:
    """Remove positioned running headers/footers without affecting body headings.

    The publication label is accepted as running matter only in the upper 5% of
    a page. The changing footer title is removed only when it shares the lower
    footer row with a standalone page number; this prevents a service heading
    such as ``Amazon EMR`` in the body from being filtered by its text alone.
    """
    top_boundary = page_height * 0.05
    # This PDF's footer baseline is about 60 points above the page edge, slightly
    # outside a literal five-percent band on its 792-point pages.
    bottom_boundary = page_height - max(page_height * 0.05, 64.0)
    publication_lines = {"overview of amazon web services", "aws whitepaper"}
    excluded: set[int] = {
        index
        for index, line in enumerate(lines)
        if line["bbox"][3] <= top_boundary and line["text"].casefold() in publication_lines
    }

    footer_indexes = [index for index, line in enumerate(lines) if line["bbox"][1] >= bottom_boundary]
    page_number_indexes = [
        index for index in footer_indexes if re.fullmatch(r"\d+", lines[index]["text"])
    ]
    for number_index in page_number_indexes:
        number_y = lines[number_index]["bbox"][1]
        same_row = [
            index
            for index in footer_indexes
            if abs(lines[index]["bbox"][1] - number_y) <= 3
        ]
        # A title plus its standalone number is the footer signature. Keep any
        # unrelated lower-margin text unless that signature is present.
        if any(
            index != number_index
            and not re.fullmatch(r"\d+", lines[index]["text"])
            and len(lines[index]["text"]) <= 120
            for index in same_row
        ):
            excluded.update(same_row)
    return [line for index, line in enumerate(lines) if index not in excluded]


def _line_records(page: fitz.Page, page_number: int, table_regions: list[fitz.Rect]) -> list[dict[str, Any]]:
    """Extract ordered text lines, excluding text physically inside recognized tables."""
    records: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            bbox = fitz.Rect(line["bbox"])
            if any(bbox.intersects(region) for region in table_regions):
                continue
            spans = line.get("spans", [])
            text = _join_line_spans(spans).strip()
            if not text:
                continue
            records.append({
                "text": re.sub(r"\s+", " ", clean_extracted_text(text)),
                "page": page_number,
                "bbox": list(bbox),
                "size": max(float(span.get("size", 0)) for span in spans),
                "bold": any(_is_bold(span) for span in spans),
            })
    records = _strip_running_matter(records, page.rect.height)
    return sorted(records, key=lambda record: (record["bbox"][1], record["bbox"][0]))


def _heading_style_profile(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer recurring category/service styles using the complete document."""
    styles = Counter(round(line["size"], 1) for line in lines if line["bold"] and 1 <= len(line["text"]) <= 120)
    recurring = [size for size, count in styles.items() if count >= 3 and size >= 10]
    # Service names are the dense, repeated tier in this glossary.
    service_size = max(recurring, key=lambda size: (styles[size], -size))
    # The category tier is the nearest recurring larger style, not the largest
    # style in the PDF: title/front-matter headings are intentionally larger.
    category_options = [size for size in recurring if size > service_size and size >= 16]
    if not category_options:
        raise ValueError("Could not infer a recurring category-heading style above service headings.")
    category_size = min(category_options)
    return {
        "category_size": category_size,
        "service_size": service_size,
        "tolerance": 0.35,
        "bold_style_counts": {str(size): styles[size] for size in sorted(styles, reverse=True)},
    }


def _matches_style(line: dict[str, Any], size: float, tolerance: float) -> bool:
    return line["bold"] and abs(line["size"] - size) <= tolerance and len(line["text"]) <= 120


def _links_for_locations(document: fitz.Document, locations: list[dict[str, Any]]) -> list[str]:
    """Return URI links whose rectangles overlap the service's text lines."""
    urls: set[str] = set()
    per_page: dict[int, list[fitz.Rect]] = {}
    for location in locations:
        per_page.setdefault(location["page"], []).append(fitz.Rect(location["bbox"]))
    for page_number, boxes in per_page.items():
        for link in document[page_number - 1].get_links():
            uri, link_box = link.get("uri"), link.get("from")
            if uri and link_box and any(fitz.Rect(link_box).intersects(box) for box in boxes):
                urls.add(uri)
    return sorted(urls)


def _finalize_service(service: dict[str, Any] | None, document: fitz.Document) -> dict[str, Any] | None:
    if service is None:
        return None
    service["description"] = "\n".join(service.pop("_description_lines")).strip()
    service["source_urls"] = _links_for_locations(document, service.pop("_locations"))
    return service


def _finalize_category(category: dict[str, Any] | None, document: fitz.Document) -> dict[str, Any] | None:
    if category is None:
        return None
    service = _finalize_service(category.pop("_active_service", None), document)
    if service and service["description"]:
        category["services"].append(service)
    return category


def _matter_ranges(categories: list[dict[str, Any]], page_count: int) -> list[dict[str, Any]]:
    """Describe ranges outside the first-to-last valid category span."""
    if not categories:
        return [{"kind": "unparsed_document", "page_start": 1, "page_end": page_count}]
    skipped: list[dict[str, Any]] = []
    if categories[0]["page_start"] > 1:
        skipped.append({"kind": "front_matter", "page_start": 1, "page_end": categories[0]["page_start"] - 1})
    if categories[-1]["page_end"] < page_count:
        skipped.append({"kind": "back_matter", "page_start": categories[-1]["page_end"] + 1, "page_end": page_count})
    return skipped


def _separate_concepts(categories: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep Core Concepts entries out of the AWS product-service catalog.

    These headings occur before the service catalog in this source document.
    Their text remains available for later chunking, but their metadata must not
    imply that a deployment model or security overview is an AWS product.
    """
    catalog_categories: list[dict[str, Any]] = []
    concepts: list[dict[str, Any]] = []
    for category in categories:
        if category["name"] in CONCEPT_GROUP_NAMES:
            for entry in category["services"]:
                entry["record_type"] = "concept"
                entry["concept_group"] = category["name"]
                concepts.append(entry)
            continue
        for service in category["services"]:
            service["record_type"] = "service"
        catalog_categories.append(category)
    return catalog_categories, concepts


def parse_pdf(pdf_path: Path) -> dict[str, Any]:
    """Parse categories, services, tables, links, and skipped matter from a PDF."""
    with fitz.open(pdf_path) as document:
        tables, unrecognized_tables = extract_tables(document)
        regions_by_page: dict[int, list[fitz.Rect]] = {}
        for table in tables:
            for region in table.pop("regions"):
                regions_by_page.setdefault(region["page"], []).append(fitz.Rect(region["bbox"]))

        lines = [line for index, page in enumerate(document) for line in _line_records(page, index + 1, regions_by_page.get(index + 1, []))]
        profile = _heading_style_profile(lines)
        categories: list[dict[str, Any]] = []
        current_category: dict[str, Any] | None = None
        skipped_headings: list[dict[str, Any]] = []

        for line in lines:
            if _matches_style(line, profile["category_size"], profile["tolerance"]):
                finished = _finalize_category(current_category, document)
                if finished:
                    if finished["services"]:
                        categories.append(finished)
                    else:
                        skipped_headings.append({"name": finished["name"], "page": finished["page_start"], "reason": "no services detected"})
                current_category = {"name": line["text"], "page_start": line["page"], "page_end": line["page"], "services": [], "_active_service": None}
                continue
            if current_category and _matches_style(line, profile["service_size"], profile["tolerance"]):
                service = _finalize_service(current_category["_active_service"], document)
                if service and service["description"]:
                    current_category["services"].append(service)
                current_category["_active_service"] = {
                    "name": line["text"], "description": "", "page_start": line["page"], "page_end": line["page"], "source_urls": [],
                    "_description_lines": [], "_locations": [line],
                }
                current_category["page_end"] = line["page"]
                continue
            if current_category and current_category["_active_service"]:
                active = current_category["_active_service"]
                active["_description_lines"].append(line["text"])
                active["_locations"].append(line)
                active["page_end"] = line["page"]
                current_category["page_end"] = line["page"]

        finished = _finalize_category(current_category, document)
        if finished:
            if finished["services"]:
                categories.append(finished)
            else:
                skipped_headings.append({"name": finished["name"], "page": finished["page_start"], "reason": "no services detected"})

        skipped_front_back_matter = _matter_ranges(categories, len(document))
        categories, concepts = _separate_concepts(categories)
        service_count = sum(len(category["services"]) for category in categories)
        report = {
            "heading_style_profile": profile,
            "category_count": len(categories),
            "service_count": service_count,
            "concept_count": len(concepts),
            "content_record_count": service_count + len(concepts),
            "extracted_table_count": len(tables),
            "expected_shape_table_count": sum(
                table["shape"] == "category_aws_service" for table in tables
            ),
            "unrecognized_tables": unrecognized_tables,
            "skipped_headings": skipped_headings,
            "skipped_front_back_matter": skipped_front_back_matter,
        }
        return {"source_document": str(pdf_path), "page_count": len(document), "categories": categories, "concepts": concepts, "tables": tables, "parsing_report": report}


def write_structure(structure: dict[str, Any], output_path: Path) -> None:
    """Write extracted structure as readable UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(structure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_report(structure: dict[str, Any]) -> None:
    report = structure["parsing_report"]
    print(f"Categories: {report['category_count']}")
    print(f"Services: {report['service_count']}")
    print(f"Concepts: {report['concept_count']}")
    print(f"Content records: {report['content_record_count']}")
    print(f"Extracted tables: {report['extracted_table_count']}")
    print(f"Expected Category | AWS service tables: {report['expected_shape_table_count']}")
    print(f"Unrecognized tables: {len(report['unrecognized_tables'])}")
    print(f"Skipped front/back matter: {report['skipped_front_back_matter']}")
    print(f"Heading style profile: {report['heading_style_profile']}")
    expected = {"Amazon EMR", "AWS Lambda", "Amazon Aurora"}
    found = {service["name"] for category in structure["categories"] for service in category["services"]}
    print(f"Expected services found: {sorted(expected & found)}")
    print(f"Table titles: {[table['title'] for table in structure['tables']]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Generate ``structure.json`` and optionally print parser diagnostics."""
    parser = argparse.ArgumentParser(description="Parse the AWS Overview PDF into structured JSON.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", action="store_true", help="Print extraction diagnostics.")
    args = parser.parse_args(argv)
    structure = parse_pdf(args.pdf)
    write_structure(structure, args.output)
    if args.report:
        _print_report(structure)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
