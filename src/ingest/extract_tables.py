"""Extract AWS comparison tables without flattening them into prose."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import fitz


EXPECTED_HEADERS = ("category", "aws service")


def _clean_extracted_text(value: object) -> str:
    """Normalize ligatures and repair UTF-8 text mis-decoded as CP1252."""
    text = str(value or "")
    if any(marker in text for marker in ("â", "Ã", "Â", "ï¬")):
        try:
            text = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return unicodedata.normalize("NFKC", text)


def _join_line_spans(spans: list[dict[str, Any]]) -> str:
    """Join table-title spans while restoring physical word gaps."""
    pieces: list[str] = []
    previous: dict[str, Any] | None = None
    for span in spans:
        text = _clean_extracted_text(span.get("text", ""))
        if not text:
            continue
        if previous is not None:
            prior = _clean_extracted_text(previous.get("text", ""))
            gap = float(span["bbox"][0]) - float(previous["bbox"][2])
            if prior and not prior[-1].isspace() and not text[0].isspace() and gap >= 0.75:
                pieces.append(" ")
        pieces.append(text)
        previous = span
    return "".join(pieces)


def _normalise(value: object) -> str:
    """Return a comparison-friendly representation of a table cell."""
    return re.sub(r"\s+", " ", _clean_extracted_text(value).strip().lower())


def _clean_cell(value: object) -> str:
    """Make a PDF table cell safe and readable in Markdown."""
    text = re.sub(r"\s*\n\s*", "<br>", _clean_extracted_text(value).strip())
    return text.replace("|", "\\|")


def table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Render table rows as a standalone Markdown table."""
    header_line = "| " + " | ".join(_clean_cell(cell) for cell in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_clean_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _title_near_table(page: fitz.Page, bbox: fitz.Rect, headers: list[str]) -> str:
    """Use the closest preceding comparison heading as the human-readable title."""
    candidates: list[tuple[float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = _join_line_spans(line.get("spans", [])).strip()
            line_bbox = fitz.Rect(line["bbox"])
            if text and line_bbox.y1 <= bbox.y0 and "compare" in text.lower():
                candidates.append((bbox.y0 - line_bbox.y1, text))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return " | ".join(headers)


def _is_expected_shape(headers: list[str]) -> bool:
    return len(headers) == 2 and tuple(_normalise(header) for header in headers) == EXPECTED_HEADERS


def _remove_repeated_headers(rows: Iterable[list[str]], headers: list[str]) -> list[list[str]]:
    """Remove repeated header rows from a table continuation."""
    expected = tuple(_normalise(cell) for cell in headers)
    return [
        row
        for row in rows
        if tuple(_normalise(cell) for cell in row) != expected and any(_normalise(cell) for cell in row)
    ]


def merge_continued_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive-page tables with the same expected repeated header."""
    merged: list[dict[str, Any]] = []
    for table in tables:
        if (
            merged
            and table["page_start"] == merged[-1]["page_end"] + 1
            and tuple(_normalise(item) for item in table["headers"])
            == tuple(_normalise(item) for item in merged[-1]["headers"])
        ):
            merged[-1]["rows"].extend(table["rows"])
            merged[-1]["page_end"] = table["page_end"]
            merged[-1]["markdown"] = table_to_markdown(merged[-1]["headers"], merged[-1]["rows"])
            merged[-1]["regions"].extend(table["regions"])
        else:
            merged.append(table)
    return merged


def extract_tables(document: fitz.Document) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract all detected tables and report each nonstandard comparison shape.

    Nonstandard tables are retained so the report never becomes a silent data
    loss path.  Later chunking can decide whether to index a reported shape.
    """
    extracted_tables: list[dict[str, Any]] = []
    unrecognized: list[dict[str, Any]] = []

    for page_index, page in enumerate(document):
        page_number = page_index + 1
        try:
            found_tables = page.find_tables().tables
        except Exception as error:
            unrecognized.append({"page": page_number, "reason": f"find_tables failed: {error}"})
            continue

        for found in found_tables:
            extracted = found.extract()
            headers = [_clean_extracted_text(cell).strip() for cell in (found.header.names or [])]
            rows = [[_clean_extracted_text(cell).strip() for cell in row] for row in extracted]
            if not headers and rows:
                headers, rows = rows[0], rows[1:]
            rows = _remove_repeated_headers(rows, headers)
            is_expected = _is_expected_shape(headers)
            if not is_expected:
                unrecognized.append({"page": page_number, "headers": headers, "bbox": [round(value, 2) for value in found.bbox], "reason": "header is not exactly Category | AWS service"})

            bbox = fitz.Rect(found.bbox)
            extracted_tables.append({
                "title": _title_near_table(page, bbox, headers),
                "page_start": page_number,
                "page_end": page_number,
                "headers": headers,
                "rows": rows,
                "markdown": table_to_markdown(headers, rows),
                "shape": "category_aws_service" if is_expected else "unrecognized_table",
                "regions": [{"page": page_number, "bbox": list(bbox)}],
            })
    return merge_continued_tables(extracted_tables), unrecognized
