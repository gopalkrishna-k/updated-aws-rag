"""ingestion.py — PDF Extraction & Document Sanitization.

Extracts the AWS Whitepaper PDF into clean, structured Markdown while filtering
out non-informational noise and embedding discrete page indicators.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pymupdf4llm
from langchain_core.documents import Document

DEFAULT_PDF = Path("data/raw/aws-overview.pdf")
DEFAULT_OUTPUT = Path("data/processed/aws-overview.md")
TOC_END_PAGE = 10


def clean_page_text(text: str) -> str:
    """Sanitize text by removing recurring headers, footers, copyright disclaimers, and standalone page numbers."""
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # 1. Skip recurring headers / footers
        if stripped in ("Overview of Amazon Web Services", "AWS Whitepaper", "Overview of Amazon Web Services AWS Whitepaper"):
            continue

        # 2. Skip copyright disclaimers
        if re.search(r"Copyright\s+©\s+\d{4}.*Amazon Web Services", stripped, re.IGNORECASE):
            continue
        if stripped.startswith("Copyright ©") or stripped.startswith("©"):
            continue

        # 3. Skip standalone page numbers (e.g. "14")
        if re.fullmatch(r"\d+", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def process_pdf_to_markdown(
    pdf_path: Path = DEFAULT_PDF,
    output_path: Path = DEFAULT_OUTPUT,
    toc_end_page: int = TOC_END_PAGE,
) -> Document:
    """Extract PDF to markdown, filter front matter, sanitize noise, inject page markers, and write output file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    # Extract page chunks using pymupdf4llm
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    cleaned_pages = []

    for idx, page in enumerate(page_chunks):
        # pymupdf4llm page numbers are 0-indexed dicts; resolve physical 1-based page number
        raw_page_num = page.get("page", idx)
        page_num = raw_page_num + 1 if isinstance(raw_page_num, int) else idx + 1

        # Skip Front-Matter / TOC pages (pages 1 to 10)
        if page_num <= toc_end_page:
            continue

        raw_text = page.get("text", "")
        sanitized_text = clean_page_text(raw_text).strip()

        if not sanitized_text:
            continue

        # Inject Discrete Page Marker comment
        page_entry = f"<!-- PAGE_{page_num} -->\n{sanitized_text}"
        cleaned_pages.append(page_entry)

    full_markdown = "\n\n".join(cleaned_pages)

    # Ensure output directory exists and write markdown file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_markdown, encoding="utf-8")
    print(f"Cleaned Markdown successfully written to {output_path} (Pages {toc_end_page + 1}+ included)")

    return Document(
        page_content=full_markdown,
        metadata={"source": str(pdf_path), "output_path": str(output_path)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest AWS Overview PDF to cleaned Markdown.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--toc-end-page", type=int, default=TOC_END_PAGE)
    args = parser.parse_args()

    process_pdf_to_markdown(args.pdf, args.output, args.toc_end_page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

