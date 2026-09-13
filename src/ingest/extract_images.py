"""Extract category hero images and caption their visible relationships."""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import load_config


DEFAULT_PDF = Path("data/raw/aws-overview.pdf")
DEFAULT_STRUCTURE = Path("data/processed/structure.json")
DEFAULT_IMAGES_DIR = Path("data/processed/images")
DEFAULT_REPORT = Path("data/processed/image_report.json")
MIN_IMAGE_DIMENSION = 150

CAPTION_PROMPT = """You are captioning a diagram from the AWS “Overview of Amazon Web Services” whitepaper for a closed-corpus RAG system.

Describe only information visibly supported by this image. Focus on relationships that may be lost in nearby prose:

1. Identify any ordered process, pipeline, or sequence. State it explicitly using arrows, for example: “ingestion → processing → analytics → warehousing”.
2. Identify groupings, layers, categories, or containment relationships, and list the AWS services/components within each when legible.
3. Describe directional connections, dependencies, inputs, outputs, and data flow.
4. Preserve legible labels exactly where practical.
5. Do not add AWS knowledge that is not visible in the image. If no ordering or relationship is shown, say so.

Return one concise, factual paragraph suitable for retrieval. Do not describe colors, styling, or generic visual appearance unless it conveys a relationship."""


@dataclass(frozen=True)
class ImageRecord:
    """A visible embedded image and its source and rendered dimensions."""

    page: int
    index: int
    data: bytes
    extension: str
    pixel_width: int
    pixel_height: int
    render_width: float
    render_height: float
    y_start: float

    @property
    def passes_size_filter(self) -> bool:
        """Require both source-pixel and rendered page dimensions to be sufficient."""
        return (
            self.pixel_width >= MIN_IMAGE_DIMENSION
            and self.pixel_height >= MIN_IMAGE_DIMENSION
            and self.render_width >= MIN_IMAGE_DIMENSION
            and self.render_height >= MIN_IMAGE_DIMENSION
        )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _mime_type(extension: str) -> str:
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
        extension.lower(), "image/png"
    )


def extract_visible_images(pdf_path: Path) -> Iterator[ImageRecord]:
    """Yield actual page image blocks, excluding inherited but undisplayed resources."""
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            image_index = 0
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 1:
                    continue
                image_index += 1
                bbox = block["bbox"]
                yield ImageRecord(
                    page=page_number,
                    index=image_index,
                    data=block["image"],
                    extension=str(block.get("ext", "png")),
                    pixel_width=int(block.get("width", 0)),
                    pixel_height=int(block.get("height", 0)),
                    render_width=round(bbox[2] - bbox[0], 2),
                    render_height=round(bbox[3] - bbox[1], 2),
                    y_start=round(bbox[1], 2),
                )


def _category_heading_positions(document: fitz.Document, categories: list[dict[str, Any]]) -> list[tuple[int, float, str]]:
    """Locate known category headings so same-page boundaries use vertical position."""
    category_names = {_normalise(category["name"]): category["name"] for category in categories}
    headings: list[tuple[int, float, str]] = []
    for page_number, page in enumerate(document, start=1):
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                category = category_names.get(_normalise(text))
                if category:
                    headings.append((page_number, round(line["bbox"][1], 2), category))
    return sorted(headings)


def attribute_image_categories(pdf_path: Path, structure: dict[str, Any], images: list[ImageRecord]) -> dict[tuple[int, int], str | None]:
    """Assign images to the closest preceding category heading by page and Y-position."""
    with fitz.open(pdf_path) as document:
        headings = _category_heading_positions(document, structure["categories"])

    assignments: dict[tuple[int, int], str | None] = {}
    for image in images:
        preceding = [
            heading
            for heading in headings
            if heading[0] < image.page or (heading[0] == image.page and heading[1] <= image.y_start)
        ]
        assignments[(image.page, image.index)] = preceding[-1][2] if preceding else None
    return assignments


def build_caption_model() -> ChatGoogleGenerativeAI:
    """Create the configured Gemini captioning model from the local environment.

    TODO(provider abstraction): A future provider factory may return any
    vision-capable LangChain chat model with the same ``invoke(messages)`` contract.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required in .env for image captioning.")
    return ChatGoogleGenerativeAI(model=load_config().llm_model, api_key=api_key, temperature=0)


def caption_image(model: ChatGoogleGenerativeAI, image: ImageRecord) -> str:
    """Caption an image using the relationship-focused Gemini prompt.

    Gemini returns structured content parts. Only their user-visible ``text``
    values are persisted; response signatures and provider metadata are never
    written to the corpus.
    """
    encoded_image = base64.b64encode(image.data).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": f"data:{_mime_type(image.extension)};base64,{encoded_image}"},
        ]
    )
    response = model.invoke([message])
    text = _extract_caption_text(response.content)
    if not text:
        raise RuntimeError("Gemini returned no text content for an image caption.")
    return text


def _extract_caption_text(content: object) -> str:
    """Read only visible text from Gemini content, excluding signatures/extras."""
    if isinstance(content, Mapping):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(content, (list, tuple)):
        return " ".join(filter(None, (_extract_caption_text(part) for part in content))).strip()
    if isinstance(content, str):
        # Some adapter versions stringify a structured content part. Parse only
        # that Python-literal wrapper and return its text member, never extras.
        try:
            parsed = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            return content.strip()
        return _extract_caption_text(parsed)
    text = getattr(content, "text", None)
    return text.strip() if isinstance(text, str) else ""


def _report_entry(image: ImageRecord, status: str, reason: str, category: str | None = None) -> dict[str, Any]:
    return {
        "page": image.page,
        "image_index": image.index,
        "pixel_dimensions": [image.pixel_width, image.pixel_height],
        "rendered_dimensions": [image.render_width, image.render_height],
        "status": status,
        "reason": reason,
        "category": category,
    }


def attach_captions(
    structure: dict[str, Any], captions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach image-caption metadata to the appropriate category records."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for caption in captions:
        by_category.setdefault(caption.pop("category"), []).append(caption)
    for category in structure["categories"]:
        category.pop("image_caption", None)
        if category["name"] in by_category:
            category["image_caption"] = by_category[category["name"]]
    return structure


def sanitize_existing_captions(structure_path: Path) -> int:
    """Rewrite persisted caption wrappers as text-only fields without API calls."""
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    updated_count = 0
    for category in structure.get("categories", []):
        for image_caption in category.get("image_caption", []):
            clean_text = _extract_caption_text(image_caption.get("caption", ""))
            if not clean_text:
                raise ValueError(
                    f"Caption on page {image_caption.get('page')} has no extractable text."
                )
            if image_caption["caption"] != clean_text:
                image_caption["caption"] = clean_text
                updated_count += 1
    structure_path.write_text(
        json.dumps(structure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return updated_count


def process_images(
    pdf_path: Path,
    structure_path: Path,
    images_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Extract, caption, attach, and report category hero images."""
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    images = list(extract_visible_images(pdf_path))
    assignments = attribute_image_categories(pdf_path, structure, images)
    images_dir.mkdir(parents=True, exist_ok=True)
    existing_caption_pages = {
        (category["name"], caption["page"])
        for category in structure["categories"]
        for caption in category.get("image_caption", [])
    }

    report_entries: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    model: ChatGoogleGenerativeAI | None = None
    for image in images:
        category = assignments[(image.page, image.index)]
        if not image.passes_size_filter:
            report_entries.append(_report_entry(image, "skipped", "decorative_size", category))
            continue
        if category is None:
            report_entries.append(_report_entry(image, "skipped", "outside_catalog_category", None))
            continue
        if (category, image.page) in existing_caption_pages:
            report_entries.append(_report_entry(image, "skipped", "already_captioned", category))
            continue
        if model is None:
            model = build_caption_model()
        caption = caption_image(model, image)
        image_path = images_dir / f"page-{image.page:03d}-image-{image.index:02d}.{image.extension}"
        image_path.write_bytes(image.data)
        caption_record = {"category": category, "page": image.page, "image_path": str(image_path), "caption": caption}
        captions.append(caption_record)
        category_record = next(item for item in structure["categories"] if item["name"] == category)
        category_record.setdefault("image_caption", []).append(
            {key: value for key, value in caption_record.items() if key != "category"}
        )
        existing_caption_pages.add((category, image.page))
        # Checkpoint every external API result so an interrupted run can resume.
        structure_path.write_text(
            json.dumps(structure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report_entries.append(_report_entry(image, "captioned", "hero_image", category))

    # Captions are already checkpointed above. Retain the helper contract for
    # callers that build captions in memory, without clearing prior checkpoints.
    report = {"minimum_dimension": MIN_IMAGE_DIMENSION, "images": report_entries}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print the complete keep/skip page report for manual validation."""
    for image in report["images"]:
        print(
            f"page {image['page']:>3} image {image['image_index']:>2}: {image['status']} "
            f"({image['reason']}; pixels={image['pixel_dimensions']}; rendered={image['rendered_dimensions']}; "
            f"category={image['category']})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Extract and caption qualifying category hero images."""
    parser = argparse.ArgumentParser(description="Extract and caption AWS Overview hero images.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report", action="store_true", help="Print every keep/skip decision.")
    parser.add_argument(
        "--sanitize-existing",
        action="store_true",
        help="Remove provider metadata from existing captions without calling Gemini.",
    )
    args = parser.parse_args(argv)
    if args.sanitize_existing:
        updated_count = sanitize_existing_captions(args.structure)
        print(f"Sanitized {updated_count} caption fields in {args.structure}")
        return 0
    report = process_images(args.pdf, args.structure, args.images_dir, args.report_path)
    if args.report:
        _print_report(report)
    print(f"Wrote {args.structure}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
