"""Report likely unspaced English word merges in ``structure.json``."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from src.text_quality import VALIDATION_MIN_RUN_LENGTH, normalise_protected_terms, split_candidate


DEFAULT_INPUT = Path("data/processed/structure.json")
DEFAULT_CHUNKS = Path("data/chunks/chunks.jsonl")
_RUN_PATTERN = re.compile(rf"(?<![A-Za-z0-9])([a-z]{{{VALIDATION_MIN_RUN_LENGTH},}})(?![A-Za-z0-9])")
_SKIP_FIELDS = {"source_document", "source_urls", "image_path"}


def _protected_terms(structure: dict[str, Any]) -> set[str]:
    """Collect corpus product/category labels that must never be split."""
    labels = [
        category["name"] for category in structure.get("categories", [])
    ] + [
        service["name"]
        for category in structure.get("categories", [])
        for service in category.get("services", [])
    ] + [concept["name"] for concept in structure.get("concepts", [])]
    return normalise_protected_terms(labels)


def _text_fields(value: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    """Yield semantic text fields while omitting URLs and file paths."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _SKIP_FIELDS:
                continue
            yield from _text_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _text_fields(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def find_suspected_merges(structure: dict[str, Any]) -> list[dict[str, str]]:
    """Find 12+ lowercase runs with a plausible two-or-more-word split."""
    protected = _protected_terms(structure)
    findings: list[dict[str, str]] = []
    for path, text in _text_fields(structure):
        for match in _RUN_PATTERN.finditer(text):
            token = match.group(1)
            parts = split_candidate(token, protected)
            if parts:
                findings.append(
                    {
                        "path": path,
                        "token": token,
                        "suggested_split": " ".join(parts),
                    }
                )
    return findings


def find_chunk_suspected_merges(
    chunks: Sequence[dict[str, Any]], structure: dict[str, Any]
) -> list[dict[str, str]]:
    """Scan final JSONL chunk text after ``repair_merged_words`` has run."""
    protected = _protected_terms(structure)
    findings: list[dict[str, str]] = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        for match in _RUN_PATTERN.finditer(text):
            token = match.group(1)
            parts = split_candidate(token, protected)
            if parts:
                findings.append(
                    {
                        "path": f"chunks:{chunk.get('id', '<unknown>')}",
                        "token": token,
                        "suggested_split": " ".join(parts),
                    }
                )
    return findings


def _print_findings(label: str, findings: list[dict[str, str]], examples: int) -> None:
    """Print a count and deterministic sample for one validation source."""
    samples = random.Random(42).sample(findings, k=min(examples, len(findings)))
    print(f"{label} suspected merged lowercase runs (length >= {VALIDATION_MIN_RUN_LENGTH}): {len(findings)}")
    for finding in samples:
        print(f"{finding['token']} -> {finding['suggested_split']} ({finding['path']})")


def main(argv: Sequence[str] | None = None) -> int:
    """Print count and ten deterministic examples for manual merge review."""
    parser = argparse.ArgumentParser(description="Validate likely merged English words in structure and final chunks.")
    parser.add_argument("--structure", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args(argv)
    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in args.chunks.read_text(encoding="utf-8").splitlines() if line]
    _print_findings("Structure", find_suspected_merges(structure), args.examples)
    _print_findings("Final chunks", find_chunk_suspected_merges(chunks, structure), args.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
