"""Configuration loading for the AWS Overview RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings shared by the project's pluggable components."""

    embedding_model: str
    llm_model: str
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    vectorstore_path: str
    top_k: int
    rerank_top_n: int

    # Backward compatibility properties
    @property
    def chunk_max_tokens(self) -> int:
        return self.child_chunk_size

    @property
    def parent_max_tokens(self) -> int:
        return self.parent_chunk_size

    @property
    def child_overlap(self) -> int:
        return self.child_chunk_overlap


def load_config(path: Path = Path("config.yaml")) -> AppConfig:
    """Load and validate the project's YAML configuration file."""
    if not path.exists():
        values = {}
    else:
        with path.open(encoding="utf-8") as config_file:
            values: Any = yaml.safe_load(config_file) or {}

    if not isinstance(values, dict):
        raise ValueError(f"Configuration at {path} must be a YAML mapping.")

    return AppConfig(
        embedding_model=str(values.get("embedding_model", "BAAI/bge-large-en-v1.5")),
        llm_model=str(values.get("llm_model", "gemini-3.6-flash")),
        parent_chunk_size=int(values.get("parent_chunk_size", 2000)),
        parent_chunk_overlap=int(values.get("parent_chunk_overlap", 100)),
        child_chunk_size=int(values.get("child_chunk_size", 400)),
        child_chunk_overlap=int(values.get("child_chunk_overlap", 50)),
        postgres_host=os.environ.get("POSTGRES_HOST", str(values.get("postgres_host", "localhost"))),
        postgres_port=int(os.environ.get("POSTGRES_PORT", values.get("postgres_port", 5432))),
        postgres_db=os.environ.get("POSTGRES_DB", str(values.get("postgres_db", "aws_rag_db"))),
        postgres_user=os.environ.get("POSTGRES_USER", str(values.get("postgres_user", "postgres"))),
        postgres_password=os.environ.get("POSTGRES_PASSWORD", str(values.get("postgres_password", ""))),
        vectorstore_path=str(values.get("vectorstore_path", "chroma_db")),
        top_k=int(values.get("top_k", 5)),
        rerank_top_n=int(values.get("rerank_top_n", 5)),
    )

