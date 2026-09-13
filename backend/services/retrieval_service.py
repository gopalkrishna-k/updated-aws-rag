"""Backend service wrapper re-exporting RetrievalService and preprocess_query."""

from __future__ import annotations

from src.retrieval.retrieval_service import RetrievalService, preprocess_query

__all__ = ["RetrievalService", "preprocess_query"]

