"""FastAPI REST API Microservice for AWS Overview RAG.

Endpoints:
1. GET  /health -> Quick health status check.
2. POST /query  -> Executes RAG pipeline for natural-language questions.

Run with:
    python api.py
Then open: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path first
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shim must come BEFORE sentence_transformers / retrieval imports
import datasets_shim  # noqa: F401

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.generation.rag_chain import generate_answer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AWS RAG Microservice",
    version="1.0",
    description=(
        "REST API for the AWS Overview RAG pipeline. "
        "Ask any question about AWS services and get a grounded, cited answer."
    ),
)


class QueryRequest(BaseModel):
    question: str


@app.get("/health", summary="Health check")
def health_check():
    """Returns healthy status for quick connection checks."""
    return {"status": "healthy"}


@app.post("/query", summary="Query the RAG pipeline")
def query_rag(request: QueryRequest):
    """
    Accept a natural-language question, run it through the full RAG pipeline
    (PostgreSQL hybrid retrieval → Gemini LLM generation), and return the answer.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = generate_answer(question)
        return {
            "question": request.question,
            "answer": result.get("answer", ""),
            "status": "success",
        }
    except Exception as exc:
        logger.error("Error generating answer for query '%s': %s", question, exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
