"""Grounded answer-generation chain for AWS Overview RAG.

This module exposes ``generate_answer(query: str) -> dict[str, Any]`` using an LCEL chain:
1. Retriever (Phase 5 hybrid retrieval + category filter + parent expansion).
2. Grounded prompt template enforcing closed-corpus answers, strict inline citations,
   and exact refusal ("This isn't covered in the document.").
3. ChatGoogleGenerativeAI (gemini-3.6-flash).
4. Output parser returning structured answer, citations, and chunks_used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import AppConfig, load_config

load_dotenv()

logger = logging.getLogger(__name__)
# Ensure retry warnings from tenacity are captured
logging.getLogger("tenacity").setLevel(logging.INFO)


EXACT_REFUSAL_STRING = "This isn't covered in the document."

RAG_PROMPT_TEMPLATE = """You are a precise, grounded question-answering assistant for the "Overview of Amazon Web Services" whitepaper.

CRITICAL INSTRUCTIONS:
1. Answer the question ONLY using the provided retrieved context below. Do NOT use any general external knowledge about AWS, technology, or other topics beyond what is explicitly written in the context.
2. If the provided context does NOT contain sufficient information to answer the question, or if the subject is not covered in the document (e.g. non-AWS services or topics not in the text), you MUST respond with EXACTLY:
This isn't covered in the document.
Do NOT guess, do NOT hypothesize, do NOT provide partial external knowledge, and do NOT apologize or hedge.
3. Every factual claim in your answer MUST be cited immediately following the claim. Use the citation label specified in the snippet header (e.g. "(Amazon EMR, page 24)", "(Table: Compare AWS compute services, page 42)", or "(Diagram: Containers, page 53)").
4. Keep answers concise, factual, and direct — typically a few sentences to a short paragraph, unless the query asks for a broader comparison or listing across multiple services.

Retrieved Context:
{context}

Question: {question}

Answer:"""


def get_citation_label(doc: Document) -> str:
    """Derive clean human-readable citation label for a document snippet."""
    chunk_type = doc.metadata.get("chunk_type", "service")
    service_name = doc.metadata.get("service_name") or ""
    category = doc.metadata.get("category") or ""
    page_start = doc.metadata.get("page_start", 0)
    page_end = doc.metadata.get("page_end", 0)
    chunk_id = doc.metadata.get("id", "")

    page_str = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"

    if chunk_type == "service" and service_name:
        return f"{service_name}, {page_str}"
    if chunk_type == "table":
        # Extract title from ID if available e.g. table:compare-aws-compute-services:p42
        table_name = "Comparison table"
        if "compare" in chunk_id.lower() and "compute" in chunk_id.lower():
            table_name = "Table: Compare AWS compute services"
        elif ":" in chunk_id:
            table_name = f"Table: {chunk_id.split(':')[1].replace('-', ' ').title()}"
        return f"{table_name}, {page_str}"
    if chunk_type == "image_caption":
        diag_name = f"Diagram: {category}" if category else "Diagram: Architecture"
        return f"{diag_name}, {page_str}"
    if chunk_type == "concept":
        group = doc.metadata.get("concept_group") or "Cloud Computing Concepts"
        return f"Concept: {group}, {page_str}"
    if service_name:
        return f"{service_name}, {page_str}"
    return f"{category or 'AWS Overview'}, {page_str}"


def format_context_docs(docs: list[Document]) -> str:
    """Format retrieved documents into structured context blocks for the prompt."""
    if not docs:
        return "No relevant context found."

    formatted_blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        citation_label = get_citation_label(doc)
        chunk_type = doc.metadata.get("chunk_type", "text")
        header = f"--- [Snippet {i}] Citation format to use: ({citation_label}) | Type: {chunk_type} ---"
        formatted_blocks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_blocks)


def extract_structured_citations(docs: list[Document], answer: str) -> list[dict[str, Any]]:
    """Extract structured metadata records for only the documents actually referenced in the answer."""
    if answer.strip() == EXACT_REFUSAL_STRING:
        return []

    # Find all parenthesized or bracketed blocks in the answer text
    raw_citations: list[str] = []
    raw_citations.extend(re.findall(r"\(([^)]+)\)", answer))
    raw_citations.extend(re.findall(r"\[([^\]]+)\]", answer))

    # Split compound citation strings on semicolons (e.g. "(Service A, page 2; Service B, page 3)")
    cite_segments: list[str] = []
    for raw in raw_citations:
        for seg in raw.split(";"):
            seg_clean = seg.strip().lower()
            if seg_clean:
                cite_segments.append(seg_clean)

    citations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for doc in docs:
        cid = doc.metadata.get("id", "")
        if cid in seen_ids:
            continue

        label = get_citation_label(doc).lower()
        service_name = (doc.metadata.get("service_name") or "").strip().lower()
        category = (doc.metadata.get("category") or "").strip().lower()
        chunk_type = doc.metadata.get("chunk_type", "")
        page_start = doc.metadata.get("page_start", 0)
        page_end = doc.metadata.get("page_end", 0)

        is_cited = False

        # Check against every extracted citation segment
        for seg in cite_segments:
            # 1. Exact or partial service name match
            if service_name and (service_name in seg or seg in service_name):
                is_cited = True
                break

            # 2. Table citation match
            if chunk_type == "table" and ("table" in seg or "compare" in seg):
                if f"page {page_start}" in seg or f"pages {page_start}" in seg or "compute" in seg:
                    is_cited = True
                    break

            # 3. Diagram / Image Caption citation match
            if chunk_type == "image_caption" and ("diagram" in seg or "figure" in seg):
                if category and category in seg:
                    is_cited = True
                    break
                if f"page {page_start}" in seg or f"pages {page_start}" in seg:
                    is_cited = True
                    break

            # 4. Concept citation match
            if chunk_type == "concept" and "concept" in seg:
                is_cited = True
                break
            if chunk_type == "concept":
                group = (doc.metadata.get("concept_group") or "").lower()
                if "concept" in seg or (group and group in seg) or f"page {page_start}" in seg:
                    is_cited = True
                    break


            # 5. Page number match if service keywords match
            if f"page {page_start}" in seg or f"pages {page_start}" in seg:
                if service_name:
                    words = [w for w in service_name.split() if len(w) > 3 and w not in ["amazon", "service", "services"]]
                    if words and any(w in seg for w in words):
                        is_cited = True
                        break

        # Fallback: check if the exact formatted citation label appears anywhere in the raw answer
        if not is_cited and label in answer.lower():
            is_cited = True

        if is_cited:
            seen_ids.add(cid)
            citations.append(
                {
                    "id": cid,
                    "citation_label": get_citation_label(doc),
                    "service_name": doc.metadata.get("service_name") or "",
                    "category": doc.metadata.get("category") or "",
                    "chunk_type": chunk_type,
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )

    return citations


def _clean_llm_response_text(content: Any) -> str:
    """Extract pure plain text from LLM response content, unwrapping list of blocks if present."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
        return "".join(text_parts).strip()
    if hasattr(content, "text"):
        return str(content.text).strip()
    return str(content).strip()


class LLMDiagnosticCallbackHandler(BaseCallbackHandler):
    """Tracks LLM invocations, retry attempts, and errors during generation."""

    def __init__(self) -> None:
        super().__init__()
        self.retry_count = 0
        self.retry_reasons: list[str] = []

    def on_retry(self, retry_state: Any, **kwargs: Any) -> None:
        self.retry_count += 1
        exc_str = ""
        if hasattr(retry_state, "outcome") and retry_state.outcome:
            try:
                exc = retry_state.outcome.exception()
                exc_str = str(exc) if exc else ""
            except Exception:  # noqa: BLE001
                exc_str = "unknown exception"
        self.retry_reasons.append(exc_str or "unspecified retry")
        logger.warning(
            "Gemini API retry attempt #%d triggered. Reason: %s",
            self.retry_count,
            exc_str or "unspecified",
        )


class RAGChain:
    """Encapsulates the grounded LCEL retrieval and generation pipeline."""

    def __init__(
        self,
        retriever: Any | None = None,
        config: AppConfig | None = None,
    ):
        if config is None:
            config = load_config()
        self.config = config

        if retriever is None:
            try:
                from src.retrieval.retrieval_service import RetrievalService
                self.retrieval_service = RetrievalService(config=config)
            except Exception as exc:
                logger.warning("Could not initialize RetrievalService (%s); falling back to HybridRetriever", exc)
                self.retrieval_service = None
                from src.retrieval.postgres_retriever import make_retriever
                retriever = make_retriever()
        else:
            self.retrieval_service = None

        self.retriever = retriever

        from backend.utils.key_rotator import key_rotator
        api_key = key_rotator.get_next_key("gemini")

        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            google_api_key=api_key,
            temperature=0.0,
        )

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        self.parser = StrOutputParser()

        # Build LCEL chain
        self._llm_chain = self.prompt | self.llm | self.parser

    def generate(self, query: str, k: int | None = None) -> dict[str, Any]:
        """Retrieve context and generate grounded answer with citations."""
        t_total_start = time.perf_counter()

        # 1. Retrieve context
        if self.retrieval_service is not None:
            t_ret_start = time.perf_counter()
            ret_res = self.retrieval_service.retrieve(query, top_k=k or 5)
            retrieval_s = time.perf_counter() - t_ret_start
            rerank_s = 0.0

            formatted_context = ret_res["formatted_context"]
            resolved_parents = ret_res["resolved_parents"]
            docs = [
                Document(
                    page_content=parent["text_content"],
                    metadata={
                        "id": parent["doc_id"],
                        "category": parent.get("category", "General"),
                        "service_name": parent.get("service_name", "General"),
                        "chunk_type": "parent",
                    },
                )
                for parent in resolved_parents
            ]
            retrieval_timings = {"retrieval_s": retrieval_s, "rerank_s": rerank_s}
        else:
            docs, retrieval_timings = self.retriever._retrieve_timed(query, k=k)
            formatted_context = format_context_docs(docs)
        context_chars = len(formatted_context)
        context_tokens_approx = max(1, context_chars // 4)

        # 3. LLM Generation (timed + instrumented)
        diag_cb = LLMDiagnosticCallbackHandler()
        t_gen_start = time.perf_counter()

        prompt_val = self.prompt.invoke(
            {
                "context": formatted_context,
                "question": query,
            }
        )

        def _invoke_llm(api_key: str):
            llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=api_key,
                temperature=0.0,
            )
            return llm.invoke(prompt_val, config={"callbacks": [diag_cb]})

        from backend.utils.key_rotator import key_rotator
        raw_response = key_rotator.execute_with_retry(_invoke_llm, provider="gemini")
        generation_s = time.perf_counter() - t_gen_start
        parsed_output = self.parser.invoke(raw_response)
        answer = _clean_llm_response_text(parsed_output)

        # 4. Extract LLM response metadata & token usage
        response_metadata = getattr(raw_response, "response_metadata", {}) or {}
        usage_metadata = getattr(raw_response, "usage_metadata", {}) or {}
        finish_reason = response_metadata.get("finish_reason", "STOP")
        
        token_usage = response_metadata.get("token_usage", {})
        input_tokens = (
            usage_metadata.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("prompt_token_count")
            or "N/A"
        )
        output_tokens = (
            usage_metadata.get("output_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("candidates_token_count")
            or "N/A"
        )
        total_tokens = (
            usage_metadata.get("total_tokens")
            or token_usage.get("total_tokens")
            or token_usage.get("total_token_count")
            or "N/A"
        )

        llm_diagnostics = {
            "context_chars": context_chars,
            "context_tokens_approx": context_tokens_approx,
            "retries": diag_cb.retry_count,
            "retry_reasons": diag_cb.retry_reasons,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "response_metadata": response_metadata,
        }

        logger.info(
            "LLM Diagnostics: context=%d chars (~%d tokens), retries=%d, "
            "tokens [in=%s, out=%s, total=%s], finish_reason=%s, gen_time=%.2fs",
            context_chars,
            context_tokens_approx,
            diag_cb.retry_count,
            input_tokens,
            output_tokens,
            total_tokens,
            finish_reason,
            generation_s,
        )

        # 5. Strict Refusal normalization
        if "not covered in the document" in answer.lower() or "isn't covered in the document" in answer.lower():
            if answer != EXACT_REFUSAL_STRING:
                answer = EXACT_REFUSAL_STRING

        # 6. Extract citations and chunk IDs
        citations = extract_structured_citations(docs, answer)
        chunks_used = [doc.metadata.get("id", "") for doc in docs]
        retrieved_chunks = [
            {
                "id": doc.metadata.get("id", ""),
                "chunk_type": doc.metadata.get("chunk_type", ""),
                "category": doc.metadata.get("category", ""),
                "service_name": doc.metadata.get("service_name", ""),
                "concept_group": doc.metadata.get("concept_group", ""),
                "page_start": doc.metadata.get("page_start", 0),
                "page_end": doc.metadata.get("page_end", 0),
                "text": doc.page_content,
            }
            for doc in docs
        ]

        total_s = time.perf_counter() - t_total_start

        timings = {
            "retrieval_s": retrieval_timings["retrieval_s"],
            "rerank_s": retrieval_timings["rerank_s"],
            "generation_s": generation_s,
            "total_s": total_s,
        }

        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": chunks_used,
            "retrieved_chunks": retrieved_chunks,
            "timings": timings,
            "llm_diagnostics": llm_diagnostics,
        }



# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------
_chain_instance: RAGChain | None = None


def generate_answer(query: str, k: int | None = None) -> dict[str, Any]:
    """Generate a grounded answer with citations for a query.

    Args:
        query: Natural-language question.
        k: Optional number of chunks after reranking.

    Returns:
        Dict with keys:
          - ``answer`` (str): Generated answer text with inline citations or exact refusal.
          - ``citations`` (list[dict]): Structured citation records.
          - ``chunks_used`` (list[str]): List of chunk IDs retrieved as context.
    """
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = RAGChain()
    return _chain_instance.generate(query, k=k)


# ---------------------------------------------------------------------------
# CLI & Self-Verification Test Harness
# ---------------------------------------------------------------------------
def _run_verification() -> bool:
    """Run the 5 required verification queries exactly once and report output."""
    print("=" * 75)
    print("PHASE 6 RAG CHAIN SELF-VERIFICATION REPORT")
    print("=" * 75)

    test_queries = [
        ("1. What is Amazon EMR?", "What is Amazon EMR?"),
        ("2. What does MWAA stand for?", "What does MWAA stand for?"),
        (
            "3. Compare EC2 and Fargate for running containers",
            "Compare EC2 and Fargate for running containers",
        ),
        (
            "4. What analytics services does AWS offer?",
            "What analytics services does AWS offer?",
        ),
        ("5. What is Google BigQuery?", "What is Google BigQuery?"),
    ]

    chain = RAGChain()
    gemini_call_count = 0
    all_passed = True

    for label, query in test_queries:
        print(f"\n--- Test {label} ---")
        gemini_call_count += 1
        result = chain.generate(query)
        answer = result["answer"]
        citations = result["citations"]
        chunks = result["chunks_used"]

        print(f"Query: {query}")
        print(f"Answer:\n{answer}\n")
        print(f"Citations ({len(citations)}):")
        for c in citations[:5]:
            print(f"  - [{c['id']}] {c['citation_label']} ({c['chunk_type']})")
        if len(citations) > 5:
            print(f"  - ... and {len(citations) - 5} more citations")
        print(f"Chunks Used: {chunks[:5]}")

        # Verification rules
        if label.startswith("1."):
            passed = "amazon emr" in answer.lower() and "page" in answer.lower()
            print(f"Passed Check 1 (EMR + page citation): {passed}")
            if not passed:
                all_passed = False
        elif label.startswith("2."):
            passed = (
                "managed workflows for apache airflow" in answer.lower()
                and "page" in answer.lower()
            )
            print(f"Passed Check 2 (MWAA full expansion + citation): {passed}")
            if not passed:
                all_passed = False
        elif label.startswith("3."):
            has_table_or_services = (
                "fargate" in answer.lower()
                and "ec2" in answer.lower()
                and "page" in answer.lower()
            )
            print(f"Passed Check 3 (EC2 vs Fargate comparison + citations): {has_table_or_services}")
            if not has_table_or_services:
                all_passed = False
        elif label.startswith("4."):
            # Check for multiple analytics services in answer
            multiple_services = len(citations) >= 10
            print(f"Passed Check 4 (Broad category listing, {len(citations)} services): {multiple_services}")
            if not multiple_services:
                all_passed = False
        elif label.startswith("5."):
            refused_cleanly = answer.strip() == EXACT_REFUSAL_STRING
            print(f"Passed Check 5 (Exact refusal on ungrounded query): {refused_cleanly}")
            if not refused_cleanly:
                all_passed = False

    print(f"\n{'='*75}")
    print(f"TOTAL GEMINI CALLS IN VERIFICATION: {gemini_call_count}")
    print(f"ALL 5 CHECKS PASSED: {all_passed}")
    print(f"{'='*75}")
    return all_passed


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for interactive query or verification."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Phase 6 Grounded RAG Chain.")
    parser.add_argument("query", nargs="?", help="Query string to answer.")
    parser.add_argument("--verify", action="store_true", help="Run 5 verification queries.")
    args = parser.parse_args(argv)

    if args.verify:
        passed = _run_verification()
        return 0 if passed else 1

    if args.query:
        result = generate_answer(args.query)
        print(f"\nQuery: {args.query}\n")
        print(f"Answer:\n{result['answer']}\n")
        print("Citations:")
        for c in result["citations"]:
            print(f"  - {c['citation_label']} ({c['id']})")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
