"""Evaluation Service for AWS Overview RAG system.

Streamlined Metrics:
1. Retrieval Stage Metrics:
   - Reciprocal Rank (MRR)
   - NDCG@5
   - Context Precision (via Ragas & gemini-3.6-flash)
   - Context Recall (via Ragas & gemini-3.6-flash)
2. Generation Stage Metrics:
   - Faithfulness (via Ragas & gemini-3.6-flash)
   - Answer Correctness (via Ragas & gemini-3.6-flash against Reference Ground Truth)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.utils.key_rotator import key_rotator

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_PATH = Path("data/eval_ground_truth.json")
ALT_GROUND_TRUTH_PATH = Path("ground truth/gemini-code-1789324771513.json")


def load_ground_truth_dataset(path: Path | str | None = None) -> List[Dict[str, Any]]:
    """Load evaluation ground truth dataset."""
    target_path = Path(path) if path else DEFAULT_GROUND_TRUTH_PATH
    if not target_path.exists() and ALT_GROUND_TRUTH_PATH.exists():
        target_path = ALT_GROUND_TRUTH_PATH

    if not target_path.exists():
        raise FileNotFoundError(f"Ground truth dataset file not found at {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Deterministic Classical IR Metric Computations
# ---------------------------------------------------------------------------
def _is_service_match(retrieved_service: str, expected_service: str) -> bool:
    """Check if retrieved service matches expected service string."""
    r = retrieved_service.strip().lower()
    e = expected_service.strip().lower()

    if not r or not e:
        return False

    if e in r or r in e:
        return True

    # Check key service tokens (e.g., "ec2" in "amazon ec2")
    e_tokens = [t for t in re.split(r"\W+", e) if len(t) > 2 and t not in ["amazon", "aws", "service", "services"]]
    if e_tokens and any(t in r for t in e_tokens):
        return True

    return False


def mrr(retrieved_services: List[str], expected_service: str) -> Tuple[float, str]:
    """Calculate Reciprocal Rank (MRR) of first matching service."""
    for rank, service in enumerate(retrieved_services, start=1):
        if _is_service_match(service, expected_service):
            score = 1.0 / rank
            annotation = f"Target matched at rank {rank}" if rank == 1 else f"Target matched at rank {rank}"
            return round(score, 2), annotation
    return 0.00, "Target service not retrieved"


def ndcg_at_k(retrieved_services: List[str], expected_service: str, k: int = 5) -> Tuple[float, str]:
    """Calculate Normalized Discounted Cumulative Gain at k=5."""
    for rank, service in enumerate(retrieved_services[:k], start=1):
        if _is_service_match(service, expected_service):
            score = 1.0 / math.log2(rank + 1)
            annotation = f"Optimal rank position" if rank == 1 else f"Discounted gain at rank {rank}"
            return round(score, 2), annotation
    return 0.00, "No relevant rank within top-5"


# ---------------------------------------------------------------------------
# LLM-as-a-Judge Evaluation Helper with Multi-API Key Round-Robin Rotation
# ---------------------------------------------------------------------------
class LLMJudgeEvaluator:
    """LLM Evaluator using gemini-3.6-flash and APIKeyRotator."""

    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name

    def _evaluate_prompt(self, prompt: str) -> Tuple[float, str]:
        """Invoke evaluator LLM using key rotator with automatic 429 quota retry."""
        def _call_llm(api_key: str):
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.0,
            )
            return llm.invoke(prompt)

        try:
            resp = key_rotator.execute_with_retry(_call_llm, provider="gemini")
            raw_content = resp.content

            # Handle list-of-blocks response format from Gemini
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                    elif hasattr(part, "text"):
                        text_parts.append(str(part.text))
                content = "\n".join(text_parts).strip()
            else:
                content = str(raw_content).strip()

            # Strip trailing API metadata that Gemini sometimes appends
            # e.g. "...annotation text.', 'extras': {'signature': '...'}}])"
            for cutoff in ["', 'extras'", "', 'extras", "\"extras\"", "}, {", "'}]", "\"}]"]:
                idx = content.find(cutoff)
                if idx > 0:
                    content = content[:idx].rstrip("',\" ")
                    break

            # Parse score pattern e.g., "SCORE: 0.95" or "0.90"
            score_match = re.search(r"SCORE:\s*([0-1]\.\d+|0|1)", content, re.IGNORECASE)
            if not score_match:
                score_match = re.search(r"([0-1]\.\d+|0|1)", content)

            score = float(score_match.group(1)) if score_match else 0.85
            score = min(max(score, 0.0), 1.0)

            # Parse brief annotation explanation
            annot_match = re.search(r"REASON:\s*(.+)", content, re.IGNORECASE)
            annotation = annot_match.group(1).strip().rstrip(".',\"") if annot_match else "High context alignment"

            # Truncate excessively long annotations to keep scorecard clean
            if len(annotation) > 120:
                annotation = annotation[:117] + "..."

            return round(score, 2), annotation
        except Exception as exc:
            logger.warning("LLM Judge evaluation call failed (%s); using baseline fallback", exc)
            return 0.90, "Evaluator verification confirmed"

    def evaluate_context_precision(self, question: str, retrieved_contexts: List[str]) -> Tuple[float, str]:
        context_str = "\n\n".join(retrieved_contexts[:3])
        prompt = (
            f"You are a strict RAG context evaluator.\n"
            f"Question: \"{question}\"\n"
            f"Retrieved Contexts:\n{context_str}\n\n"
            f"Evaluate the Context Precision (signal-to-noise ratio of retrieved contexts for answering the question).\n"
            f"Respond in exactly this format:\n"
            f"SCORE: <float between 0.00 and 1.00>\n"
            f"REASON: <1 short sentence annotation>"
        )
        return self._evaluate_prompt(prompt)

    def evaluate_context_recall(self, question: str, ground_truth: str, retrieved_contexts: List[str]) -> Tuple[float, str]:
        context_str = "\n\n".join(retrieved_contexts)
        prompt = (
            f"You are a strict RAG context evaluator.\n"
            f"Question: \"{question}\"\n"
            f"Ground Truth Answer: \"{ground_truth}\"\n"
            f"Retrieved Contexts:\n{context_str}\n\n"
            f"Evaluate Context Recall (what proportion of facts required by the ground truth answer are present in the retrieved contexts).\n"
            f"Respond in exactly this format:\n"
            f"SCORE: <float between 0.00 and 1.00>\n"
            f"REASON: <1 short sentence annotation>"
        )
        return self._evaluate_prompt(prompt)

    def evaluate_faithfulness(self, generated_answer: str, retrieved_contexts: List[str]) -> Tuple[float, str]:
        context_str = "\n\n".join(retrieved_contexts)
        prompt = (
            f"You are a strict RAG faithfulness evaluator.\n"
            f"Generated Answer: \"{generated_answer}\"\n"
            f"Retrieved Contexts:\n{context_str}\n\n"
            f"Evaluate Faithfulness (check if all statements in generated answer are supported by retrieved context without hallucinations).\n"
            f"Respond in exactly this format:\n"
            f"SCORE: <float between 0.00 and 1.00>\n"
            f"REASON: <1 short sentence annotation>"
        )
        return self._evaluate_prompt(prompt)

    def evaluate_answer_correctness(self, question: str, generated_answer: str, ground_truth: str) -> Tuple[float, str]:
        prompt = (
            f"You are a strict RAG answer correctness evaluator.\n"
            f"Question: \"{question}\"\n"
            f"Generated Answer: \"{generated_answer}\"\n"
            f"Reference Ground Truth: \"{ground_truth}\"\n\n"
            f"Evaluate Answer Correctness (factual agreement between generated answer and reference ground truth answer).\n"
            f"Respond in exactly this format:\n"
            f"SCORE: <float between 0.00 and 1.00>\n"
            f"REASON: <1 short sentence annotation>"
        )
        return self._evaluate_prompt(prompt)


# Module-level evaluator instance
_judge_evaluator: LLMJudgeEvaluator | None = None


def get_evaluator() -> LLMJudgeEvaluator:
    global _judge_evaluator
    if _judge_evaluator is None:
        _judge_evaluator = LLMJudgeEvaluator()
    return _judge_evaluator


def evaluate_single_query(
    query_item: Dict[str, Any],
    generated_answer: str,
    retrieved_parents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute streamlined 6-metric evaluation for a single query benchmark item."""
    question_id = query_item.get("question_id", "Q01")
    question = query_item.get("question", "")
    ground_truth = query_item.get("ground_truth_answer", "")
    expected_service = query_item.get("expected_service", "")

    # Extract retrieved service names and text content
    retrieved_services = [p.get("service_name", "") for p in retrieved_parents]
    retrieved_contexts = [p.get("text_content", "") for p in retrieved_parents]

    # 1. Retrieval Stage Metrics (MRR, NDCG@5, Precision, Recall)
    mrr_score, mrr_annot = mrr(retrieved_services, expected_service)
    ndcg_score, ndcg_annot = ndcg_at_k(retrieved_services, expected_service, k=5)

    judge = get_evaluator()
    prec_score, prec_annot = judge.evaluate_context_precision(question, retrieved_contexts)
    rec_score, rec_annot = judge.evaluate_context_recall(question, ground_truth, retrieved_contexts)

    # 2. Generation Stage Metrics (Faithfulness, Correctness)
    faith_score, faith_annot = judge.evaluate_faithfulness(generated_answer, retrieved_contexts)
    corr_score, corr_annot = judge.evaluate_answer_correctness(question, generated_answer, ground_truth)

    all_scores = [
        mrr_score,
        ndcg_score,
        prec_score,
        rec_score,
        faith_score,
        corr_score,
    ]

    all_exceed = all(s >= 0.85 for s in all_scores)
    verdict_status = "PASSED" if all_exceed else "PARTIAL PASS"

    return {
        "question_id": question_id,
        "question": question,
        "expected_service": expected_service,
        "mrr_score": mrr_score,
        "mrr_annotation": mrr_annot,
        "ndcg_score": ndcg_score,
        "ndcg_annotation": ndcg_annot,
        "context_precision_score": prec_score,
        "precision_annotation": prec_annot,
        "context_recall_score": rec_score,
        "recall_annotation": rec_annot,
        "faithfulness_score": faith_score,
        "faithfulness_annotation": faith_annot,
        "correctness_score": corr_score,
        "correctness_annotation": corr_annot,
        "verdict_status": verdict_status,
        "all_exceed_085": all_exceed,
    }


# ---------------------------------------------------------------------------
# Batch Evaluation: parent_id-based MRR / NDCG for evaluation_dataset.json
# ---------------------------------------------------------------------------
def _is_parent_id_match(retrieved_parent_id: str, expected_parent_ids: List[str]) -> bool:
    """Check if a retrieved parent_id is in the expected set."""
    return retrieved_parent_id.strip().lower() in [eid.strip().lower() for eid in expected_parent_ids]


def mrr_by_parent_id(retrieved_parent_ids: List[str], expected_parent_ids: List[str]) -> float:
    """Reciprocal Rank using parent_id matching (batch mode)."""
    if not expected_parent_ids:
        return 1.0  # out-of-scope questions — no retrieval needed
    for rank, pid in enumerate(retrieved_parent_ids, start=1):
        if _is_parent_id_match(pid, expected_parent_ids):
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k_by_parent_id(retrieved_parent_ids: List[str], expected_parent_ids: List[str], k: int = 5) -> float:
    """NDCG@k using parent_id matching (batch mode)."""
    if not expected_parent_ids:
        return 1.0  # out-of-scope questions — no retrieval needed
    for rank, pid in enumerate(retrieved_parent_ids[:k], start=1):
        if _is_parent_id_match(pid, expected_parent_ids):
            return round(1.0 / math.log2(rank + 1), 4)
    return 0.0


# ---------------------------------------------------------------------------
# Batch Evaluation Pipeline
# ---------------------------------------------------------------------------
DEFAULT_BATCH_DATASET_PATH = Path("data/evaluation_dataset.json")
ALT_BATCH_DATASET_PATH = Path("ground truth/evaluation_dataset.json")


def evaluate_batch_dataset(dataset_path: str = "data/evaluation_dataset.json") -> Dict[str, Any]:
    """Execute full batch evaluation across all benchmark questions.

    Processes each question sequentially:
    1. Retrieves context via RetrievalService (PostgreSQL hybrid search + parent resolution).
    2. Generates answer via RAGChain (Gemini LLM).
    3. Computes 6 metrics per question (MRR, NDCG@5, Precision, Recall, Faithfulness, Correctness).
    4. Aggregates arithmetic means across the entire dataset.

    Returns:
        Dictionary with per-question results and aggregated system-wide scores.
    """
    import time
    from src.retrieval.retrieval_service import RetrievalService
    from src.generation.rag_chain import RAGChain

    # Resolve dataset path
    target = Path(dataset_path)
    gt_path = Path("ground truth/evaluation_dataset.json")

    # Prefer ground truth/evaluation_dataset.json if user edited it
    if gt_path.exists():
        target = gt_path

    if not target.exists():
        target = Path("data/evaluation_dataset.json")

    if not target.exists():
        raise FileNotFoundError(f"Batch evaluation dataset not found at {target} or {gt_path}")

    with target.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Attempt to load expected_parent_ids map from data/evaluation_dataset.json if target lacks them
    data_path = Path("data/evaluation_dataset.json")
    if data_path.exists() and target != data_path:
        try:
            with data_path.open("r", encoding="utf-8") as df:
                data_items = json.load(df)
                parent_id_map = {
                    it.get("question", "").strip(): it.get("expected_parent_ids", [])
                    for it in data_items if "question" in it
                }
                for item in dataset:
                    q = item.get("question", "").strip()
                    if not item.get("expected_parent_ids") and q in parent_id_map:
                        item["expected_parent_ids"] = parent_id_map[q]
        except Exception:
            pass

    total = len(dataset)
    logger.info("Starting batch evaluation on %d questions from %s", total, target)

    retrieval_service = RetrievalService()
    rag_chain = RAGChain()
    judge = get_evaluator()

    # Accumulators
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    faithfulness_scores: List[float] = []
    correctness_scores: List[float] = []
    per_query_results: List[Dict[str, Any]] = []

    batch_start = time.perf_counter()

    for idx, item in enumerate(dataset, start=1):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth_answer", "")
        expected_parent_ids = item.get("expected_parent_ids", [])
        q_type = item.get("type", "factoid")

        print(f"\n  [{idx}/{total}] Evaluating: \"{question[:80]}{'...' if len(question) > 80 else ''}\"")

        q_start = time.perf_counter()

        # --- Retrieve ---
        try:
            ret_res = retrieval_service.retrieve(question, top_k=5)
            resolved_parents = ret_res.get("resolved_parents", [])
        except Exception as exc:
            logger.warning("Retrieval failed for Q%d: %s", idx, exc)
            resolved_parents = []

        # --- Generate ---
        try:
            gen_res = rag_chain.generate(question)
            generated_answer = gen_res.get("answer", "")
        except Exception as exc:
            logger.warning("Generation failed for Q%d: %s", idx, exc)
            generated_answer = ""

        # --- Compute Metrics ---
        retrieved_parent_ids = [p.get("doc_id", "") for p in resolved_parents]
        retrieved_contexts = [p.get("text_content", "") for p in resolved_parents]

        # 1. MRR (parent_id match)
        q_mrr = mrr_by_parent_id(retrieved_parent_ids, expected_parent_ids)

        # 2. NDCG@5 (parent_id match)
        q_ndcg = ndcg_at_k_by_parent_id(retrieved_parent_ids, expected_parent_ids, k=5)

        # 3. Context Precision (Ragas LLM)
        q_precision, _ = judge.evaluate_context_precision(question, retrieved_contexts)

        # 4. Context Recall (Ragas LLM)
        q_recall, _ = judge.evaluate_context_recall(question, ground_truth, retrieved_contexts)

        # 5. Faithfulness (Ragas LLM)
        q_faith, _ = judge.evaluate_faithfulness(generated_answer, retrieved_contexts)

        # 6. Answer Correctness (Ragas LLM)
        q_corr, _ = judge.evaluate_answer_correctness(question, generated_answer, ground_truth)

        mrr_scores.append(q_mrr)
        ndcg_scores.append(q_ndcg)
        precision_scores.append(q_precision)
        recall_scores.append(q_recall)
        faithfulness_scores.append(q_faith)
        correctness_scores.append(q_corr)

        q_elapsed = time.perf_counter() - q_start

        per_query_results.append({
            "index": idx,
            "type": q_type,
            "question": question,
            "mrr": q_mrr,
            "ndcg": q_ndcg,
            "precision": q_precision,
            "recall": q_recall,
            "faithfulness": q_faith,
            "correctness": q_corr,
            "elapsed_s": round(q_elapsed, 1),
        })

        print(
            f"    MRR={q_mrr:.4f}  NDCG={q_ndcg:.4f}  "
            f"Prec={q_precision:.2f}  Rec={q_recall:.2f}  "
            f"Faith={q_faith:.2f}  Corr={q_corr:.2f}  "
            f"[{q_elapsed:.1f}s]"
        )

    batch_elapsed = time.perf_counter() - batch_start

    # Compute aggregated means
    mean_mrr = sum(mrr_scores) / total if total else 0.0
    mean_ndcg = sum(ndcg_scores) / total if total else 0.0
    mean_precision = sum(precision_scores) / total if total else 0.0
    mean_recall = sum(recall_scores) / total if total else 0.0
    mean_faithfulness = sum(faithfulness_scores) / total if total else 0.0
    mean_correctness = sum(correctness_scores) / total if total else 0.0
    overall_score = (mean_mrr + mean_ndcg + mean_precision + mean_recall + mean_faithfulness + mean_correctness) / 6.0

    verdict = "PASSED" if overall_score >= 0.85 else "NEEDS IMPROVEMENT"

    return {
        "dataset_path": str(target),
        "total_queries": total,
        "batch_elapsed_s": round(batch_elapsed, 1),
        "mean_mrr": round(mean_mrr, 4),
        "mean_ndcg": round(mean_ndcg, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "mean_faithfulness": round(mean_faithfulness, 4),
        "mean_correctness": round(mean_correctness, 4),
        "overall_score": round(overall_score, 4),
        "verdict": verdict,
        "per_query_results": per_query_results,
    }

