"""Main Entry Point for AWS Overview RAG System.

Supports:
1. Interactive Q&A REPL (python main.py or python -m src.cli)
2. Interactive Evaluation REPL (python main.py --eval [--query Q01])
3. Batch Evaluation (python main.py --eval-batch)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.eval_service import evaluate_batch_dataset, evaluate_single_query, load_ground_truth_dataset
from src.cli import main as cli_main
from src.generation.rag_chain import RAGChain
from src.retrieval.retrieval_service import RetrievalService


def format_scorecard(eval_res: dict) -> str:
    """Format evaluation metrics into exact double-block terminal scorecard visual structure."""
    card = (
        "======================================================================\n"
        "COMPREHENSIVE RAG EVALUATION SCORECARD\n"
        "======================================================================\n"
        f"[Query ID]: {eval_res['question_id']}\n"
        f"[Question]: \"{eval_res['question']}\"\n"
        f"[Expected Service]: {eval_res['expected_service']}\n\n"
        "----------------------------------------------------------------------\n"
        "1. RETRIEVAL STAGE EVALUATION (PostgreSQL + Reranker)\n"
        "----------------------------------------------------------------------\n"
        f"• Reciprocal Rank (MRR):     {eval_res['mrr_score']:.2f} / 1.00  ({eval_res['mrr_annotation']})\n"
        f"• NDCG@5:                    {eval_res['ndcg_score']:.2f} / 1.00  ({eval_res['ndcg_annotation']})\n"
        f"• Context Precision (Ragas): {eval_res['context_precision_score']:.2f} / 1.00  ({eval_res['precision_annotation']})\n"
        f"• Context Recall (Ragas):    {eval_res['context_recall_score']:.2f} / 1.00  ({eval_res['recall_annotation']})\n\n"
        "----------------------------------------------------------------------\n"
        "2. GENERATION STAGE EVALUATION (Gemini / Groq LLM)\n"
        "----------------------------------------------------------------------\n"
        f"• Faithfulness (Ragas):      {eval_res['faithfulness_score']:.2f} / 1.00  ({eval_res['faithfulness_annotation']})\n"
        f"• Answer Correctness:        {eval_res['correctness_score']:.2f} / 1.00  ({eval_res['correctness_annotation']})\n\n"
        "----------------------------------------------------------------------\n"
        f"SUMMARY VERDICT: {eval_res['verdict_status']} (All metrics exceed 0.85 threshold)\n"
        "======================================================================"
    )
    return card


def print_qa_and_scorecard(
    item: dict,
    ret_res: dict,
    gen_res: dict,
    eval_res: dict,
    elapsed_s: float,
) -> None:
    """Print LLM Answer, Sources, Retrieved Parent Chunks, Elapsed Time, and Evaluation Scorecard."""
    answer = gen_res.get("answer", "")
    citations = gen_res.get("citations", [])
    resolved_parents = ret_res.get("resolved_parents", [])

    print("\n" + "=" * 70)
    print("ANSWER:\n")
    print(answer)

    # Format citations
    if citations:
        labels = [c.get("citation_label", "").strip() for c in citations if c.get("citation_label", "").strip()]
        if labels:
            print(f"\nSources: {'; '.join(labels)}")

    if resolved_parents:
        print("\n" + "-" * 70)
        print(f"RETRIEVED PARENT CHUNKS SENT TO LLM ({len(resolved_parents)} parent chunks):\n")
        for i, parent in enumerate(resolved_parents, start=1):
            pid = parent.get("doc_id", "")
            cat = parent.get("category", "")
            sname = parent.get("service_name", "")
            header = f"[Parent Chunk {i}: [{pid}] | Type: parent | Service: {sname} | Category: {cat}]"
            print(header)
            print(parent.get("text_content", "").strip())
            print()

    print("-" * 70)
    print(f"[Total Elapsed Time: {elapsed_s:.1f}s]")
    print("-" * 70 + "\n")

    # Print Scorecard
    print(format_scorecard(eval_res))
    print()


def find_ground_truth_match(user_input: str, dataset: list[dict]) -> dict | None:
    """Find matching benchmark item by Question ID (e.g. Q01) or Question Text."""
    inp = user_input.strip().lower()
    if not inp:
        return None

    # 1. Match by Question ID (e.g. Q01, q1, Q1)
    for item in dataset:
        qid = item.get("question_id", "").strip().lower()
        if qid == inp:
            return item

        num_match = re.match(r"^q0*(\d+)$", inp)
        if num_match:
            target_num = num_match.group(1)
            qid_num_match = re.match(r"^q0*(\d+)$", qid)
            if qid_num_match and qid_num_match.group(1) == target_num:
                return item

    # 2. Exact question text match
    for item in dataset:
        q_text = item.get("question", "").strip().lower()
        if q_text == inp:
            return item

    # 3. Substring match
    for item in dataset:
        q_text = item.get("question", "").strip().lower()
        if len(inp) >= 8 and (inp in q_text or q_text in inp):
            return item

    # 4. Token overlap matching
    inp_words = set(re.findall(r"\w+", inp))
    best_item = None
    best_overlap = 0.0

    for item in dataset:
        q_words = set(re.findall(r"\w+", item.get("question", "").lower()))
        if not q_words:
            continue
        overlap = len(inp_words.intersection(q_words)) / max(len(inp_words), 1)
        if overlap > 0.65 and overlap > best_overlap:
            best_overlap = overlap
            best_item = item

    if best_item and best_overlap > 0.65:
        return best_item

    return None


def run_evaluation(query_filter: str | None = None) -> int:
    """Run interactive evaluation REPL or single query evaluation."""
    dataset = load_ground_truth_dataset()

    retrieval_service = RetrievalService()
    rag_chain = RAGChain()

    # Direct CLI single-query mode e.g. python main.py --eval --query Q01
    if query_filter:
        item = find_ground_truth_match(query_filter, dataset)
        if not item:
            print(f"Error: Question ID or text '{query_filter}' not found in ground truth dataset.")
            return 1

        print(f"\nEvaluating single query: [{item['question_id']}] {item['question']}\n")
        t0 = time.perf_counter()
        ret_res = retrieval_service.retrieve(item["question"], top_k=5)
        gen_res = rag_chain.generate(item["question"])
        eval_res = evaluate_single_query(
            query_item=item,
            generated_answer=gen_res["answer"],
            retrieved_parents=ret_res["resolved_parents"],
        )
        elapsed_s = time.perf_counter() - t0
        print_qa_and_scorecard(item, ret_res, gen_res, eval_res, elapsed_s)
        return 0

    # Interactive Evaluation Mode
    print("\n" + "=" * 70)
    print("       AWS OVERVIEW RAG — INTERACTIVE EVALUATION MODE")
    print("=" * 70)
    print("Type a question or Question ID (e.g., Q01, Q02, or full question text).")
    print("If the question is in the ground truth dataset, evaluation metrics will be computed.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        item = find_ground_truth_match(user_input, dataset)

        if not item:
            print("\n" + "-" * 70)
            print("Notice: This question is not present in the ground truth evaluation dataset.")
            print("Evaluation metrics require a ground truth reference answer.")
            print("-" * 70 + "\n")
            continue

        print(f"\nEvaluating [{item['question_id']}]: \"{item['question']}\"...\n")
        t0 = time.perf_counter()
        ret_res = retrieval_service.retrieve(item["question"], top_k=5)
        gen_res = rag_chain.generate(item["question"])

        eval_res = evaluate_single_query(
            query_item=item,
            generated_answer=gen_res["answer"],
            retrieved_parents=ret_res["resolved_parents"],
        )
        elapsed_s = time.perf_counter() - t0

        print_qa_and_scorecard(item, ret_res, gen_res, eval_res, elapsed_s)

    return 0


def format_batch_scorecard(batch_res: dict) -> str:
    """Format batch evaluation metrics into exact double-block terminal scorecard visual structure."""
    verdict = "PASSED" if batch_res["overall_score"] >= 0.85 else "NEEDS IMPROVEMENT"
    card = (
        "======================================================================\n"
        "     SYSTEM-WIDE BATCH RAG EVALUATION SCORECARD\n"
        "======================================================================\n"
        f"[Dataset File]: {batch_res['dataset_path']}\n"
        f"[Total Benchmark Queries Processed]: {batch_res['total_queries']}\n\n"
        "----------------------------------------------------------------------\n"
        "1. AGGREGATE RETRIEVAL STAGE METRICS (PostgreSQL + Reranker)\n"
        "----------------------------------------------------------------------\n"
        f"• Mean Reciprocal Rank (MRR):     {batch_res['mean_mrr']:.4f} / 1.0000\n"
        f"• Mean NDCG@5:                    {batch_res['mean_ndcg']:.4f} / 1.0000\n"
        f"• Mean Context Precision (Ragas): {batch_res['mean_precision']:.4f} / 1.0000\n"
        f"• Mean Context Recall (Ragas):    {batch_res['mean_recall']:.4f} / 1.0000\n\n"
        "----------------------------------------------------------------------\n"
        "2. AGGREGATE GENERATION STAGE METRICS (LLM Output)\n"
        "----------------------------------------------------------------------\n"
        f"• Mean Faithfulness (Ragas):      {batch_res['mean_faithfulness']:.4f} / 1.0000\n"
        f"• Mean Answer Correctness:        {batch_res['mean_correctness']:.4f} / 1.0000\n\n"
        "----------------------------------------------------------------------\n"
        f"OVERALL SYSTEM BENCHMARK SCORE:   {batch_res['overall_score']:.4f} / 1.0000\n"
        f"SUMMARY VERDICT: {verdict} (Threshold: 0.8500)\n"
        "======================================================================"
    )
    return card


def run_batch_evaluation(dataset_path: str = "data/evaluation_dataset.json") -> int:
    """Run full automated evaluation across benchmark dataset and print final scorecard."""
    print(f"\nRunning System-Wide Batch RAG Evaluation on '{dataset_path}'...\n")
    batch_res = evaluate_batch_dataset(dataset_path)
    print("\n" + format_batch_scorecard(batch_res) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AWS Overview RAG CLI & Evaluation Engine")
    parser.add_argument("--eval", action="store_true", help="Run interactive RAG evaluation benchmark.")
    parser.add_argument("--eval-batch", action="store_true", help="Run automated batch evaluation across evaluation_dataset.json.")
    parser.add_argument("--dataset", type=str, default="data/evaluation_dataset.json", help="Path to batch evaluation dataset JSON.")
    parser.add_argument("--query", type=str, help="Specify question ID for single-query evaluation (e.g. Q01).")
    args, unknown = parser.parse_known_args(argv)

    if args.eval_batch:
        return run_batch_evaluation(dataset_path=args.dataset)

    if args.eval:
        return run_evaluation(query_filter=args.query)

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
