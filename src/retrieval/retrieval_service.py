"""Multi-Stage Hybrid Retrieval Service for AWS Overview RAG.

Stages:
1. Query Pre-Processing & Intent Routing (Query expansion, comparison/technical/summary detection)
2. Parallel Hybrid Retrieval on PostgreSQL child_chunks (Dense vector + Sparse tsvector search)
3. Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
4. Parent Context Resolution & Prompt Context Assembly
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import load_config

logger = logging.getLogger(__name__)

# ACRONYM & SYNONYM MAP FOR QUERY EXPANSION
ACRONYM_EXPANSIONS: Dict[str, str] = {
    "ec2": "Amazon Elastic Compute Cloud (EC2)",
    "s3": "Amazon Simple Storage Service (S3)",
    "emr": "Amazon Elastic MapReduce (EMR)",
    "mwaa": "Amazon Managed Workflows for Apache Airflow (MWAA)",
    "eks": "Amazon Elastic Kubernetes Service (EKS)",
    "ecs": "Amazon Elastic Container Service (ECS)",
    "rds": "Amazon Relational Database Service (RDS)",
    "dynamodb": "Amazon DynamoDB NoSQL database",
    "lambda": "AWS Lambda serverless compute",
    "vpc": "Amazon Virtual Private Cloud (VPC)",
    "iam": "AWS Identity and Access Management (IAM)",
    "ebs": "Amazon Elastic Block Store (EBS)",
    "efs": "Amazon Elastic File System (EFS)",
    "sqs": "Amazon Simple Queue Service (SQS)",
    "sns": "Amazon Simple Notification Service (SNS)",
    "athena": "Amazon Athena interactive query service",
    "glue": "AWS Glue serverless data integration",
    "kinesis": "Amazon Kinesis real-time data streaming",
}


def preprocess_query(user_query: str) -> Dict[str, Any]:
    """STAGE 1: Query Pre-Processing & Intent Routing.

    Generates alternative query variations and detects query intent:
    - Comparison queries (is_comparison = True)
    - Technical/code/parameter queries (is_technical = True, adjusts BM25/Vector weights)
    - Summary/category queries (is_summary = True)
    """
    query_lower = user_query.strip().lower()

    # 1. Intent Detection
    is_comparison = any(kw in query_lower for kw in [" vs ", " vs.", "compare", "difference", "versus"])

    # Check for technical terms, parameters, numbers, code/model tokens
    tech_patterns = [
        r"\b\d+\b",  # Numbers (ports, sizes, dimensions)
        r"\b[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\b",  # Model / service codes
        r"\_|\.|=|:|\/|\\",  # Code or configuration delimiters
        r"\b(parameter|param|config|memory|cpu|gb|tb|mb|port|api|endpoint|sdk|cli)\b",
    ]
    is_technical = any(re.search(pat, query_lower) for pat in tech_patterns)

    is_summary = any(kw in query_lower for kw in ["overview", "summary", "list all", "what services", "categories", "all services"])

    # Adjust RRF weights based on intent
    if is_technical:
        bm25_weight = 0.7
        vector_weight = 0.3
    else:
        bm25_weight = 0.5
        vector_weight = 0.5

    # 2. Query Expansion (Generate 2 alternative query variations)
    variations: List[str] = []

    # Variation 1: Acronym expansion & AWS context padding
    var1 = user_query
    for acr, exp in ACRONYM_EXPANSIONS.items():
        if re.search(rf"\b{acr}\b", query_lower):
            var1 = re.sub(rf"\b{acr}\b", exp, var1, flags=re.IGNORECASE)

    if var1 != user_query:
        variations.append(var1)
    else:
        variations.append(f"AWS Web Services overview {user_query}")

    # Variation 2: Structural rephrasing based on query intent
    if is_comparison:
        variations.append(f"Detailed comparison of architectural features and differences in {user_query}")
    elif is_summary:
        variations.append(f"Full list and description of AWS services for {user_query}")
    elif is_technical:
        variations.append(f"Technical specifications configuration and parameters for {user_query}")
    else:
        variations.append(f"AWS Cloud platform service capabilities regarding {user_query}")

    # Ensure exactly 2 distinct variations
    variations = list(dict.fromkeys(variations))[:2]

    return {
        "user_query": user_query,
        "query_variations": variations,
        "is_comparison": is_comparison,
        "is_technical": is_technical,
        "is_summary": is_summary,
        "bm25_weight": bm25_weight,
        "vector_weight": vector_weight,
    }


class RetrievalService:
    """Multi-stage hybrid retrieval service communicating directly with PostgreSQL aws_rag_db."""

    def __init__(self, config: Any | None = None):
        self.cfg = config or load_config()

        # Load BGE Embeddings
        logger.info("Initializing SentenceTransformer BGE embedder ('%s')...", self.cfg.embedding_model)
        self.embedder = SentenceTransformer(self.cfg.embedding_model)

        # Lazy load CrossEncoder to minimize startup latency
        self._cross_encoder: CrossEncoder | None = None

        # PostgreSQL Connection Settings
        self.db_config = {
            "host": self.cfg.postgres_host,
            "port": self.cfg.postgres_port,
            "dbname": self.cfg.postgres_db,
            "user": self.cfg.postgres_user,
            "password": self.cfg.postgres_password,
        }

    def _get_connection(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(**self.db_config)
        register_vector(conn)
        return conn

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            logger.info("Loading Cross-Encoder model ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    def _embed_text(self, text: str) -> List[float]:
        emb = self.embedder.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def hybrid_search_child_chunks(
        self,
        queries: List[str],
        dense_limit: int = 20,
        sparse_limit: int = 20,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """STAGE 2: Parallel Hybrid Search against PostgreSQL child_chunks table."""
        dense_results: List[Dict[str, Any]] = []
        sparse_results: List[Dict[str, Any]] = []
        seen_dense: set = set()
        seen_sparse: set = set()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Dense Vector Search for main query and all variations
                for q in queries:
                    q_vec = self._embed_text(q)
                    sql_dense = """
                        SELECT doc_id, parent_id, category, service_name, text_content, 
                               (1 - (embedding <=> %s::vector)) AS score
                        FROM child_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """
                    cur.execute(sql_dense, (str(q_vec), str(q_vec), dense_limit))
                    for row in cur.fetchall():
                        doc_id, parent_id, cat, sname, text, score = row
                        if doc_id not in seen_dense:
                            seen_dense.add(doc_id)
                            dense_results.append(
                                {
                                    "doc_id": str(doc_id),
                                    "parent_id": str(parent_id),
                                    "category": cat,
                                    "service_name": sname,
                                    "text_content": text,
                                    "score": float(score),
                                }
                            )

                # 2. Sparse Keyword Search (tsvector)
                for q in queries:
                    clean_q = re.sub(r"[^\w\s]", " ", q).strip()
                    if not clean_q:
                        continue

                    sql_sparse = """
                        SELECT doc_id, parent_id, category, service_name, text_content,
                               ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS score
                        FROM child_chunks
                        WHERE tsv @@ websearch_to_tsquery('english', %s)
                        ORDER BY score DESC
                        LIMIT %s;
                    """
                    cur.execute(sql_sparse, (clean_q, clean_q, sparse_limit))
                    rows = cur.fetchall()

                    # Fallback to plainto_tsquery if websearch returned no hits
                    if not rows:
                        sql_fallback = """
                            SELECT doc_id, parent_id, category, service_name, text_content,
                                   ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
                            FROM child_chunks
                            WHERE tsv @@ plainto_tsquery('english', %s)
                            ORDER BY score DESC
                            LIMIT %s;
                        """
                        cur.execute(sql_fallback, (clean_q, clean_q, sparse_limit))
                        rows = cur.fetchall()

                    for row in rows:
                        doc_id, parent_id, cat, sname, text, score = row
                        if doc_id not in seen_sparse:
                            seen_sparse.add(doc_id)
                            sparse_results.append(
                                {
                                    "doc_id": str(doc_id),
                                    "parent_id": str(parent_id),
                                    "category": cat,
                                    "service_name": sname,
                                    "text_content": text,
                                    "score": float(score),
                                }
                            )

        finally:
            conn.close()

        return dense_results, sparse_results

    def reciprocal_rank_fusion(
        self,
        dense_list: List[Dict[str, Any]],
        sparse_list: List[Dict[str, Any]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        c: int = 60,
        top_n: int = 15,
    ) -> List[Dict[str, Any]]:
        """STAGE 3A: Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process Dense Ranks
        for rank, item in enumerate(dense_list, start=1):
            did = item["doc_id"]
            doc_map[did] = item
            rrf_scores[did] = rrf_scores.get(did, 0.0) + dense_weight * (1.0 / (c + rank))

        # Process Sparse Ranks
        for rank, item in enumerate(sparse_list, start=1):
            did = item["doc_id"]
            if did not in doc_map:
                doc_map[did] = item
            rrf_scores[did] = rrf_scores.get(did, 0.0) + sparse_weight * (1.0 / (c + rank))

        # Sort by composite RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_n]

        fused_candidates: List[Dict[str, Any]] = []
        for did in sorted_ids:
            cand = dict(doc_map[did])
            cand["rrf_score"] = rrf_scores[did]
            fused_candidates.append(cand)

        return fused_candidates

    def rerank_cross_encoder(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """STAGE 3B: Cross-Encoder Reranking using ms-marco-MiniLM-L-6-v2."""
        if not candidates:
            return []

        pairs = [(query, c["text_content"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["cross_encoder_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]
        return reranked

    def resolve_parent_contexts(
        self,
        child_chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """STAGE 4: Parent Context Resolution & Assembly.

        Extracts parent_id keys, fetches parent records from PostgreSQL parent_chunks table,
        collapses duplicate parents while preserving rank order, and builds prompt context text.
        """
        if not child_chunks:
            return [], "No relevant context found."

        # Extract parent_id list in order of appearance
        ordered_parent_ids: List[str] = []
        for c in child_chunks:
            pid = c["parent_id"]
            if pid not in ordered_parent_ids:
                ordered_parent_ids.append(pid)

        conn = self._get_connection()
        parent_map: Dict[str, Dict[str, Any]] = {}
        try:
            with conn.cursor() as cur:
                sql_parents = """
                    SELECT doc_id, category, service_name, text_content
                    FROM parent_chunks
                    WHERE doc_id = ANY(%s::uuid[]);
                """
                cur.execute(sql_parents, (ordered_parent_ids,))
                for row in cur.fetchall():
                    doc_id, cat, sname, text = row
                    parent_map[str(doc_id)] = {
                        "doc_id": str(doc_id),
                        "category": cat,
                        "service_name": sname,
                        "text_content": text,
                    }
        finally:
            conn.close()

        # Assemble resolved parent list preserving rank order
        resolved_parents: List[Dict[str, Any]] = []
        for pid in ordered_parent_ids:
            if pid in parent_map:
                resolved_parents.append(parent_map[pid])

        # Format context string
        context_blocks: List[str] = []
        for i, p in enumerate(resolved_parents, start=1):
            block = (
                f"--- CONTEXT BLOCK ---\n"
                f"[Category: {p['category']}]\n"
                f"[Service: {p['service_name']}]\n"
                f"{p['text_content'].strip()}\n"
                f"--- END CONTEXT BLOCK ---"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        return resolved_parents, formatted_context

    def retrieve(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute the full 4-stage retrieval pipeline."""
        # STAGE 1: Preprocess & Intent Routing
        intent = preprocess_query(user_query)
        all_queries = [user_query] + intent["query_variations"]

        dense_limit = 30 if intent["is_comparison"] else 20
        sparse_limit = 30 if intent["is_comparison"] else 20

        # STAGE 2: Parallel Hybrid Retrieval
        dense_candidates, sparse_candidates = self.hybrid_search_child_chunks(
            queries=all_queries,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
        )

        # STAGE 3A: RRF Fusion
        fused_15 = self.reciprocal_rank_fusion(
            dense_list=dense_candidates,
            sparse_list=sparse_candidates,
            dense_weight=intent["vector_weight"],
            sparse_weight=intent["bm25_weight"],
            top_n=15,
        )

        # STAGE 3B: Cross-Encoder Reranking
        top_5_children = self.rerank_cross_encoder(
            query=user_query,
            candidates=fused_15,
            top_n=top_k,
        )

        # STAGE 4: Parent Context Resolution
        resolved_parents, formatted_context = self.resolve_parent_contexts(top_5_children)

        return {
            "user_query": user_query,
            "intent": intent,
            "dense_candidates_count": len(dense_candidates),
            "sparse_candidates_count": len(sparse_candidates),
            "fused_candidates_count": len(fused_15),
            "top_5_children": top_5_children,
            "resolved_parents": resolved_parents,
            "formatted_context": formatted_context,
        }

    def retrieve_context(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Convenience alias for retrieve() returning resolved parent chunks as primary payload."""
        return self.retrieve(user_query, top_k=top_k)

