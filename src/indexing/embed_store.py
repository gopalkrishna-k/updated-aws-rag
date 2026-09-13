"""embed_store.py — PostgreSQL pgvector Schema Creation & Batch Ingestion.

Creates parent_chunks and child_chunks tables in PostgreSQL with pgvector cosine HNSW index
and full-text search (tsvector GIN index), computes 1024-dim BGE embeddings for child chunks,
and performs batch ingestion.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PARENTS_FILE = Path("data/chunks/parent_chunks.jsonl")
DEFAULT_CHILDREN_FILE = Path("data/chunks/child_chunks.jsonl")

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

DROP_TABLES_SQL = """
DROP TABLE IF EXISTS child_chunks CASCADE;
DROP TABLE IF EXISTS parent_chunks CASCADE;
ALTER_MIGRATION_SQL = """
ALTER TABLE parent_chunks 
    DROP COLUMN IF EXISTS chunk_type,
    DROP COLUMN IF EXISTS header_path,
    DROP COLUMN IF EXISTS source;

ALTER TABLE child_chunks 
    DROP COLUMN IF EXISTS chunk_type,
    DROP COLUMN IF EXISTS header_path,
    DROP COLUMN IF EXISTS source;
"""

CREATE_PARENTS_TABLE_SQL = """
CREATE TABLE parent_chunks (
CREATE TABLE IF NOT EXISTS parent_chunks (
    doc_id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    service_name TEXT NOT NULL,
    header_path TEXT NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'parent',
    text_content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CHILDREN_TABLE_SQL = """
CREATE TABLE child_chunks (
CREATE TABLE IF NOT EXISTS child_chunks (
    doc_id UUID PRIMARY KEY,
    parent_id UUID NOT NULL REFERENCES parent_chunks(doc_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    service_name TEXT NOT NULL,
    header_path TEXT NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'child',
    text_content TEXT NOT NULL,
    embedding vector(1024),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text_content)) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS child_chunks_embedding_hnsw_idx 
ON child_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS child_chunks_tsv_idx 
ON child_chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS child_chunks_parent_id_idx 
ON child_chunks (parent_id);
"""

TRUNCATE_TABLES_SQL = """
TRUNCATE TABLE child_chunks, parent_chunks RESTART IDENTITY CASCADE;
"""


class BGEEmbedder:
    """Embedder wrapper for BAAI/bge-large-en-v1.5 delivering 1024-dimensional normalized vectors."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        logger.info("Loading embedding model '%s'...", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return embeddings.tolist()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_db_connection() -> psycopg2.extensions.connection:
    cfg = load_config()

    # 1. Connect to default postgres DB to ensure target database exists
    try:
        root_conn = psycopg2.connect(
            host=cfg.postgres_host,
            port=cfg.postgres_port,
            dbname="postgres",
            user=cfg.postgres_user,
            password=cfg.postgres_password,
        )
        root_conn.autocommit = True
        with root_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (cfg.postgres_db,))
            if not cur.fetchone():
                logger.info("Database '%s' does not exist. Creating database...", cfg.postgres_db)
                cur.execute(f'CREATE DATABASE "{cfg.postgres_db}";')
        root_conn.close()
    except Exception as exc:
        logger.info("Notice: Root database connection check skipped (%s)", exc)

    # 2. Connect to target project database & enable pgvector extension first
    conn = psycopg2.connect(
        host=cfg.postgres_host,
        port=cfg.postgres_port,
        dbname=cfg.postgres_db,
        user=cfg.postgres_user,
        password=cfg.postgres_password,
    )
    with conn.cursor() as cur:
        cur.execute(CREATE_EXTENSION_SQL)
    conn.commit()

    # 3. Register vector type handler with psycopg2
    register_vector(conn)
    return conn


def init_db_schema(conn: psycopg2.extensions.connection) -> None:
    """Create pgvector extension, drop existing tables, and build parent/child tables and indexes."""
    """Create pgvector extension, ensure tables/indexes exist, and execute ALTER TABLE column removal migration."""
    with conn.cursor() as cur:
        logger.info("Creating pgvector extension if not exists...")
        cur.execute(CREATE_EXTENSION_SQL)
        logger.info("Dropping existing child_chunks and parent_chunks tables...")
        cur.execute(DROP_TABLES_SQL)
        logger.info("Creating parent_chunks table...")
        logger.info("Ensuring parent_chunks and child_chunks tables exist...")
        cur.execute(CREATE_PARENTS_TABLE_SQL)
        logger.info("Creating child_chunks table...")
        cur.execute(CREATE_CHILDREN_TABLE_SQL)
        logger.info("Executing ALTER TABLE column removals in-place (chunk_type, header_path, source)...")
        cur.execute(ALTER_MIGRATION_SQL)
        logger.info("Creating HNSW, GIN (TSV), and parent_id indexes...")
        cur.execute(CREATE_INDEXES_SQL)
        logger.info("Truncating existing data from parent_chunks and child_chunks...")
        cur.execute(TRUNCATE_TABLES_SQL)
    conn.commit()


def embed_and_store(
    parents: List[Dict[str, Any]],
    children: List[Dict[str, Any]],
    embedding_model_name: str = "BAAI/bge-large-en-v1.5",
) -> None:
    conn = get_db_connection()
    try:
        init_db_schema(conn)

        with conn.cursor() as cur:
            # Batch insert parent chunks
            logger.info("Inserting %d parent chunks into PostgreSQL parent_chunks table...", len(parents))
            insert_parent_sql = """
                INSERT INTO parent_chunks (doc_id, source, category, service_name, header_path, chunk_type, text_content)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                INSERT INTO parent_chunks (doc_id, category, service_name, text_content)
                VALUES (%s, %s, %s, %s);
            """
            for p in parents:
                cur.execute(
                    insert_parent_sql,
                    (
                        p["doc_id"],
                        p.get("source", "aws-overview.md"),
                        p.get("category", "General"),
                        p.get("service_name") or p.get("service", "General"),
                        p.get("header_path", "General"),
                        p.get("chunk_type", "parent"),
                        p["text_content"],
                    ),
                )

            # Compute embeddings for child chunks
            logger.info("Generating 1024-dim vector embeddings for %d child chunks...", len(children))
            embedder = BGEEmbedder(model_name=embedding_model_name)
            child_texts = [c["text_content"] for c in children]
            child_embeddings = embedder.embed_documents(child_texts)

            # Batch insert child chunks
            logger.info("Inserting %d child chunks with vector embeddings into PostgreSQL child_chunks table...", len(children))
            insert_child_sql = """
                INSERT INTO child_chunks (doc_id, parent_id, source, category, service_name, header_path, chunk_type, text_content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                INSERT INTO child_chunks (doc_id, parent_id, category, service_name, text_content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            for c, emb in zip(children, child_embeddings):
                cur.execute(
                    insert_child_sql,
                    (
                        c["doc_id"],
                        c["parent_id"],
                        c.get("source", "aws-overview.md"),
                        c.get("category", "General"),
                        c.get("service_name") or c.get("service", "General"),
                        c.get("header_path", "General"),
                        c.get("chunk_type", "child"),
                        c["text_content"],
                        emb,
                    ),
                )

        conn.commit()
        logger.info("Successfully committed all parent and child chunks with 1024-dim embeddings and HNSW/GIN indexes to PostgreSQL!")
    except Exception as exc:
        conn.rollback()
        logger.error("Database transaction failed! Rolled back changes: %s", exc)
        raise exc
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed child chunks and store parents/children in PostgreSQL.")
    parser.add_argument("--parents", type=Path, default=DEFAULT_PARENTS_FILE)
    parser.add_argument("--children", type=Path, default=DEFAULT_CHILDREN_FILE)
    args = parser.parse_args()

    cfg = load_config()
    parents = load_jsonl(args.parents)
    children = load_jsonl(args.children)

    embed_and_store(parents, children, embedding_model_name=cfg.embedding_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

