import argparse
import json
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


DEFAULT_PARENTS_FILE = Path("data/chunks/parent_chunks.jsonl")
DEFAULT_CHILDREN_FILE = Path("data/chunks/child_chunks.jsonl")

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_PARENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    id VARCHAR(64) PRIMARY KEY,
    text_content TEXT NOT NULL,
    category TEXT,
    service TEXT,
    source TEXT,
    page INTEGER
);
"""

CREATE_CHILDREN_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS child_chunks (
    id VARCHAR(64) PRIMARY KEY,
    parent_id VARCHAR(64) REFERENCES parent_chunks(id) ON DELETE CASCADE,
    text_content TEXT NOT NULL,
    embedding VECTOR(1024),
    source TEXT,
    page INTEGER
);
"""

TRUNCATE_SQL = "TRUNCATE TABLE child_chunks, parent_chunks RESTART IDENTITY CASCADE;"


class BGEEmbedder:
    """Embedder wrapper for BAAI/bge-large-en-v1.5 delivering 1024-dimensional normalized vectors."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
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
                print(f"Database '{cfg.postgres_db}' does not exist. Creating database...")
                cur.execute(f'CREATE DATABASE "{cfg.postgres_db}";')
        root_conn.close()
    except Exception as exc:
        print(f"Notice: Root database connection check skipped ({exc})")

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




def init_db(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_EXTENSION_SQL)
        cur.execute(CREATE_PARENTS_TABLE_SQL)
        cur.execute(CREATE_CHILDREN_TABLE_SQL)
    conn.commit()


def embed_and_store(
    parents: List[Dict[str, Any]],
    children: List[Dict[str, Any]],
    embedding_model_name: str = "BAAI/bge-large-en-v1.5",
) -> None:
    conn = get_db_connection()
    try:
        init_db(conn)

        with conn.cursor() as cur:
            # Idempotent cleanup before insert
            cur.execute(TRUNCATE_SQL)

            # Insert parent chunks
            print(f"Inserting {len(parents)} parent chunks into PostgreSQL...")
            insert_parent_sql = """
                INSERT INTO parent_chunks (id, text_content, category, service, source, page)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            for p in parents:
                cur.execute(
                    insert_parent_sql,
                    (p["id"], p["text_content"], p.get("category"), p.get("service"), p.get("source"), p.get("page")),
                )

            # Compute embeddings for child chunks
            print(f"Generating 1024-dim embeddings for {len(children)} child chunks...")
            embedder = BGEEmbedder(model_name=embedding_model_name)
            child_texts = [c["text_content"] for c in children]
            child_embeddings = embedder.embed_documents(child_texts)

            # Insert child chunks
            print(f"Inserting {len(children)} child chunks with embeddings into PostgreSQL...")
            insert_child_sql = """
                INSERT INTO child_chunks (id, parent_id, text_content, embedding, source, page)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            for c, emb in zip(children, child_embeddings):
                cur.execute(
                    insert_child_sql,
                    (c["id"], c["parent_id"], c["text_content"], emb, c.get("source"), c.get("page")),
                )

        conn.commit()
        print("Successfully committed parent and child chunks to PostgreSQL with pgvector!")
    except Exception as exc:
        conn.rollback()
        print(f"Database transaction failed! Rolled back changes: {exc}")
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
