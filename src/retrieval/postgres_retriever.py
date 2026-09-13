import logging
from pathlib import Path
from typing import List

import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_core.documents import Document

from src.config import load_config
from src.indexing.build_vectorstore import BGEEmbeddings
from src.retrieval.retriever import (
    CustomEnsembleRetriever,
    HybridRetriever,
    load_bm25_from_disk,
    DEFAULT_BM25_FILENAME,
    DEFAULT_CHUNKS,
    load_chunks,
    _build_sibling_index,
    expand_parent_chunks,
)

logger = logging.getLogger(__name__)


class PostgresDenseRetriever:
    """Dense retriever that queries child_chunks in PostgreSQL using pgvector.

    It embeds the query with the same BGEEmbeddings used for indexing and
    performs a similarity search via the ``<=>`` operator provided by pgvector.
    """

    def __init__(self, config=None):
        self.cfg = config or load_config()
        self.embeddings = BGEEmbeddings(model_name=self.cfg.embedding_model)
        self.conn = psycopg2.connect(
            host=self.cfg.postgres_host,
            port=self.cfg.postgres_port,
            dbname=self.cfg.postgres_db,
            user=self.cfg.postgres_user,
            password=self.cfg.postgres_password,
        )
        register_vector(self.conn)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Return top‑k child chunks most similar to *query*.

        Returns :class:`Document` objects with page_content and minimal metadata.
        """
        q_vec = self.embeddings.embed_query(query)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, parent_id, text_content, source, page
                FROM child_chunks
                ORDER BY embedding <=> %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (q_vec, k),
                (str(q_vec), k),
            )
            rows = cur.fetchall()
        docs: List[Document] = []
        for child_id, parent_id, text, source, page in rows:
            metadata = {
                "id": child_id,
                "parent_id": parent_id,
                "source": source,
                "page": page,
            }
            docs.append(Document(page_content=text, metadata=metadata))
        return docs


class PostgresHybridRetriever(HybridRetriever):
    """Hybrid retriever that replaces the Chroma dense store with PostgreSQL."""

    def __init__(self, config=None, dense_weight: float = 0.5, bm25_weight: float = 0.5):
        super().__init__(config=config, dense_weight=dense_weight, bm25_weight=bm25_weight)
        """Initialize a PostgreSQL‑backed hybrid retriever.

        This replaces the Chroma vector store used in the base ``HybridRetriever`` with
        a ``PostgresDenseRetriever``.  We replicate the portions of ``HybridRetriever``
        that are required for BM25 loading, chunk handling and sibling indexing,
        but we **do not** create a ``Chroma`` instance.
        """
        # Load configuration and weights
        if config is None:
            config = load_config()
        self.config = config
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        # 1. BGE embeddings (used by the dense PostgreSQL retriever)
        self._embeddings = BGEEmbeddings(model_name=config.embedding_model)

        # 2. Dense retriever backed by PostgreSQL
        self._dense_retriever = PostgresDenseRetriever(self.config)

        # 3. BM25 retriever – load from configured path
        bm25_path = Path(self.config.vectorstore_path) / DEFAULT_BM25_FILENAME
        self._bm25 = load_bm25_from_disk(bm25_path)

        # 4. Ensemble that combines dense and BM25 with provided weights
        self._ensemble = CustomEnsembleRetriever(
            dense_retriever=self._dense_retriever,
            bm25_retriever=self._bm25,
            weights=[self.dense_weight, self.bm25_weight],
        )

        # 5. Load hierarchical chunks and build sibling index for parent expansion
        self._chunks = load_chunks(DEFAULT_CHUNKS)
        self._sibling_index = _build_sibling_index(self._chunks)


def make_retriever():
    """Factory returning PostgreSQL‑backed retriever when configured."""
    cfg = load_config()
    if getattr(cfg, "postgres_host", None):
        logger.info("Using PostgreSQL‑backed HybridRetriever")
        return PostgresHybridRetriever()
    logger.info("Using default Chroma HybridRetriever")
    return HybridRetriever()

