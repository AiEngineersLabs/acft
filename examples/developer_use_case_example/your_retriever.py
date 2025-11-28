# your_project/custom_retriever.py
from __future__ import annotations

from typing import List, Tuple

from acft.engine import Retriever, Embedder


class PostgresRetriever(Retriever):
    def __init__(self, embedder: Embedder, conn):
        self.embedder = embedder
        self.conn = conn  # psycopg connection or similar

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        # 1) Embed the query
        q_vec = self.embedder.embed(query)

        # 2) Use your own ANN / cosine similarity logic in DB
        #    Example pseudo-code:
        #    rows = self.conn.execute("SELECT id, text FROM docs ORDER BY embedding <-> %s LIMIT %s", (q_vec, k))
        #    return [(row["id"], row["text"]) for row in rows]

        # For demo purposes, return empty list
        return []