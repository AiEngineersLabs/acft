# acft/retrieval/in_memory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable

import numpy as np

from acft.retrieval.base import Retriever
from acft.embeddings.ollama_embedder import OllamaEmbedder


@dataclass
class InMemoryRetriever(Retriever):
    embedder: OllamaEmbedder
    doc_ids: List[str]
    doc_texts: List[str]
    doc_vectors: np.ndarray  # (N, D)

    @classmethod
    def from_texts(
        cls,
        embedder: OllamaEmbedder,
        docs: Iterable[Tuple[str, str]],
    ) -> "InMemoryRetriever":
        doc_ids: List[str] = []
        doc_texts: List[str] = []
        vectors: List[np.ndarray] = []

        for doc_id, text in docs:
            doc_ids.append(doc_id)
            doc_texts.append(text)
            vec = embedder.embed(text)
            vectors.append(vec)

        if not vectors:
            doc_vectors = np.zeros((0, 1), dtype=float)
        else:
            doc_vectors = np.stack(vectors, axis=0)

        return cls(
            embedder=embedder,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            doc_vectors=doc_vectors,
        )

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        if self.doc_vectors.shape[0] == 0:
            return []

        q_vec = self.embedder.embed(query)
        sims = self.doc_vectors @ q_vec

        k = min(k, len(self.doc_ids))
        idx = np.argsort(-sims)[:k]

        return [(self.doc_ids[i], self.doc_texts[i]) for i in idx]