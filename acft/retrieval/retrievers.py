# acft/retrieval/retrievers.py
from __future__ import annotations

import os
from typing import Optional

from acft.embeddings.ollama_embedder import OllamaEmbedder
from acft.retrieval.in_memory import InMemoryRetriever
from acft.retrieval.loader import load_docs_from_folder


def create_default_retriever(
    embedder: OllamaEmbedder,
    rag_folder: Optional[str] = None,
) -> Optional[InMemoryRetriever]:
    """
    Demo helper used by the CLI and examples.

    - Reads folder from:
        1) explicit `rag_folder` argument, OR
        2) ACFT_RAG_FOLDER env var
    - Builds an InMemoryRetriever over those docs.

    Returns None if:
      - folder is missing, OR
      - no docs were loaded.
    """
    folder = rag_folder or os.getenv("ACFT_RAG_FOLDER")
    if not folder:
        return None

    docs = load_docs_from_folder(folder)
    if not docs:
        return None

    return InMemoryRetriever.from_texts(embedder=embedder, docs=docs)


__all__ = ["create_default_retriever"]