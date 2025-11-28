# acft/retrieval/loader.py
from __future__ import annotations

import json
import os
from typing import List, Tuple, Optional

from acft.embeddings.ollama_embedder import OllamaEmbedder
from acft.retrieval.in_memory import InMemoryRetriever


def load_docs_from_folder(folder: str) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []

    if not os.path.isdir(folder):
        return docs

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            continue

        if name.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((name, f.read()))
        elif name.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    docs.append((obj.get("id", name), obj.get("text", "")))
        elif name.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, obj in enumerate(data):
                    if isinstance(obj, dict):
                        docs.append(
                            (str(obj.get("id", f"{name}-{i}")), obj.get("text", ""))
                        )

    return docs


def create_default_retriever(
    embedder: OllamaEmbedder,
    rag_folder: Optional[str] = None,
) -> Optional[InMemoryRetriever]:
    folder = rag_folder or os.getenv("ACFT_RAG_FOLDER")
    if not folder:
        return None

    docs = load_docs_from_folder(folder)
    if not docs:
        return None

    return InMemoryRetriever.from_texts(embedder=embedder, docs=docs)