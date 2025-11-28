# acft/embeddings/ollama_embedder.py
from __future__ import annotations

import requests
import numpy as np

from acft.embeddings.base import Embedder


class OllamaEmbedder(Embedder):
    """
    Minimal Ollama embedder wrapper.
    Uses /api/embeddings with model name (e.g. 'nomic-embed-text').
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> np.ndarray:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding") or data.get("embeddings")
        if isinstance(vec, list):
            return np.array(vec, dtype=float)
        return np.zeros((768,), dtype=float)