# acft/embeddings/__init__.py
from __future__ import annotations

from .base import Embedder
from .ollama_embedder import OllamaEmbedder

__all__ = ["Embedder", "OllamaEmbedder"]