# acft/retrieval/__init__.py
from __future__ import annotations

from .base import Retriever
from .in_memory import InMemoryRetriever
from .loader import create_default_retriever, load_docs_from_folder

__all__ = [
    "Retriever",
    "InMemoryRetriever",
    "create_default_retriever",
    "load_docs_from_folder",
]