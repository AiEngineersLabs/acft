# acft/retrieval/base.py
from __future__ import annotations

from typing import Protocol, List, Tuple


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        ...