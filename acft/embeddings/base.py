# acft/embeddings/base.py
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray:
        ...