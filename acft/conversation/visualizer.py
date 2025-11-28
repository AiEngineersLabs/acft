from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .memory import ConversationMemory, ConversationTurn


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Safe cosine similarity between two vectors, returns None if one is missing
    or if norm is zero.
    """
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return None

    return float(np.dot(a, b) / (na * nb))


class AttractorVisualizer:
    """
    Utility for printing an "attractor view" of the conversation in the CLI.

    For each turn, we show:
      - turn index
      - decision
      - stability
      - cosine similarity of final_phi to previous turn's final_phi
    """

    @staticmethod
    def describe(memory: ConversationMemory) -> List[Dict[str, Optional[float]]]:
        rows: List[Dict[str, Optional[float]]] = []

        prev_phi = None
        for idx, t in enumerate(memory.turns, start=1):
            cos = None
            if prev_phi is not None and t.final_phi is not None:
                cos = _cosine(prev_phi, t.final_phi)

            rows.append(
                {
                    "turn": idx,
                    "decision": t.decision,
                    "stability": t.stability,
                    "cosine_to_prev": cos,
                }
            )

            if t.final_phi is not None:
                prev_phi = t.final_phi

        return rows