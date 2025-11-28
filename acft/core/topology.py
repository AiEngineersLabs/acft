from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def _vector_norm(v: np.ndarray) -> float:
    """
    Safe L2 norm:
    - converts to float array
    - replaces NaN / inf with 0
    - always returns a finite float
    """
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    return float(np.linalg.norm(v))


@dataclass
class TopologySummary:
    n_components: int
    n_loops: int
    euler_characteristic: int


class TopologyAnalyzer:
    """
    Very simple topology / homology-style analyzer H(phi).

    We treat the sequence of field states as nodes in a graph.
    - Nodes: each time step
    - Edges: between states with distance < epsilon

    Then:
    - connected components ~ Betti_0
    - loops ~ Betti_1 approx using E - N + C
    """

    def __init__(self, epsilon: float = 0.7):
        self.epsilon = epsilon

    def analyze(self, field_states: List[np.ndarray]) -> TopologySummary:
        n = len(field_states)
        if n == 0:
            return TopologySummary(n_components=0, n_loops=0, euler_characteristic=0)
        if n == 1:
            return TopologySummary(n_components=1, n_loops=0, euler_characteristic=1)

        # Build adjacency matrix based on distance threshold
        dist = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = _vector_norm(field_states[i] - field_states[j])
                dist[i, j] = d
                dist[j, i] = d

        adjacency = dist < self.epsilon
        np.fill_diagonal(adjacency, False)

        # Connected components via DFS / BFS
        visited = [False] * n
        n_components = 0
        edges_nonlocal = [0]

        def dfs(start: int) -> None:
            stack = [start]
            visited[start] = True
            while stack:
                u = stack.pop()
                for v in range(n):
                    if adjacency[u, v]:
                        edges_nonlocal[0] += 1
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)

        for i in range(n):
            if not visited[i]:
                n_components += 1
                dfs(i)

        # Each undirected edge counted twice in DFS scanning (u->v and v->u)
        edges = edges_nonlocal[0] // 2

        # Simple Euler characteristic: chi = N - E + C
        euler = n - edges + n_components
        # Approx Betti_1 ~ loops = E - N + C
        n_loops = max(0, edges - n + n_components)

        return TopologySummary(
            n_components=n_components,
            n_loops=n_loops,
            euler_characteristic=euler,
        )


__all__ = ["TopologyAnalyzer", "TopologySummary"]