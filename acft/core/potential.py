# acft/core/potential.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from acft.neural_operator import LearnablePotentialMLP


@dataclass
class MultiWellPotential:
    """
    Multi-well potential function V(phi) with attractor basins.

    V(phi) = sum_i alpha_i * ||phi - mu_i||^2 + beta * ||phi - mu_i||^4

    - mu_i       : attractor centers (stable meanings)
    - alpha_i    : quadratic weights
    - beta       : higher-order shaping term
    """

    attractors: List[np.ndarray]
    alphas: List[float]
    beta: float = 0.01

    def energy(self, phi: np.ndarray) -> float:
        phi = np.asarray(phi, dtype=float).reshape(-1)
        total = 0.0
        for mu, alpha in zip(self.attractors, self.alphas):
            diff = phi - mu
            r2 = float(np.dot(diff, diff))
            total += alpha * r2 + self.beta * (r2 ** 2)
        return float(total)

    def grad(self, phi: np.ndarray) -> np.ndarray:
        """
        Gradient of V with respect to phi.

        d/dphi[ alpha * ||phi - mu||^2 + beta * ||phi - mu||^4 ]
        = 2 * alpha * (phi - mu) + 4 * beta * ||phi - mu||^2 * (phi - mu)
        """
        phi = np.asarray(phi, dtype=float).reshape(-1)
        grad = np.zeros_like(phi)
        for mu, alpha in zip(self.attractors, self.alphas):
            diff = phi - mu
            r2 = float(np.dot(diff, diff))
            grad += 2.0 * alpha * diff + 4.0 * self.beta * r2 * diff
        return grad


def _normalize_field_states(field_states: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize raw field states into a list of 1D float arrays
    **all with the same dimension**.

    Strategy:
      - convert each state to 1D float
      - drop any empty ones
      - truncate all to the minimum dimensionality

    This makes np.stack() safe and avoids ValueError when some
    embeddings have slightly different shapes.
    """
    flat: List[np.ndarray] = []
    for phi in field_states:
        if phi is None:
            continue
        v = np.asarray(phi, dtype=float).reshape(-1)
        if v.size == 0:
            continue
        flat.append(v)

    if not flat:
        return []

    # Ensure all have same length by truncating to min dimension
    min_dim = min(v.shape[0] for v in flat)
    if min_dim <= 0:
        return []

    normalized = [v[:min_dim] for v in flat]
    return normalized


def build_default_potential(field_states: List[np.ndarray]) -> Optional[MultiWellPotential]:
    """
    Build a simple multi-well potential from the trajectory:

      - Normalize field_states so they all have the same dimension.
      - Attractor 1: mean of first half of states
      - Attractor 2: mean of second half of states (if enough steps)

    If we cannot safely build a potential (e.g., no valid states),
    return None so the engine can skip analytic V(phi) gracefully.
    """
    if not field_states:
        return None

    states = _normalize_field_states(field_states)
    if not states:
        return None

    arr = np.stack(states, axis=0)  # (T, D) guaranteed same shape
    n = arr.shape[0]

    if n == 1:
        mu = arr[0]
        return MultiWellPotential(attractors=[mu], alphas=[1.0], beta=0.01)

    mid = n // 2
    mu1 = np.mean(arr[:mid], axis=0)
    mu2 = np.mean(arr[mid:], axis=0)

    return MultiWellPotential(
        attractors=[mu1, mu2],
        alphas=[1.0, 1.0],
        beta=0.01,
    )


# Optional helper: combine analytic + learned potential
def total_potential_energy(
    phi: np.ndarray,
    analytic: Optional[MultiWellPotential],
    learned: Optional[LearnablePotentialMLP],
) -> float:
    """
    Convenience helper for PDE / diagnostics:
    total energy = V_analytic(phi) + V_learned(phi)
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    total = 0.0
    if analytic is not None:
        total += analytic.energy(phi)
    if learned is not None:
        total += learned.energy(phi)
    return float(total)


__all__ = [
    "MultiWellPotential",
    "build_default_potential",
    "total_potential_energy",
]