# acft/core/pde.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from acft.neural_operator import NeuralOperatorMLP, LearnablePotentialMLP
from .potential import MultiWellPotential


@dataclass
class PDEEvolutionConfig:
    """
    Parameters for PDE-style evolution:

    dphi/dt = D * Laplacian(phi) - grad V_total(phi)

    We apply this per state vector phi (treating its dimensions as a 1D lattice).
    """

    diffusion: float = 0.1
    dt: float = 0.05
    num_steps: int = 5


def _laplacian_1d(phi: np.ndarray) -> np.ndarray:
    """
    Simple 1D Laplacian over the embedding dimensions.
    Uses periodic boundaries for simplicity.
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    if phi.size == 0:
        return phi

    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    return left - 2.0 * phi + right


def _normalize_phi_for_pde(
    phi_init: np.ndarray,
    potential: Optional[MultiWellPotential],
    learned_potential: Optional[LearnablePotentialMLP],
) -> np.ndarray:
    """
    Normalize a single state vector phi for PDE evolution so that its
    dimensionality is consistent with the potential's attractors.

    Strategy:
      - Flatten phi to 1D float.
      - Choose a reference dimension:
          * if analytic potential exists: len(first attractor)
          * else if learned potential exists and has an input_dim attribute: use it
          * else: use phi's own length
      - If phi is empty: create zeros(ref_dim).
      - If phi is longer: truncate to ref_dim.
      - If phi is shorter: pad with zeros up to ref_dim.
    """
    phi = np.asarray(phi_init, dtype=float).reshape(-1)

    # Determine reference dimension
    ref_dim: Optional[int] = None

    if potential is not None and potential.attractors:
        ref_dim = int(potential.attractors[0].shape[0])
    elif learned_potential is not None and hasattr(learned_potential, "input_dim"):
        try:
            ref_dim = int(getattr(learned_potential, "input_dim"))
        except Exception:
            ref_dim = None

    if ref_dim is None or ref_dim <= 0:
        # Fallback: use phi's own length; if even that is 0, nothing we can do
        ref_dim = phi.shape[0]
        if ref_dim <= 0:
            return np.zeros((0,), dtype=float)

    # Adjust phi to match ref_dim
    if phi.size == 0:
        return np.zeros((ref_dim,), dtype=float)

    if phi.size > ref_dim:
        return phi[:ref_dim]

    if phi.size < ref_dim:
        out = np.zeros((ref_dim,), dtype=float)
        out[: phi.size] = phi
        return out

    return phi


def evolve_field_states_pde(
    field_states: List[np.ndarray],
    pde_config: PDEEvolutionConfig,
    potential: Optional[MultiWellPotential],
    learned_potential: Optional[LearnablePotentialMLP] = None,
    neural_operator: Optional[NeuralOperatorMLP] = None,
) -> Tuple[List[np.ndarray], float, float]:
    """
    Apply PDE-style evolution to each field state:

        phi_{k+1} = phi_k + dt * (D * Laplacian(phi_k) - grad V_total(phi_k))
        V_total = V_analytic + V_learned (if both exist)

    If neural_operator is provided, apply an additional learned correction:

        phi_{k+1} = phi_{k+1} + neural_operator(phi_{k+1})

    Returns:
        evolved_states: list of evolved phi
        avg_energy: average V_total(phi) over all states (after evolution)
        max_energy: max V_total(phi) over all states (after evolution)

    This implementation is **robust to shape mismatches** between:
      - raw embeddings
      - potential attractors
      - learned potential input dims
    """
    if not field_states:
        return [], 0.0, 0.0

    # ---- Force proper numeric types (defensive) ----
    try:
        D = float(pde_config.diffusion)
    except Exception:
        D = 0.1

    try:
        dt = float(pde_config.dt)
    except Exception:
        dt = 0.05

    try:
        num_steps = int(pde_config.num_steps)
    except Exception:
        num_steps = 5

    if num_steps < 0:
        num_steps = 0

    evolved_states: List[np.ndarray] = []
    energies: List[float] = []

    for phi_init in field_states:
        # Normalize phi to a consistent dimension
        phi = _normalize_phi_for_pde(phi_init, potential, learned_potential)

        # If still empty, just skip PDE and keep as-is
        if phi.size == 0:
            evolved_states.append(phi)
            energies.append(0.0)
            continue

        for _ in range(num_steps):
            lap = _laplacian_1d(phi)

            # total grad V(phi): analytic + learned (if present)
            gradV_total = np.zeros_like(phi)

            if potential is not None:
                g = potential.grad(phi)
                g = np.asarray(g, dtype=float).reshape(-1)
                # ensure broadcast-safe
                if g.size == phi.size:
                    gradV_total += g
                else:
                    # truncate/pad gradient to match phi
                    if g.size > phi.size:
                        gradV_total += g[: phi.size]
                    else:
                        tmp = np.zeros_like(phi)
                        tmp[: g.size] = g
                        gradV_total += tmp

            if learned_potential is not None:
                g_l = learned_potential.grad(phi)
                g_l = np.asarray(g_l, dtype=float).reshape(-1)
                if g_l.size == phi.size:
                    gradV_total += g_l
                else:
                    if g_l.size > phi.size:
                        gradV_total += g_l[: phi.size]
                    else:
                        tmp = np.zeros_like(phi)
                        tmp[: g_l.size] = g_l
                        gradV_total += tmp

            # PDE update
            phi = phi + dt * (D * lap - gradV_total)

            # Neural operator correction (optional)
            if neural_operator is not None:
                delta_nn = neural_operator(phi)
                delta_nn = np.asarray(delta_nn, dtype=float).reshape(-1)
                if delta_nn.size == phi.size:
                    phi = phi + delta_nn
                else:
                    if delta_nn.size > phi.size:
                        phi = phi + delta_nn[: phi.size]
                    else:
                        tmp = np.zeros_like(phi)
                        tmp[: delta_nn.size] = delta_nn
                        phi = phi + tmp

        evolved_states.append(phi)

        # total energy: analytic + learned (if present)
        total_energy = 0.0
        if potential is not None:
            total_energy += potential.energy(phi)
        if learned_potential is not None:
            total_energy += learned_potential.energy(phi)
        energies.append(total_energy)

    if energies:
        avg_energy = float(sum(energies) / len(energies))
        max_energy = float(max(energies))
    else:
        avg_energy = 0.0
        max_energy = 0.0

    return evolved_states, avg_energy, max_energy


__all__ = ["PDEEvolutionConfig", "evolve_field_states_pde"]