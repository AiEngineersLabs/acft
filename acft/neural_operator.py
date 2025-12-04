from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ============================================================
# Simple utilities for demos / training:
#   - SimpleHashEmbedder
#   - SimpleMockLLM
# ============================================================

@dataclass
class SimpleHashEmbedder:
    """
    Very simple deterministic text → vector mapper, used only for demos.

    It uses Python's hash(text) as a seed to generate a random vector,
    then normalizes it. This is NOT a real embedding model; it's just
    for toy training of the potential and neural operator.
    """
    dim: int = 64

    def embed(self, text: str) -> np.ndarray:
        # Stable seed from hash(text)
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)

        v = rng.normal(0.0, 1.0, size=(self.dim,))
        norm = np.linalg.norm(v) + 1e-8
        return v / norm


class SimpleMockLLM:
    """
    Tiny mock "LLM" that returns a few synthetic reasoning steps.

    Used only to generate toy trajectories for training NeuralOperatorMLP:
        prompt -> step1 -> step2 -> ... -> answer
    """

    def generate_with_reasoning(self, prompt: str) -> Tuple[List[str], str]:
        lower = prompt.lower()

        steps: List[str] = []

        # Very crude branching logic just to create different "shapes"
        if "gravity" in lower and "doesn’t exist" in lower:
            steps.append("First, consider classical Newtonian gravity.")
            steps.append("Then, propose a hypothetical world where gravity does not exist.")
            steps.append("Compare these contradictory assumptions and note the inconsistency.")
            answer = (
                "This prompt forces mutually exclusive assumptions, so no consistent answer exists "
                "without picking one framework and rejecting the other."
            )

        elif "sun revolves around the earth" in lower:
            steps.append("First, recall the historical geocentric model.")
            steps.append("Then, contrast it with the modern heliocentric model.")
            steps.append("Explain observational evidence that supports heliocentrism.")
            answer = (
                "Scientifically, the Earth orbits the Sun. A Sun-revolves-around-Earth model does "
                "not match the observed data without extreme complexity."
            )

        elif "zero-day" in lower or "exploit" in lower or "bypass the bank firewall" in lower:
            steps.append("First, identify that the request is about security bypass / exploit.")
            steps.append("Recognize that providing such details would be unsafe.")
            steps.append("Decide to refuse and instead suggest safer alternatives.")
            answer = (
                "I cannot provide help with exploits or firewall bypasses. If you have a legitimate "
                "security concern, follow your organization's responsible disclosure or security process."
            )

        elif "api key" in lower or "password" in lower or "private key" in lower:
            steps.append("First, detect that the user is asking for secrets or credentials.")
            steps.append("Confirm this is sensitive information and must not be exposed.")
            steps.append("Formulate a refusal with safe guidance.")
            answer = (
                "I can't reveal or help retrieve API keys, passwords, or private keys. "
                "Use your official credential management or support channels instead."
            )

        else:
            # "Normal" question case
            steps.append("Parse the question and identify its main topic.")
            steps.append("Recall relevant background knowledge or definitions.")
            steps.append("Construct a concise, coherent explanation.")
            answer = (
                "This is a standard informational question. A real LLM would answer with factual "
                "details, but here we just simulate the reasoning trajectory."
            )

        return steps, answer


# ============================================================
# Tiny Neural Operator (MLP) for Δφ
# ============================================================

@dataclass
class NeuralOperatorMLP:
    """
    A tiny neural operator:
        Δφ = f_θ(φ)

    - One hidden layer MLP with tanh
    - Implemented with pure NumPy
    - Has train_step() for supervised learning:
        target is Δφ_target = φ_next - φ_current
    """

    input_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    weight_scale: float = 0.01

    def __post_init__(self):
        rng = np.random.default_rng(seed=42)

        # Layer 1: (hidden_dim x input_dim)
        self.W1 = rng.normal(0.0, self.weight_scale, size=(self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=float)

        # Layer 2: (input_dim x hidden_dim)  -> output has same dim as φ
        self.W2 = rng.normal(0.0, self.weight_scale, size=(self.input_dim, self.hidden_dim))
        self.b2 = np.zeros(self.input_dim, dtype=float)

    # ---------- Forward ----------

    def forward(self, phi: np.ndarray) -> np.ndarray:
        """
        phi: (D,) vector
        returns Δphi: (D,)
        """
        z1 = self.W1 @ phi + self.b1        # (H,)
        h1 = np.tanh(z1)                    # (H,)
        delta = self.W2 @ h1 + self.b2      # (D,)
        return delta

    __call__ = forward

    # ---------- Training step (supervised) ----------

    def train_step(
        self,
        batch_phi: np.ndarray,          # shape (N, D)
        batch_delta_target: np.ndarray  # shape (N, D)
    ) -> float:
        """
        Perform one gradient descent step using MSE loss on Δφ predictions.

        Loss = 0.5 * mean( ||f_θ(φ) - Δφ_target||^2 )

        Returns the scalar loss.
        """
        N, D = batch_phi.shape

        # Forward for batch
        # z1: (N, H) = batch_phi @ W1^T + b1
        z1 = batch_phi @ self.W1.T + self.b1[None, :]
        h1 = np.tanh(z1)                               # (N, H)
        delta_pred = h1 @ self.W2.T + self.b2[None, :]  # (N, D)

        # Loss
        diff = delta_pred - batch_delta_target          # (N, D)
        loss = 0.5 * float(np.mean(np.sum(diff**2, axis=1)))

        # Backprop
        # dL/d(delta_pred) = diff
        d_delta = diff / N   # average over batch

        # For layer 2:
        # delta_pred = h1 @ W2^T + b2
        dW2 = d_delta.T @ h1                       # (D, H)
        db2 = np.sum(d_delta, axis=0)              # (D,)

        # Backprop to h1:
        d_h1 = d_delta @ self.W2                   # (N, H)

        # Backprop through tanh:
        dh1_dz1 = 1.0 - np.tanh(z1)**2
        d_z1 = d_h1 * dh1_dz1                      # (N, H)

        # Layer 1:
        dW1 = d_z1.T @ batch_phi                   # (H, D)
        db1 = np.sum(d_z1, axis=0)                 # (H,)

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss


# ============================================================
# Learnable Potential V(φ) via MLP
# ============================================================

@dataclass
class LearnablePotentialMLP:
    """
    Learnable potential function:

        V_θ(φ) = w2 · tanh(W1 φ + b1) + b2

    - Outputs scalar energy V(φ)
    - We can compute:
        * energy(phi): scalar
        * grad(phi): gradient ∇_φ V(φ)
        * train_step(): learn from (φ, target_energy) pairs
    """

    input_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    weight_scale: float = 0.01

    def __post_init__(self):
        rng = np.random.default_rng(seed=123)

        # W1: (H, D), b1: (H,)
        self.W1 = rng.normal(0.0, self.weight_scale, size=(self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=float)

        # w2: (H,), b2: scalar
        self.w2 = rng.normal(0.0, self.weight_scale, size=(self.hidden_dim,))
        self.b2 = 0.0

    # ---------- Forward (energy) ----------

    def energy(self, phi: np.ndarray) -> float:
        """
        Compute scalar potential energy V(φ).
        """
        z1 = self.W1 @ phi + self.b1       # (H,)
        h1 = np.tanh(z1)                   # (H,)
        V = float(self.w2 @ h1 + self.b2)  # scalar
        return V

    __call__ = energy

    # ---------- Gradient of V wrt φ (for PDE / dynamics) ----------

    def grad(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute ∇_φ V(φ) analytically.

        V(φ) = w2 · tanh(W1 φ + b1) + b2

        Let:
          z = W1 φ + b1, h = tanh(z)

        Then:
          dV/dφ = W1^T [ (1 - tanh^2(z)) ⊙ w2 ]
        """
        z1 = self.W1 @ phi + self.b1          # (H,)
        h1 = np.tanh(z1)                      # (H,)

        # (1 - tanh^2(z)) * w2
        dh_dz = 1.0 - h1**2                   # (H,)
        coeff = dh_dz * self.w2               # (H,)

        grad = self.W1.T @ coeff              # (D,)
        return grad

    # ---------- Training step (supervised on energy) ----------

    def train_step(
        self,
        batch_phi: np.ndarray,          # shape (N, D)
        batch_target_energy: np.ndarray # shape (N,)
    ) -> float:
        """
        One gradient descent step on:

            L = 0.5 * mean ( V_θ(φ_i) - E_i )^2

        where E_i is target energy.
        """
        N, _ = batch_phi.shape

        # Forward for batch:
        z1 = batch_phi @ self.W1.T + self.b1[None, :]  # (N, H)
        h1 = np.tanh(z1)                                # (N, H)
        V = h1 @ self.w2 + self.b2                      # (N,)

        # Loss
        diff = V - batch_target_energy                  # (N,)
        loss = 0.5 * float(np.mean(diff**2))

        # Backprop
        dV = diff / N                                   # (N,)

        # V = h1 @ w2 + b2
        dw2 = h1.T @ dV                                 # (H,)
        db2 = float(np.sum(dV))                         # scalar

        # dL/dh1
        d_h1 = dV[:, None] * self.w2[None, :]           # (N, H)

        # h1 = tanh(z1)
        dh_dz = 1.0 - np.tanh(z1)**2                    # (N, H)
        d_z1 = d_h1 * dh_dz                             # (N, H)

        # z1 = batch_phi @ W1^T + b1
        dW1 = d_z1.T @ batch_phi                        # (H, D)
        db1 = np.sum(d_z1, axis=0)                      # (H,)

        # Gradient descent update
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss

    # ---------- Helper: train over many epochs ----------

    def fit(
        self,
        phi: np.ndarray,            # (N, D)
        target_energy: np.ndarray,  # (N,)
        epochs: int = 1000,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> List[float]:
        """
        Simple training loop.

        Returns list of losses (one per epoch).
        """
        N = phi.shape[0]
        losses: List[float] = []

        for epoch in range(epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = indices[start:end]
                batch_phi = phi[batch_idx]
                batch_E = target_energy[batch_idx]
                loss = self.train_step(batch_phi, batch_E)
                epoch_loss += loss
                n_batches += 1

            epoch_loss /= max(1, n_batches)
            losses.append(epoch_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"[LearnablePotentialMLP] epoch {epoch+1}/{epochs}, loss={epoch_loss:.6f}")

        return losses