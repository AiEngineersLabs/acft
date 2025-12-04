from __future__ import annotations

"""
examples/load_learned_potential_demo.py

Load the physics heads trained by examples/train_potential_demo.py
and plug them into an ACFTEngine instance.

This demo:
  1. Loads:
       - learned_potential_params.npz
       - neural_operator_params.npz
  2. Reconstructs LearnablePotentialMLP and NeuralOperatorMLP
  3. Builds an ACFTEngine using SimpleMockLLM + SimpleHashEmbedder
  4. Runs a sample prompt and prints decision, answer, and debug report

Run from the project root:

    source .venv/bin/activate
    python examples/load_learned_potential_demo.py

Make sure you have already run:

    python examples/train_potential_demo.py

so that the NPZ parameter files exist in the current working directory.
"""

import json
import numpy as np

from acft.core.engine import (
    ACFTEngine,
    ACFTConfig,
    ACFTThresholds,
    PDEEvolutionConfig,
)

from acft.neural_operator import (
    SimpleHashEmbedder,
    SimpleMockLLM,
    LearnablePotentialMLP,
    NeuralOperatorMLP,
)


# ------------------------------------------------------------
# 1. Loader helpers
# ------------------------------------------------------------

def load_learnable_potential(
    params_path: str = "learned_potential_params.npz",
    input_dim: int = 64,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_scale: float = 0.01,
) -> LearnablePotentialMLP:
    """
    Recreate LearnablePotentialMLP and load weights from NPZ.

    The NPZ file must contain:
      - W1 (H, D)
      - b1 (H,)
      - w2 (H,)
      - b2 (scalar)
    """
    data = np.load(params_path)

    potential = LearnablePotentialMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_scale=weight_scale,
    )

    potential.W1 = data["W1"]
    potential.b1 = data["b1"]
    potential.w2 = data["w2"]
    potential.b2 = float(data["b2"])

    print(f"[load_learned_potential_demo] Loaded LearnablePotentialMLP from {params_path}")
    return potential


def load_neural_operator(
    params_path: str = "neural_operator_params.npz",
    input_dim: int = 64,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_scale: float = 0.01,
) -> NeuralOperatorMLP:
    """
    Recreate NeuralOperatorMLP and load weights from NPZ.

    The NPZ file must contain:
      - W1 (H, D)
      - b1 (H,)
      - W2 (D, H)
      - b2 (D,)
    """
    data = np.load(params_path)

    op = NeuralOperatorMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_scale=weight_scale,
    )

    op.W1 = data["W1"]
    op.b1 = data["b1"]
    op.W2 = data["W2"]
    op.b2 = data["b2"]

    print(f"[load_learned_potential_demo] Loaded NeuralOperatorMLP from {params_path}")
    return op


# ------------------------------------------------------------
# 2. Build engine with learned physics
# ------------------------------------------------------------

def build_engine_with_learned_physics() -> ACFTEngine:
    """
    Construct an ACFTEngine that uses:

      - SimpleMockLLM (toy LLM for trajectories)
      - SimpleHashEmbedder (toy embedding)
      - LearnablePotentialMLP (V(φ))
      - NeuralOperatorMLP (Δφ = f(φ))

    This engine will still produce reasoning + debug reports,
    but now the stability / dynamics can be driven by the learned
    potential and learned operator instead of purely hand-coded
    metrics.
    """
    dim = 64
    hidden_dim = 64

    # 1) Load learned physics modules from NPZ files
    potential = load_learnable_potential(
        params_path="learned_potential_params.npz",
        input_dim=dim,
        hidden_dim=hidden_dim,
        lr=1e-3,
        weight_scale=0.01,
    )

    neural_op = load_neural_operator(
        params_path="neural_operator_params.npz",
        input_dim=dim,
        hidden_dim=hidden_dim,
        lr=1e-3,
        weight_scale=0.01,
    )

    # 2) Build ACFTConfig with these components enabled
    acft_config = ACFTConfig(
        thresholds=ACFTThresholds(
            emit_min_stability=0.5,
            regen_min_stability=0.3,
            retrieve_min_stability=0.1,
        ),
        max_regenerations=1,
        use_retrieval=False,
        max_retrieval_docs=0,
        security_mode=False,
        security_policy=None,
        use_pde_dynamics=True,
        pde_config=PDEEvolutionConfig(
            diffusion=0.1,
            dt=0.05,
            num_steps=5,
        ),
        use_topology=True,
        use_learned_potential=True,
        learned_potential=potential,
        use_neural_operator=True,
        neural_operator=neural_op,
    )

    # 3) Use toy LLM + toy embedder for this demo
    llm = SimpleMockLLM()
    embedder = SimpleHashEmbedder(dim=dim)

    engine = ACFTEngine(
        llm=llm,
        embedder=embedder,
        retriever=None,
        config=acft_config,
    )
    return engine


# ------------------------------------------------------------
# 3. Demo main
# ------------------------------------------------------------

def main() -> None:
    engine = build_engine_with_learned_physics()

    print("✅ Engine with learned potential + neural operator is ready.\n")

    prompt = (
        "Prove both that gravity exists and does not exist, and comment on whether "
        "this reasoning is stable."
    )
    print(f"User prompt:\n  {prompt}\n")

    result = engine.run(prompt, debug=True)

    print(f"ACFT decision : {result.decision}")
    print(f"ACFT answer   : {result.answer}\n")

    if result.debug_report:
        print("ACFT debug report:")
        print(json.dumps(result.debug_report, indent=2))


if __name__ == "__main__":
    main()