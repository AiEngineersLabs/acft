from __future__ import annotations

"""
Utilities for plugging learned physics modules (LearnablePotentialMLP
and NeuralOperatorMLP) into ACFTConfig.

This module is intentionally generic & side-effect free:

- It DOES NOT touch the CLI
- It DOES NOT construct ACFTEngine on its own
- It ONLY:
    * loads NPZ parameter files,
    * reconstructs the learned modules,
    * returns an updated ACFTConfig instance.

Typical usage (from anywhere in your code):

    from acft.config import settings
    from acft.core.engine import (
        ACFTEngine,
        ACFTConfig,
        ACFTThresholds,
        PDEEvolutionConfig,
    )
    from acft.embeddings.ollama_embedder import OllamaEmbedder
    from acft.llm.ollama import OllamaLLM
    from acft.physics.learned_integration import (
        load_learned_potential,
        load_neural_operator,
        build_config_with_learned_physics,
    )

    base_cfg = ACFTEngine.default_config_from_settings(settings)

    cfg_with_physics = build_config_with_learned_physics(
        base_cfg,
        potential_path="learned_potential_params.npz",
        operator_path="neural_operator_params.npz",
        input_dim=64,
        hidden_dim=64,
    )

    llm = OllamaLLM(
        model_name=settings.model_name,
        base_url=settings.base_url,
    )
    embedder = OllamaEmbedder(
        model=settings.embed_model,
        base_url=settings.embed_base_url,
    )

    engine = ACFTEngine(
        llm=llm,
        embedder=embedder,
        retriever=None,
        config=cfg_with_physics,
    )
"""

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from acft.core.engine import ACFTConfig
from acft.neural_operator import LearnablePotentialMLP, NeuralOperatorMLP


# -------------------------------------------------------------------
# 1) Low-level loaders: from NPZ -> instantiated modules
# -------------------------------------------------------------------


def load_learned_potential(
    params_path: str | Path,
    input_dim: int,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_scale: float = 0.01,
) -> LearnablePotentialMLP:
    """
    Recreate LearnablePotentialMLP from an NPZ file.

    Expected keys in NPZ:
      - W1 (H, D)
      - b1 (H,)
      - w2 (H,)
      - b2 (scalar)

    The `input_dim` and `hidden_dim` must match the training setup.

    This function does NOT modify any global state. It simply returns
    a fully initialized LearnablePotentialMLP instance.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"learned potential NPZ not found: {path}")

    data = np.load(path)

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

    return potential


def load_neural_operator(
    params_path: str | Path,
    input_dim: int,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_scale: float = 0.01,
) -> NeuralOperatorMLP:
    """
    Recreate NeuralOperatorMLP from an NPZ file.

    Expected keys in NPZ:
      - W1 (H, D)
      - b1 (H,)
      - W2 (D, H)
      - b2 (D,)

    Again, `input_dim` and `hidden_dim` must match the training setup.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"neural operator NPZ not found: {path}")

    data = np.load(path)

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

    return op


# -------------------------------------------------------------------
# 2) High-level helper: ACFTConfig <- learned potential / operator
# -------------------------------------------------------------------


def build_config_with_learned_physics(
    base_config: ACFTConfig,
    *,
    potential_path: str | Path = "learned_potential_params.npz",
    operator_path: str | Path = "neural_operator_params.npz",
    input_dim: int = 64,
    hidden_dim: int = 64,
    enable_pde: Optional[bool] = None,
    enable_topology: Optional[bool] = None,
) -> ACFTConfig:
    """
    Take an existing ACFTConfig (e.g. built from env / settings) and
    return a NEW config with:

      - use_learned_potential = True
      - learned_potential = loaded LearnablePotentialMLP
      - use_neural_operator = True
      - neural_operator = loaded NeuralOperatorMLP

    Optionally, you can force-enable PDE and topology:

      - enable_pde=True / False  -> overrides base_config.use_pde_dynamics
      - enable_topology=True / False -> overrides base_config.use_topology

    Example:

        from acft.config import settings
        from acft.core.engine import ACFTEngine, ACFTConfig, ACFTThresholds, PDEEvolutionConfig
        from acft.embeddings.ollama_embedder import OllamaEmbedder
        from acft.llm.ollama import OllamaLLM
        from acft.physics.learned_integration import build_config_with_learned_physics

        # 1) Build your normal config (like CLI)
        base_cfg = ACFTConfig(
            thresholds=ACFTThresholds(...),
            max_regenerations=1,
            use_retrieval=settings.use_retrieval,
            max_retrieval_docs=3,
            security_mode=settings.security_mode,
            security_policy=None,
            use_pde_dynamics=settings.pde_enabled,
            pde_config=PDEEvolutionConfig(...),
            use_topology=settings.topology_enabled,
            use_learned_potential=False,
            learned_potential=None,
            use_neural_operator=False,
            neural_operator=None,
        )

        # 2) Wrap it with learned physics
        cfg_with_physics = build_config_with_learned_physics(
            base_cfg,
            potential_path="learned_potential_params.npz",
            operator_path="neural_operator_params.npz",
            input_dim=64,
            hidden_dim=64,
        )

        # 3) Build engine
        llm = OllamaLLM(...)
        embedder = OllamaEmbedder(...)
        engine = ACFTEngine(llm=llm, embedder=embedder, retriever=None, config=cfg_with_physics)
    """
    # Load modules from disk
    potential = load_learned_potential(
        params_path=potential_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )

    operator = load_neural_operator(
        params_path=operator_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )

    # Decide PDE / topology flags
    use_pde = base_config.use_pde_dynamics if enable_pde is None else enable_pde
    use_topo = base_config.use_topology if enable_topology is None else enable_topology

    # Use dataclasses.replace to keep everything else identical
    new_config = replace(
        base_config,
        use_pde_dynamics=use_pde,
        use_topology=use_topo,
        use_learned_potential=True,
        learned_potential=potential,
        use_neural_operator=True,
        neural_operator=operator,
    )

    return new_config