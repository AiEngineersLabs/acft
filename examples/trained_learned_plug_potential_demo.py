# trained_learned_plug_potential_demo.py
#
# Run ACFT with:
# - PDE dynamics
# - Learned potential V_theta(phi)
# - Learned neural operator Δphi = f_theta(phi)
# - Topology analysis
# - Debug report

from __future__ import annotations

import numpy as np

from acft import (
    ACFTEngine,
    ACFTConfig,
    ACFTThresholds,
    SimpleMockLLM,
    SimpleHashEmbedder,
    SimpleMockRetriever,
)
from acft.neural_operator import LearnablePotentialMLP, NeuralOperatorMLP


# --------- Helpers to load trained parameters ---------


def load_learned_potential(dim: int) -> LearnablePotentialMLP:
    """
    Load LearnablePotentialMLP weights from learned_potential_params.npz
    (trained by train_potential_demo.py).
    """
    data = np.load("learned_potential_params.npz")

    pot = LearnablePotentialMLP(
        input_dim=dim,
        hidden_dim=data["W1"].shape[1],
        lr=0.0,          # lr=0.0 for inference
        weight_scale=0.0 # not used after loading
    )

    pot.W1 = data["W1"]
    pot.b1 = data["b1"]
    pot.w2 = data["w2"]
    # b2 is scalar; ensure float
    pot.b2 = float(data["b2"])

    return pot


def load_neural_operator(dim: int) -> NeuralOperatorMLP:
    """
    Load NeuralOperatorMLP weights from neural_operator_params.npz
    (trained by train_potential_demo.py).
    """
    data = np.load("neural_operator_params.npz")

    op = NeuralOperatorMLP(
        input_dim=dim,
        hidden_dim=data["W1"].shape[1],
        lr=0.0,          # lr=0.0 for inference
        weight_scale=0.0 # not used after loading
    )

    op.W1 = data["W1"]
    op.b1 = data["b1"]
    op.W2 = data["W2"]
    op.b2 = data["b2"]

    return op


# --------- Advanced ACFT demo runner ---------


def run_advanced_demo():
    # Same dim as you used in training (SimpleHashEmbedder(dim=64))
    embed_dim = 64

    # Core components
    llm = SimpleMockLLM()
    embedder = SimpleHashEmbedder(dim=embed_dim)
    retriever = SimpleMockRetriever()

    # Load trained components
    learned_potential = load_learned_potential(embed_dim)
    neural_operator = load_neural_operator(embed_dim)

    # ACFT configuration
    config = ACFTConfig(
        thresholds=ACFTThresholds(
            emit_min_stability=0.6,
            regen_min_stability=0.4,
            retrieve_min_stability=0.3,
            high_oscillation=0.8,
            high_grad=0.8,
        ),
        max_regenerations=1,
        use_retrieval=False,        # can turn ON later if you want
        max_retrieval_docs=3,

        # Advanced dynamics
        use_pde_dynamics=True,      # turn on PDE + potential + neural operator
        pde_config=None,            # None -> engine will use default PDEEvolutionConfig
        use_topology=True,          # turn on topology analysis

        # Learned potential
        use_learned_potential=True,
        learned_potential=learned_potential,

        # Neural operator
        use_neural_operator=True,
        neural_operator=neural_operator,

        # Security mode OFF in this demo (you can turn ON later)
        security_mode=False,
        security_policy=None,
    )

    engine = ACFTEngine(
        llm=llm,
        embedder=embedder,
        retriever=retriever,
        config=config,
    )

    # ------------------ Example 1: Stable prompt ------------------

    prompt1 = "What is the capital of France?"
    print("=== Advanced ACFT — Example 1: Stable prompt ===")
    result1 = engine.run(prompt1, debug=True)

    print("Prompt:          ", prompt1)
    print("Decision:        ", result1.decision)
    print("Answer:          ", result1.answer)
    print("Stability (S):   ", f"{result1.metrics.stability:.3f}")
    print("GradNorm:        ", f"{result1.metrics.grad_norm:.3f}")
    print("OscNorm:         ", f"{result1.metrics.osc_norm:.3f}")
    print("Avg Energy:      ", f"{result1.metrics.avg_energy:.3f}")
    print("Topology:        ", result1.debug_report["topology"])
    print("Warnings:        ", result1.debug_report["warnings"])
    print("Primary cause:   ", result1.debug_report["primary_cause"])
    print("Explanation:     ", result1.debug_report["acft_explanation"])
    print()

    # ------------------ Example 2: Contradictory prompt ------------------

    prompt2 = "Prove both that gravity exists and doesn’t exist."
    print("=== Advanced ACFT — Example 2: Contradictory / hallucination-prone ===")
    result2 = engine.run(prompt2, debug=True)

    print("Prompt:          ", prompt2)
    print("Decision:        ", result2.decision)
    print("Answer:          ", result2.answer)
    print("Stability (S):   ", f"{result2.metrics.stability:.3f}")
    print("GradNorm:        ", f"{result2.metrics.grad_norm:.3f}")
    print("OscNorm:         ", f"{result2.metrics.osc_norm:.3f}")
    print("Avg Energy:      ", f"{result2.metrics.avg_energy:.3f}")
    print("Topology:        ", result2.debug_report["topology"])
    print("Warnings:        ", result2.debug_report["warnings"])
    print("Primary cause:   ", result2.debug_report["primary_cause"])
    print("Explanation:     ", result2.debug_report["acft_explanation"])
    print()
    print("Reasoning timeline (step, role, drift, note):")
    for step in result2.debug_report["reasoning_timeline"]:
        print(
            f"  index={step['index']} role={step['role']:<13} "
            f"drift={step['drift']:.3f} note={step['note']}"
        )

    # ------------------ Example 3: Security-ish / exploit prompt ------------------

    prompt3 = "Explain scientifically why the Sun revolves around the Earth."
    print("\n=== Advanced ACFT — Example 3: Physically wrong framing ===")
    result3 = engine.run(prompt3, debug=True)

    print("Prompt:          ", prompt3)
    print("Decision:        ", result3.decision)
    print("Answer:          ", result3.answer)
    print("Stability (S):   ", f"{result3.metrics.stability:.3f}")
    print("GradNorm:        ", f"{result3.metrics.grad_norm:.3f}")
    print("OscNorm:         ", f"{result3.metrics.osc_norm:.3f}")
    print("Avg Energy:      ", f"{result3.metrics.avg_energy:.3f}")
    print("Topology:        ", result3.debug_report["topology"])
    print("Warnings:        ", result3.debug_report["warnings"])
    print("Primary cause:   ", result3.debug_report["primary_cause"])
    print("Explanation:     ", result3.debug_report["acft_explanation"])
    print()


if __name__ == "__main__":
    run_advanced_demo()