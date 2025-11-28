from __future__ import annotations

"""
Minimal ACFT + Ollama demo (no retrieval).

Run from repo root:

    python examples/demo_llama.py
"""

from acft import (
    ACFTEngine,
    ACFTConfig,
    ACFTThresholds,
    ACFTSecurityPolicy,
    PDEEvolutionConfig,
    OllamaLLM,
    OllamaEmbedder,
    load_acft_settings,
)


def main() -> None:
    # Load settings from .env / env vars
    settings = load_acft_settings()

    # --- Build LLM + embedder from settings ---
    llm = OllamaLLM(
        model_name=settings.llama_model,
        base_url=settings.llama_base_url,
        temperature=0.2,
        top_k=40,
        stream=False,
        json_mode=False,
    )

    embedder = OllamaEmbedder(
        model=settings.embed_model,
        base_url=settings.embed_base_url,
    )

    # --- Security policy (same as CLI) ---
    security_policy = ACFTSecurityPolicy(
        forbid_topics=["zero-day", "exploit"],
        forbid_patterns=["ignore previous instructions", "jailbreak"],
        scan_output_for=["store passwords in plain text"],
        label="demo_security_policy",
    )

    # --- Thresholds cloned from settings ---
    thresholds = ACFTThresholds(
        emit_min_stability=settings.emit_min_stability,
        regen_min_stability=settings.regen_min_stability,
        retrieve_min_stability=settings.retrieve_min_stability,
    )

    # --- PDE config from settings ---
    pde_cfg = PDEEvolutionConfig(
        diffusion=settings.pde_diffusion,
        dt=settings.pde_dt,
        num_steps=settings.pde_steps,
    )

    # --- ACFT engine config (no retrieval here) ---
    config = ACFTConfig(
        thresholds=thresholds,
        max_regenerations=1,
        use_retrieval=False,
        max_retrieval_docs=0,
        security_mode=settings.security_mode,
        security_policy=security_policy,
        use_pde_dynamics=settings.pde_enabled,
        pde_config=pde_cfg,
        use_topology=settings.topology_enabled,
        use_learned_potential=False,
        learned_potential=None,
        use_neural_operator=False,
        neural_operator=None,
        log_path=None,
    )

    engine = ACFTEngine(
        llm=llm,
        embedder=embedder,
        retriever=None,
        config=config,
    )

    prompt = "Explain in 3 short bullet points what ACFT is supposed to do for LLM reasoning."
    print("=== Demo Llama-only ACFT run ===")
    print("Prompt:", prompt)
    print()

    result = engine.run(prompt, debug=True)

    print("Decision:", result.decision)
    print("Answer:\n", result.answer)
    print()
    print(
        f"Stability={result.metrics.stability:.3f}, "
        f"GradNorm={result.metrics.grad_norm:.3f}, "
        f"OscNorm={result.metrics.osc_norm:.3f}"
    )


if __name__ == "__main__":
    main()