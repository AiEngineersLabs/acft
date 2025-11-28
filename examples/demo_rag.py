from __future__ import annotations

"""
ACFT + RAG demo using the built-in InMemoryRetriever and rag_corpus/.

Run from repo root:

    python examples/demo_rag.py
"""

import os
from acft import (
    ACFTEngine,
    ACFTConfig,
    ACFTThresholds,
    ACFTSecurityPolicy,
    PDEEvolutionConfig,
    OllamaLLM,
    OllamaEmbedder,
    InMemoryRetriever,
    load_acft_settings,
)
from acft.retrieval.loader import load_docs_from_folder  # demo helper


def build_retriever(embedder: OllamaEmbedder) -> InMemoryRetriever | None:
    """
    Demo RAG builder.

    Uses:
      - ACFT_RAG_FOLDER env var (e.g. 'rag_corpus')
    """
    rag_folder = os.getenv("ACFT_RAG_FOLDER", "rag_corpus")
    docs = load_docs_from_folder(rag_folder)
    if not docs:
        print(f"[demo_rag] No docs found in {rag_folder!r}, skipping retrieval.")
        return None

    print(f"[demo_rag] Loaded {len(docs)} docs from {rag_folder}")
    return InMemoryRetriever.from_texts(embedder=embedder, docs=docs)


def main() -> None:
    settings = load_acft_settings()

    # --- LLM + embedder ---
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

    retriever = build_retriever(embedder)

    # --- Security policy (same shape as CLI) ---
    security_policy = ACFTSecurityPolicy(
        forbid_topics=["zero-day", "exploit"],
        forbid_patterns=["ignore previous instructions", "jailbreak"],
        scan_output_for=["store passwords in plain text"],
        label="demo_rag_security_policy",
    )

    thresholds = ACFTThresholds(
        emit_min_stability=settings.emit_min_stability,
        regen_min_stability=settings.regen_min_stability,
        retrieve_min_stability=settings.retrieve_min_stability,
    )

    pde_cfg = PDEEvolutionConfig(
        diffusion=settings.pde_diffusion,
        dt=settings.pde_dt,
        num_steps=settings.pde_steps,
    )

    config = ACFTConfig(
        thresholds=thresholds,
        max_regenerations=1,
        use_retrieval=bool(retriever is not None),
        max_retrieval_docs=3,
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
        retriever=retriever,
        config=config,
    )

    prompt = (
        "Explain how many moons Earth has and whether the Sun orbits the Earth. "
        "If you are unsure, you MUST retrieve facts from your knowledge base "
        "before answering."
    )

    print("=== Demo RAG ACFT run ===")
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