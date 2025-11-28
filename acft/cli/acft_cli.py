# acft/cli/acft_cli.py

from __future__ import annotations

import argparse
import json
import sys

from acft.config.settings import (
    settings,           # ACFTSettings instance
    load_acft_settings, # helper to re-read from env if needed
)

from acft.core.engine import (
    ACFTEngine,
    ACFTConfig,
    ACFTThresholds,
    PDEEvolutionConfig,
    ACFTSecurityPolicy,   # used only for fallback when no JSON policy is loaded
)

from acft.embeddings.ollama_embedder import OllamaEmbedder
from acft.llm.ollama import OllamaLLM

# dynamic / JSON-based policy loader
from acft.security.policy import load_security_policy_from_settings


# -------------------------------------------------
# Helper: build an ACFTEngine from settings
# -------------------------------------------------

def build_engine(model_name: str | None = None) -> ACFTEngine:
    cfg = settings  # ACFTSettings instance

    # ---- LLM adapter ----
    # IMPORTANT: OllamaLLM expects `model_name`, not `model`
    llm = OllamaLLM(
        model_name=model_name or cfg.model_name,
        base_url=cfg.base_url,
        temperature=0.2,
        top_k=40,
        stream=False,
        json_mode=False,
    )

    # ---- Embeddings adapter ----
    embedder = OllamaEmbedder(
        model=cfg.embed_model,
        base_url=cfg.embed_base_url,
    )

    # ---- Security policy resolution ----
    # 1) Try to load from JSON / env / settings via helper
    #    - respects:
    #        ACFT_SECURITY_MODE
    #        ACFT_SECURITY_POLICY_FILE_ENABLE
    #        ACFT_SECURITY_POLICY_FILE (env override)
    #        settings.security_policy_filename
    dynamic_policy = load_security_policy_from_settings(cfg)

    # 2) Fallback: built-in default policy if security_mode is ON but no file found
    if dynamic_policy is not None:
        security_policy = dynamic_policy
    elif cfg.security_mode:
        security_policy = ACFTSecurityPolicy(
            forbid_topics=["zero-day", "exploit"],
            forbid_patterns=["ignore previous instructions", "jailbreak"],
            scan_output_for=["store passwords in plain text"],
            label="acft_cli_fallback",
        )
    else:
        # security mode disabled -> no policy -> risk_level will be "UNKNOWN"
        security_policy = None

    # ---- ACFT config ----
    acft_config = ACFTConfig(
        thresholds=ACFTThresholds(
            emit_min_stability=cfg.emit_min_stability,
            regen_min_stability=cfg.regen_min_stability,
            retrieve_min_stability=cfg.retrieve_min_stability,
        ),
        max_regenerations=1,
        use_retrieval=cfg.use_retrieval,
        max_retrieval_docs=3,
        security_mode=cfg.security_mode,
        security_policy=security_policy,
        use_pde_dynamics=cfg.pde_enabled,
        pde_config=PDEEvolutionConfig(
            diffusion=cfg.pde_diffusion,
            dt=cfg.pde_dt,
            num_steps=cfg.pde_steps,
        ),
        use_topology=cfg.topology_enabled,
        # advanced stuff (learned potential / neural operator)
        use_learned_potential=False,
        learned_potential=None,
        use_neural_operator=False,
        neural_operator=None,
    )

    engine = ACFTEngine(
        llm=llm,
        embedder=embedder,
        retriever=None,  # plug in RAG retriever later
        config=acft_config,
    )
    return engine


# -------------------------------------------------
# Chat loop
# -------------------------------------------------

def run_chat(model_name: str | None = None) -> None:
    engine = build_engine(model_name=model_name)
    active_model = model_name or settings.model_name

    print(f"🚀 ACFT chat started (model: {active_model})")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if not user:
            continue

        result = engine.run(user, debug=True)

        print(f"ACFT ({result.decision}): {result.answer}")
        if result.debug_report:
            # Your requested style: DebugReport: { ... }
            print("DebugReport:", json.dumps(result.debug_report, indent=2))
        print()


# -------------------------------------------------
# CLI entrypoint
# -------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="acft",
        description="ACFT CLI – run ACFT guard + chat on top of your local LLM",
    )
    subparsers = parser.add_subparsers(dest="command")

    # acft debug-settings
    subparsers.add_parser(
        "debug-settings",
        help="Print resolved ACFT settings from environment variables.",
    )

    # acft chat
    p_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive ACFT chat session.",
    )
    p_chat.add_argument(
        "-m",
        "--model",
        default=settings.model_name,
        help="LLM model name (default from ACFT_LLAMA_MODEL).",
    )

    args = parser.parse_args()

    if args.command == "debug-settings":
        cfg = load_acft_settings()
        print("=== ACFT Debug Settings ===")
        print(cfg.json(indent=2))
        return 0

    if args.command == "chat":
        run_chat(model_name=args.model)
        return 0

    # No subcommand -> show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())