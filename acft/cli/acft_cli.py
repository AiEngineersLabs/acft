# acft/cli/acft_cli.py

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from acft.config.settings import (
    settings,            # ACFTSettings instance
    load_acft_settings,  # helper to re-read from env if needed
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

# SimpleHashEmbedder used ONLY for learned-physics mode
from acft.neural_operator import SimpleHashEmbedder

# dynamic / JSON-based policy loader
from acft.security.policy import load_security_policy_from_settings

# learned physics integration
from acft.physics.learned_integration import build_config_with_learned_physics


# -------------------------------------------------
# Helper: build an ACFTEngine from settings
# -------------------------------------------------

def build_engine(
    model_name: Optional[str] = None,
    use_learned_physics: bool = False,
) -> ACFTEngine:
    """
    Construct an ACFTEngine wired to Ollama + ACFTConfig.

    Default:
      - Uses OllamaLLM + OllamaEmbedder (nomic-embed-text)
      - Uses PDE/topology according to ACFT settings
      - No learned potential / neural operator

    If use_learned_physics=True:
      - Wraps the base ACFTConfig with learned potential + neural operator
        loaded from NPZ files.
      - Uses SimpleHashEmbedder(dim=64) so φ dimension matches the learned
        modules (64), avoiding 768 vs 64 shape mismatches.
    """
    cfg = settings  # ACFTSettings instance

    # ---- LLM adapter ----
    llm = OllamaLLM(
        model_name=model_name or cfg.model_name,
        base_url=cfg.base_url,
        temperature=0.2,
        top_k=40,
        stream=False,
        json_mode=False,
    )

    # ---- Embeddings adapter ----
    if use_learned_physics:
        # IMPORTANT:
        # Your learned potential & neural operator were trained on 64-dim
        # vectors, so we must use the same dimensionality here.
        embedder = SimpleHashEmbedder(dim=64)
    else:
        # Default behavior
        embedder = OllamaEmbedder(
            model=cfg.embed_model,
            base_url=cfg.embed_base_url,
        )

    # ---- Security policy resolution ----
    # 1) Try to load from JSON / env / settings via helper
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

    # ---- Base ACFT config (this matches your current behavior) ----
    base_config = ACFTConfig(
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
        # learned physics (off by default, same as your previous config)
        use_learned_potential=False,
        learned_potential=None,
        use_neural_operator=False,
        neural_operator=None,
    )

    # ---- Optional: wrap with learned physics heads ----
    if use_learned_physics:
        # NOTE: paths + dims must match your training demo
        config = build_config_with_learned_physics(
            base_config,
            potential_path="learned_potential_params.npz",
            operator_path="neural_operator_params.npz",
            input_dim=64,   # must match SimpleHashEmbedder dim in examples
            hidden_dim=64,  # must match training
        )
    else:
        config = base_config

    # ---- Build engine ----
    engine = ACFTEngine(
        llm=llm,
        embedder=embedder,
        retriever=None,  # plug in RAG retriever later
        config=config,
    )
    return engine


# -------------------------------------------------
# Chat loop
# -------------------------------------------------

def run_chat(
    model_name: Optional[str] = None,
    use_learned_physics: bool = False,
) -> None:
    """
    Interactive REPL.

    If use_learned_physics=True, the engine uses the learned potential
    and neural operator loaded from NPZ files (if present).
    """
    engine = build_engine(
        model_name=model_name,
        use_learned_physics=use_learned_physics,
    )

    active_model = model_name or settings.model_name
    physics_suffix = " + learned_physics" if use_learned_physics else ""

    print(f"🚀 ACFT chat started (model: {active_model}{physics_suffix})")
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
            print("DebugReport:", json.dumps(result.debug_report, indent=2))
        print()


# -------------------------------------------------
# CLI entrypoint
# -------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="acft",
        description="ACFT CLI – physics-based guard + chat on top of your local LLM",
        epilog=(
            "Examples:\n"
            "  acft debug-settings\n"
            "  acft chat\n"
            "  acft chat --model llama3.2:latest\n"
            "  acft chat --use-learned-physics\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="LLM model name (default from ACFT_LLAMA_MODEL / ACFT settings).",
    )
    p_chat.add_argument(
        "--use-learned-physics",
        action="store_true",
        help=(
            "Enable learned potential + neural operator from NPZ files "
            "(learned_potential_params.npz, neural_operator_params.npz)."
        ),
    )

    # acft help (pseudo-subcommand)
    subparsers.add_parser(
        "help",
        help="Show this help message and exit.",
    )

    args = parser.parse_args()

    # acft help
    if args.command == "help":
        # Show main help
        parser.print_help()
        # Also show detailed chat help including --use-learned-physics
        print("\n\nDetailed `acft chat` usage:\n")
        p_chat.print_help()
        return 0

    # acft debug-settings
    if args.command == "debug-settings":
        cfg = load_acft_settings()
        print("=== ACFT Debug Settings ===")
        print(cfg.json(indent=2))
        return 0

    # acft chat [--model ...] [--use-learned-physics]
    if args.command == "chat":
        run_chat(
            model_name=args.model,
            use_learned_physics=getattr(args, "use_learned_physics", False),
        )
        return 0

    # No subcommand -> default to help
    parser.print_help()
    print("\n\nDetailed `acft chat` usage:\n")
    p_chat.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())