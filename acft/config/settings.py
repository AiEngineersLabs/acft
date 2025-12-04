# acft/config/settings.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseSettings, Field


class ACFTSettings(BaseSettings):
    """
    Central ACFT configuration loaded from environment variables / .env.

    We explicitly bind each field to a concrete env var name so the mapping is
    predictable and stable, even if we rename attributes in code.

    NOTE:
      - This uses .env in the current working directory (project root).
      - python-dotenv must be installed for env_file support.
    """

    # -------- LLM backend (Ollama / vLLM / HF-compatible etc.) --------
    llm_backend: str = Field(
        "ollama",
        env="ACFT_LLM_BACKEND",
        description="LLM backend identifier (e.g. 'ollama', 'vllm', 'openai').",
    )

    # IMPORTANT: this is what `acft debug-settings` prints as model_name
    # and what the CLI uses if -m/--model is not provided.
    model_name: str = Field(
        "llama3.2:latest",
        env="ACFT_LLAMA_MODEL",  # <-- bind to your .env var
        description="Default LLM model name (e.g. 'llama3:latest').",
    )

    base_url: str = Field(
        "http://localhost:11434",
        env="ACFT_LLAMA_BASE_URL",
        description="Base URL for the LLM HTTP endpoint.",
    )

    # -------- Embeddings backend --------
    embed_backend: str = Field(
        "ollama",
        env="ACFT_EMBED_BACKEND",
        description="Embedding backend identifier (e.g. 'ollama', 'hf').",
    )

    embed_model: str = Field(
        "nomic-embed-text",
        env="ACFT_EMBED_MODEL",
        description="Embedding model name.",
    )

    embed_base_url: str = Field(
        "http://localhost:11434",
        env="ACFT_EMBED_BASE_URL",
        description="Base URL for embedding endpoint.",
    )

    # -------- Stability thresholds --------
    emit_min_stability: float = Field(
        0.6,
        env="ACFT_EMIT_MIN_STABILITY",
        description="Minimum stability required to EMIT an answer without regen.",
    )

    regen_min_stability: float = Field(
        0.4,
        env="ACFT_REGEN_MIN_STABILITY",
        description="Minimum stability below which we trigger regeneration.",
    )

    retrieve_min_stability: float = Field(
        0.3,
        env="ACFT_RETRIEVE_MIN_STABILITY",
        description="Minimum stability below which we trigger RAG retrieval.",
    )

    # -------- Retrieval & security --------
    use_retrieval: bool = Field(
        False,
        env="ACFT_USE_RETRIEVAL",
        description="Enable ACFT retrieval / RAG pipeline.",
    )

    rag_folder: str = Field(
        "rag_corpus",
        env="ACFT_RAG_FOLDER",
        description="Folder name for local RAG corpus.",
    )

    security_mode: bool = Field(
        False,
        env="ACFT_SECURITY_MODE",
        description="Turn ACFT security analysis on/off.",
    )

    security_policy_file_enable: bool = Field(
        False,
        env="ACFT_SECURITY_POLICY_FILE_ENABLE",
        description="If true, attempt to load security policy from JSON file.",
    )

    security_policy_filename: str = Field(
        "security_policy.json",
        env="ACFT_SECURITY_POLICY_FILENAME",
        description="Default security policy filename in project root.",
    )

    # Optional absolute/override path, if user wants:
    security_policy_file: Optional[str] = Field(
        None,
        env="ACFT_SECURITY_POLICY_FILE",
        description="Optional explicit path to security policy JSON.",
    )

    # -------- PDE & topology (physics layer) --------
    pde_enabled: bool = Field(
        False,
        env="ACFT_PDE_ENABLED",
        description="Enable PDE-based cognitive field evolution.",
    )

    pde_diffusion: float = Field(
        0.1,
        env="ACFT_PDE_DIFFUSION",
        description="Diffusion coefficient D in ∂φ/∂t = D∇²φ - ∇V(φ).",
    )

    pde_dt: float = Field(
        0.05,
        env="ACFT_PDE_DT",
        description="Time-step dt for PDE evolution.",
    )

    pde_steps: int = Field(
        5,
        env="ACFT_PDE_STEPS",
        description="Number of PDE steps per reasoning turn.",
    )

    topology_enabled: bool = Field(
        False,
        env="ACFT_TOPOLOGY_ENABLED",
        description="Enable topological diagnostics (components, loops, χ).",
    )

    class Config:
        # We KEEP env_file so `.env` in the current working dir is used.
        env_file = ".env"
        env_file_encoding = "utf-8"
        # We do NOT rely on env_prefix here because we specify env per-field.
        # env_prefix = "ACFT_"


def load_acft_settings() -> ACFTSettings:
    """
    Factory/helper so callers can re-load settings (e.g. in CLI debug).
    """
    return ACFTSettings()


# Singleton-style settings instance used throughout the package.
settings: ACFTSettings = load_acft_settings()