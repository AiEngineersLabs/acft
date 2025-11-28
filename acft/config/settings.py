# acft/config/settings.py
from __future__ import annotations

from functools import lru_cache
from pydantic import BaseSettings, Field


class ACFTSettings(BaseSettings):
    """
    Central ACFT configuration, loaded from environment variables
    (and optionally .env in the project root).

    Env vars are prefixed with `ACFT_`, for example:

      ACFT_LLM_BACKEND=ollama
      ACFT_LLAMA_MODEL=llama3.2:latest
      ACFT_SECURITY_MODE=true
      ACFT_SECURITY_POLICY_FILE_ENABLE=true
    """

    # ----- LLM backend -----
    llm_backend: str = Field("ollama", description="Backend type for LLM (ollama, vllm, etc.)")
    model_name: str = Field("llama3.2:latest", description="LLM model name")
    base_url: str = Field("http://localhost:11434", description="LLM base URL")

    # ----- Embeddings backend -----
    embed_backend: str = Field("ollama", description="Backend type for embeddings")
    embed_model: str = Field("nomic-embed-text", description="Embedding model name")
    embed_base_url: str = Field("http://localhost:11434", description="Embedding base URL")

    # ----- Stability thresholds -----
    emit_min_stability: float = Field(0.6, description="Threshold to EMIT answer")
    regen_min_stability: float = Field(0.4, description="Threshold to REGENERATE")
    retrieve_min_stability: float = Field(0.3, description="Threshold to RETRIEVE via RAG")

    # ----- Retrieval & RAG -----
    use_retrieval: bool = Field(False, description="Enable retrieval-augmented generation")
    rag_folder: str = Field("rag_corpus", description="Folder for RAG documents")

    # ----- Security -----
    security_mode: bool = Field(False, description="Enable ACFT security mode")
    security_policy_file_enable: bool = Field(
        False,
        description="If true, load security policy from JSON file",
    )
    security_policy_filename: str = Field(
        "security_policy.json",
        description="Default security policy filename in project root",
    )

    # Optional explicit path override:
    # e.g. ACFT_SECURITY_POLICY_FILE=/custom/path/policy.json
    security_policy_file: str | None = Field(
        None,
        description="Optional explicit filesystem path to security policy JSON",
    )

    # ----- PDE & topology -----
    pde_enabled: bool = Field(False, description="Enable PDE-based field evolution")
    pde_diffusion: float = Field(0.1, description="Diffusion coefficient for PDE")
    pde_dt: float = Field(0.05, description="Time step for PDE evolution")
    pde_steps: int = Field(5, description="Number of PDE evolution steps")
    topology_enabled: bool = Field(False, description="Enable topology analysis")

    class Config:
        # This is the key bit: make env vars like ACFT_SECURITY_MODE work.
        env_prefix = "ACFT_"
        case_sensitive = False
        # Automatically load .env from project root if present
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def load_acft_settings() -> ACFTSettings:
    """
    Load ACFTSettings once (cached). This will read environment variables
    and .env (if present) using Pydantic's BaseSettings.
    """
    return ACFTSettings()


# Singleton-style settings instance used across the package
settings: ACFTSettings = load_acft_settings()