# acft/__init__.py
from __future__ import annotations

"""
Top-level ACFT package exports.

This lets users write, for example:

    from acft import (
        ACFTEngine,
        ACFTConfig,
        ACFTThresholds,
        PDEEvolutionConfig,
        ACFTSecurityPolicy,
        SecurityAnalysisResult,
        analyze_security,
        ACFTSettings,
        settings,
        load_acft_settings,
        OllamaLLM,
        OllamaEmbedder,
    )
"""

# ---- Settings / environment ----
from .config.settings import ACFTSettings, load_acft_settings, settings

# ---- Core engine & config dataclasses ----
# NOTE: We import ACFTConfig / ACFTThresholds / PDEEvolutionConfig from core.engine
# instead of acft.config.config_model to avoid the ImportError you hit.
from .core.engine import (
    ACFTEngine,
    ACFTConfig,
    ACFTThresholds,
    PDEEvolutionConfig,
    ACFTSecurityPolicy,
)

# ---- LLM / Embeddings adapters ----
from .llm.ollama import OllamaLLM
from .embeddings.ollama_embedder import OllamaEmbedder

# ---- Security primitives ----
from .security.policy import SecurityAnalysisResult
from .security.analyzer import analyze_security

__all__ = [
    # settings
    "ACFTSettings",
    "load_acft_settings",
    "settings",
    # engine + config
    "ACFTEngine",
    "ACFTConfig",
    "ACFTThresholds",
    "PDEEvolutionConfig",
    "ACFTSecurityPolicy",
    # adapters
    "OllamaLLM",
    "OllamaEmbedder",
    # security
    "SecurityAnalysisResult",
    "analyze_security",
]