# acft/llm/__init__.py
from __future__ import annotations

from .base import LLM, GenerationResult
from .ollama import OllamaLLM

__all__ = ["LLM", "GenerationResult", "OllamaLLM"]