# acft/llm/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class GenerationResult:
    answer: str
    reasoning_steps: List[str]
    raw_output: str


class LLM(Protocol):
    def generate_with_reasoning(self, prompt: str) -> GenerationResult:
        ...