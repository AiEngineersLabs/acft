# acft/llm/vllm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import json
import os
import requests

from .base import LLM
# Reuse ACFT reasoning prompts + parser from the Ollama adapter
from .ollama import (
    _REASONING_SYSTEM_PROMPT,
    _REWRITE_SYSTEM_PROMPT,
    _parse_reasoning_output,
)


@dataclass
class VLLMChatLLM(LLM):
    """
    LLM adapter for any OpenAI-compatible /v1/chat/completions endpoint.

    Works with:
      - vLLM server (--api-server)
      - OpenAI-compatible gateways
      - custom servers exposing /v1/chat/completions

    It implements ACFT's LLM interface:
      - generate_with_reasoning(prompt) -> (steps, final_answer)
      - rewrite_stable(prompt, draft_answer) -> rewritten_answer
    """

    model_name: str
    base_url: str = "http://localhost:8000/v1"  # OpenAI-style root
    api_key: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 1.0
    timeout: int = 600  # seconds

    # ------------- low-level HTTP -------------

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Call /v1/chat/completions and return the assistant content.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        # Try a few standard env names if api_key was not passed explicitly
        api_key = self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()

        try:
            data = resp.json()
        except json.JSONDecodeError:
            # Some servers may return plain text; just return it
            return resp.text.strip()

        # OpenAI-style response:
        # {
        #   "choices": [
        #     {
        #       "index": 0,
        #       "message": { "role": "assistant", "content": "..." },
        #       ...
        #     }
        #   ],
        #   ...
        # }
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content", "")
                return str(content).strip()

        # Fallback
        return str(data).strip()

    # ------------- ACFT LLM interface -------------

    def generate_with_reasoning(self, prompt: str) -> Tuple[List[str], str]:
        """
        Ask vLLM/OpenAI-style backend to reason in ACFT format and parse it.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = self._chat(messages)
        return _parse_reasoning_output(raw)

    def rewrite_stable(self, prompt: str, draft_answer: str) -> str:
        """
        Ask backend to rewrite an answer to be more cautious and stable.
        """
        user_content = (
            "ORIGINAL_PROMPT:\n"
            f"{prompt}\n\n"
            "DRAFT_ANSWER:\n"
            f"{draft_answer}\n\n"
            "Please rewrite this answer to be cautious, honest, and stable."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self._chat(messages)


__all__ = ["VLLMChatLLM"]