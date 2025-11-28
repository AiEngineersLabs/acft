from __future__ import annotations

import json
from typing import List, Tuple, Dict, Any

import requests


class OllamaLLM:
    """
    ACFT-compatible adapter for Ollama running locally.

    Exposes:
        generate_with_reasoning(prompt: str) -> (reasoning_steps, final_answer)

    This matches the LLM Protocol expected by ACFTEngine.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        top_k: int = 40,
        stream: bool = False,
        json_mode: bool = False,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.stream = bool(stream)
        self.json_mode = bool(json_mode)

    # --------------------------------------------------
    # Core call to Ollama
    # --------------------------------------------------

    def _post_generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        if not self.stream:
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json()

        # Streaming mode: accumulate partial responses
        resp = requests.post(url, json=payload, stream=True, timeout=600)
        resp.raise_for_status()

        chunks: List[str] = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if "response" in data:
                chunks.append(data["response"])

        return {"response": "".join(chunks)}

    # --------------------------------------------------
    # Public API: generate_with_reasoning
    # --------------------------------------------------

    def generate_with_reasoning(self, prompt: str) -> Tuple[List[str], str]:
        """
        Wraps the user prompt with an instruction that encourages
        step-by-step reasoning and a clear final answer.

        Returns:
            reasoning_steps: list of strings
            final_answer:    string
        """
        wrapped_prompt = (
            "You are an AI assistant using explicit reasoning.\n"
            "First think step-by-step in numbered steps.\n"
            "Then end with a line starting with 'FINAL ANSWER:'.\n\n"
            f"Question: {prompt}\n"
        )

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": wrapped_prompt,
            "stream": self.stream,
            "options": {
                "temperature": self.temperature,
                "top_k": self.top_k,
            },
        }

        # If you want JSON mode later, you can extend payload here.
        data = self._post_generate(payload)
        raw_output = data.get("response", "").strip()

        reasoning_steps, final_answer = self._extract_reasoning(raw_output)
        return reasoning_steps, final_answer

    # --------------------------------------------------
    # Reasoning extractor
    # --------------------------------------------------

    def _extract_reasoning(self, text: str) -> Tuple[List[str], str]:
        """
        Split model output into reasoning steps and final answer.

        Strategy:
          - If 'FINAL ANSWER:' is present, split there.
          - Otherwise, last sentence is treated as final answer.
        """
        lowered = text.lower()
        marker = "final answer:"

        if marker in lowered:
            idx = lowered.index(marker)
            reasoning = text[:idx].strip()
            final = text[idx + len(marker):].strip()

            # Split reasoning into lines / pseudo-steps
            steps = [
                line.strip()
                for line in reasoning.splitlines()
                if line.strip()
            ]
            return steps, final

        # Fallback: cut on the last sentence
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) <= 1:
            return [], text

        reasoning = sentences[:-1]
        final = sentences[-1]
        return reasoning, final