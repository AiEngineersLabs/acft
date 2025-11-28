# acft/embeddings/hf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import json
import os
import numpy as np
import requests

from .base import Embedder


@dataclass
class HFEmbeddingEmbedder(Embedder):
    """
    Embedding adapter for any OpenAI-compatible /v1/embeddings endpoint.

    Works with:
      - vLLM embeddings endpoint (if exposed)
      - OpenAI /v1/embeddings
      - gateways that mimic OpenAI embeddings

    The name "HF" here is just indicative; the real requirement is:
    the server supports the OpenAI embeddings JSON schema.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    timeout: int = 60  # seconds

    def _post_embeddings(self, text: str) -> List[float]:
        url = self.base_url.rstrip("/") + "/embeddings"

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        # Try common env vars if api_key is not given explicitly
        api_key = (
            self.api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("VLLM_API_KEY")
            or os.getenv("HF_API_TOKEN")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": text,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()

        try:
            data = resp.json()
        except json.JSONDecodeError:
            # If server returns raw list, handle that
            return json.loads(resp.text)

        # OpenAI-style:
        # {
        #   "data": [
        #     {
        #       "embedding": [ ... ],
        #       ...
        #     }
        #   ],
        #   ...
        # }
        if isinstance(data, dict):
            arr = data.get("data") or []
            if arr:
                emb = arr[0].get("embedding")
                if isinstance(emb, list):
                    return emb

        # Fallback
        raise RuntimeError(f"Unexpected embeddings response format: {data!r}")

    def embed(self, text: str) -> np.ndarray:
        vec = self._post_embeddings(text)
        v = np.asarray(vec, dtype=float)

        # Basic normalization / safety
        if v.ndim != 1:
            v = v.reshape(-1)
        if not np.all(np.isfinite(v)):
            v = np.where(np.isfinite(v), v, 0.0)

        n = np.linalg.norm(v)
        if n > 0:
            v = v / n

        return v