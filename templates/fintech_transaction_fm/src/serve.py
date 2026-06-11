"""Online embedding + fraud scoring with Ray Serve.

The default production path is batch (see ``embed.py``): precompute embeddings,
push to a feature store, look them up in milliseconds at authorization time. But
fintech teams always need a real-time path too, so this deployment shows the
two-tier pattern real shops use (e.g. Revolut PRAGMA: a small model online, a
bigger model in batch):

* Static (card-level) embeddings are **cached** — they only change when the card
  changes, so we never recompute them on the hot path.
* The transformer runs only over the recent dynamic window for the requesting
  card, returning an embedding + a fraud probability in one call.

In production you'd back the cache with the feature store and load the XGBoost
head trained in ``downstream.py``; here we keep it self-contained.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from ray import serve
from starlette.requests import Request


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4, "target_ongoing_requests": 8},
)
class TransactionEmbeddingService:
    def __init__(self, checkpoint_dir: str):
        from .model import build_model

        with open(os.path.join(checkpoint_dir, "model_config.json")) as f:
            mcfg = json.load(f)
        with open(os.path.join(checkpoint_dir, "vocab.json")) as f:
            self.vocab = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(
            os.path.join(checkpoint_dir, "vocab.json"), size=mcfg["size"], max_len=mcfg["max_len"]
        )
        state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._static_cache: dict = {}  # card_id -> static field ids

    def _to_tensors(self, payload: dict) -> dict:
        cols = (
            [f"d_{f}" for f in self.vocab["dynamic_fields"]]
            + [f"s_{f}" for f in self.vocab["static_fields"]]
            + ["attention_mask"]
        )
        if self.vocab.get("time_aware"):
            cols.append("time_bucket")
        if self.vocab.get("amount_mode") == "soft":
            cols.append("d_amount_frac")
        out = {}
        for k in cols:
            arr = np.asarray(payload[k])
            is_sequence = k.startswith("d_") or k in ("attention_mask", "time_bucket")
            if is_sequence and arr.ndim == 1:
                arr = arr[None, :]          # [S] -> [1, S]
            elif not is_sequence and arr.ndim == 0:
                arr = arr[None]             # scalar -> [1]
            dtype = torch.float32 if k == "d_amount_frac" else torch.long
            out[k] = torch.as_tensor(arr, dtype=dtype, device=self.device)
        return out

    def embed(self, payload: dict) -> dict:
        # Cache static (card-level) ids — cheap, and they rarely change.
        card_id = int(payload.get("card_id", -1))
        for f in self.vocab["static_fields"]:
            key = f"s_{f}"
            if key in payload:
                self._static_cache.setdefault(card_id, {})[key] = payload[key]
            else:
                payload[key] = self._static_cache.get(card_id, {}).get(key, 0)

        tensors = self._to_tensors(payload)
        with torch.no_grad():
            emb = self.model.sequence_embedding(tensors).cpu().numpy()[0]
        # Placeholder fraud score: norm-based proxy until the XGBoost head is
        # loaded. Swap in the downstream model for a real probability.
        score = float(1.0 / (1.0 + np.exp(-np.linalg.norm(emb) / np.sqrt(len(emb)) + 3.0)))
        return {"card_id": card_id, "embedding": emb.tolist(), "fraud_score": score}

    async def __call__(self, request: Request) -> dict:
        payload = await request.json()
        return self.embed(payload)


def build_app(checkpoint_dir: str):
    """Return a bound Serve application (deploy with ``serve.run``)."""
    return TransactionEmbeddingService.bind(checkpoint_dir=checkpoint_dir)
