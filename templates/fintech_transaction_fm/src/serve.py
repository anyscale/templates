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
* Requests are **micro-batched** (``@serve.batch``) into one forward pass, and
  the forward pass runs in a worker thread so it never blocks the replica's
  asyncio event loop.

The model is small (~29M params at `full`), so this serves on CPU — a whole GPU
would sit idle. To serve on GPU instead, add a fractional
``ray_actor_options={"num_gpus": 0.25}`` and move the model to ``cuda``.

In production you'd back the cache with the feature store and load the XGBoost
head trained in ``downstream.py``; here we keep it self-contained.
"""

from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import torch

from ray import serve
from starlette.requests import Request

NUM_CPUS = 2


@serve.deployment(
    ray_actor_options={"num_cpus": NUM_CPUS},
    # Hard per-replica concurrency cap (the Ray default of 5 would sit *below*
    # the autoscaling target and starve it); autoscaling targets ~70% of it.
    max_ongoing_requests=32,
    autoscaling_config={"min_replicas": 1, "max_replicas": 4, "target_ongoing_requests": 24},
)
class TransactionEmbeddingService:
    def __init__(self, checkpoint_dir: str):
        from .model import build_model

        # Honor the CPU request — torch otherwise grabs every core on the node.
        torch.set_num_threads(NUM_CPUS)

        with open(os.path.join(checkpoint_dir, "model_config.json")) as f:
            mcfg = json.load(f)
        with open(os.path.join(checkpoint_dir, "vocab.json")) as f:
            self.vocab = json.load(f)

        self.model = build_model(
            os.path.join(checkpoint_dir, "vocab.json"),
            arch=mcfg["arch"],
            max_len=mcfg["max_len"],
        )
        state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self._static_cache: dict = {}  # card_id -> static field ids

    def _to_tensors(self, payloads: list) -> dict:
        """Stack N request payloads into one left-padded model batch.

        Windows may differ in length across requests; PAD, attention 0 and the
        soft-amount weight are all 0, so left-padding with zeros is exact.
        """
        cols = (
            [f"d_{f}" for f in self.vocab["dynamic_fields"]]
            + [f"s_{f}" for f in self.vocab["static_fields"]]
            + ["attention_mask"]
        )
        if self.vocab.get("time_aware"):
            cols.append("time_bucket")
        if self.vocab.get("amount_mode") == "soft":
            cols.append("d_amount_frac")

        seq_len = max(np.asarray(p["attention_mask"]).size for p in payloads)
        out = {}
        for k in cols:
            is_sequence = k.startswith("d_") or k in ("attention_mask", "time_bucket")
            dtype = torch.float32 if k == "d_amount_frac" else torch.long
            if is_sequence:
                mat = np.zeros((len(payloads), seq_len), dtype=np.float64)
                for i, p in enumerate(payloads):
                    arr = np.asarray(p[k]).reshape(-1)
                    mat[i, seq_len - arr.size :] = arr
                out[k] = torch.as_tensor(mat, dtype=dtype)
            else:
                out[k] = torch.as_tensor(
                    np.asarray([int(np.asarray(p[k]).reshape(())) for p in payloads]),
                    dtype=dtype,
                )
        return out

    def _embed_many(self, payloads: list) -> list:
        """One forward pass over a micro-batch (runs in a worker thread)."""
        card_ids = []
        for payload in payloads:
            # Cache static (card-level) ids — cheap, and they rarely change.
            card_id = int(payload.get("card_id", -1))
            card_ids.append(card_id)
            for f in self.vocab["static_fields"]:
                key = f"s_{f}"
                if key in payload:
                    self._static_cache.setdefault(card_id, {})[key] = payload[key]
                else:
                    payload[key] = self._static_cache.get(card_id, {}).get(key, 0)

        tensors = self._to_tensors(payloads)
        with torch.inference_mode():
            emb = self.model.sequence_embedding(tensors).numpy()
        results = []
        for card_id, e in zip(card_ids, emb):
            # Placeholder fraud score: norm-based proxy until the XGBoost head is
            # loaded. Swap in the downstream model for a real probability.
            score = float(1.0 / (1.0 + np.exp(-np.linalg.norm(e) / np.sqrt(len(e)) + 3.0)))
            results.append(
                {"card_id": card_id, "embedding": e.tolist(), "fraud_score": score}
            )
        return results

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def embed(self, payloads: list) -> list:
        # to_thread keeps the blocking torch forward off the event loop, so the
        # replica can keep accepting/queueing requests while a batch runs.
        return await asyncio.to_thread(self._embed_many, payloads)

    async def __call__(self, request: Request) -> dict:
        payload = await request.json()
        return await self.embed(payload)


def build_app(checkpoint_dir: str):
    """Return a bound Serve application (deploy with ``serve.run``)."""
    return TransactionEmbeddingService.bind(checkpoint_dir=checkpoint_dir)
