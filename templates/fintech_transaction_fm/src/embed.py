"""Batch embedding extraction with Ray Data.

Once the FM is trained, the recurring production job is: score every customer to
produce a fresh embedding. This is a heterogeneous workload — CPU for the
Parquet read, GPU for the forward pass — that streams through one Ray Data
pipeline with no intermediate disk writes. There is no clean public reference
for this stage in the transaction-FM literature; it's where Ray earns its keep.

`EmbeddingExtractor` is a Ray Data callable class: the model loads ONCE per
worker replica in ``__init__`` and embeds batches in ``__call__``. Ray Data
manages replica lifecycle and (with ``num_gpus``) GPU placement.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


class EmbeddingExtractor:
    def __init__(self, checkpoint_dir: str, pooling: str = "last"):
        from .model import build_model

        self.pooling = pooling

        with open(os.path.join(checkpoint_dir, "model_config.json")) as f:
            mcfg = json.load(f)
        with open(os.path.join(checkpoint_dir, "vocab.json")) as f:
            self.vocab = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(
            os.path.join(checkpoint_dir, "vocab.json"),
            size=mcfg["size"],
            max_len=mcfg["max_len"],
        )
        state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[embed] loaded TransactionFM on {self.device}")

    def __call__(self, batch: dict) -> dict:
        cols = (
            [f"d_{f}" for f in self.vocab["dynamic_fields"]]
            + [f"s_{f}" for f in self.vocab["static_fields"]]
            + ["attention_mask"]
        )
        if self.vocab.get("time_aware"):
            cols.append("time_bucket")

        def to_tensor(k, dtype):
            v = np.stack(batch[k]) if batch[k].dtype == object else batch[k]
            return torch.as_tensor(v, dtype=dtype, device=self.device)

        tensors = {k: to_tensor(k, torch.long) for k in cols}
        if self.vocab.get("amount_mode") == "soft":
            tensors["d_amount_frac"] = to_tensor("d_amount_frac", torch.float32)
        with torch.no_grad():
            emb = self.model.sequence_embedding(tensors, pooling=self.pooling).cpu().numpy()
        passthrough = [
            "card_id", "label", "split", "weight",
            "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts",
        ]
        out = {k: batch[k] for k in passthrough if k in batch}
        out["embedding"] = [row for row in emb.astype(np.float32)]
        return out


def extract_embeddings(
    tokenized_path: str,
    checkpoint_dir: str,
    output_path: str,
    num_workers: int = 2,
    use_gpu: bool = False,
    batch_size: int = 256,
    pooling: str = "last",
) -> str:
    """Run distributed batch embedding extraction and write Parquet."""
    import ray

    ds = ray.data.read_parquet(tokenized_path)
    ds = ds.map_batches(
        EmbeddingExtractor,
        fn_constructor_kwargs={"checkpoint_dir": checkpoint_dir, "pooling": pooling},
        batch_size=batch_size,
        concurrency=num_workers,
        num_gpus=1 if use_gpu else 0,
        batch_format="numpy",
    )
    ds.write_parquet(output_path)
    print(f"[embed] wrote customer embeddings -> {output_path}")
    embedding_health(output_path)
    return output_path


def embedding_health(output_path: str, sample: int = 2000) -> dict:
    """Cheap representation-collapse check on a sample of the embeddings.

    The classic self-supervised failure mode is silent collapse — every customer
    maps to nearly the same vector while the loss looks fine. We report mean
    pairwise cosine similarity (→1.0 means collapse) and mean feature variance
    (→0 means collapse). Mean pairwise cosine is computed in closed form from the
    summed unit vectors, so it's O(n·d), not O(n²).
    """
    import ray

    df = ray.data.read_parquet(output_path).limit(sample).to_pandas()
    X = np.vstack(df["embedding"].to_numpy()).astype(np.float64)
    n = len(X)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    s = Xn.sum(axis=0)
    mean_pair_cos = float((s @ s - n) / (n * (n - 1))) if n > 1 else float("nan")
    mean_var = float(X.var(axis=0).mean())
    flag = " ⚠️ possible collapse" if mean_pair_cos > 0.9 else ""
    print(
        f"[embed] health: mean_pairwise_cos={mean_pair_cos:.3f} (→1 = collapse), "
        f"mean_feature_var={mean_var:.4f}, n={n}{flag}"
    )
    return {"mean_pairwise_cos": mean_pair_cos, "mean_feature_var": mean_var, "n": n}
