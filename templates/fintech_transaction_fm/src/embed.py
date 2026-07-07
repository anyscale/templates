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
            arch=mcfg["arch"],
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
        with torch.inference_mode():
            emb = self.model.sequence_embedding(tensors, pooling=self.pooling).cpu().numpy()
        passthrough = [
            "card_id", "row_id", "label", "split", "weight",
            "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts",
        ]
        out = {k: batch[k] for k in passthrough if k in batch}
        out["embedding"] = [row for row in emb.astype(np.float32)]
        return out


def extract_embeddings(
    tokenized_path: str | None = None,
    checkpoint_dir: str = "",
    output_path: str = "",
    num_workers: int = 2,
    use_gpu: bool = False,
    batch_size: int = 256,
    pooling: str = "last",
    ds=None,
    gpus_per_worker: float | None = None,
) -> str:
    """Run distributed batch embedding extraction and write Parquet.

    ``ds`` may be any Ray Dataset of tokenized eval windows — including a lazy
    one, so upstream CPU tokenization streams straight into the GPU actors
    through the object store. Falls back to reading ``tokenized_path``.

    ``gpus_per_worker`` (only meaningful with ``use_gpu``) may be fractional:
    the FM uses a few hundred MB of VRAM, so 0.25-0.5 packs several replicas
    per GPU instead of reserving a whole, mostly-idle device each. Defaults to
    a full GPU per replica.
    """
    import ray

    if ds is None:
        ds = ray.data.read_parquet(tokenized_path)
        # Job-level checkpointing (Anyscale runtime): processed rows are
        # recorded by row_id, so a resubmitted/cancelled job resumes
        # mid-dataset instead of re-embedding everything — the batch-inference
        # twin of Ray Train's epoch checkpoints, and what makes spot GPUs safe
        # here. https://docs.anyscale.com/runtime/data
        # Import path moved across runtime versions; OSS Ray has neither ->
        # silently skipped (local smoke). Parquet path only: row_id is in the
        # schema and the input is durable.
        # NOTE: the manifest is deleted on success (default) ON PURPOSE — 04
        # re-runs embed the SAME row_ids with a retrained model; a persisted
        # manifest would skip every row and emit nothing for the new model.
        # Failure/cancel keeps it, which is exactly when resume is wanted.
        try:
            from ray.data.checkpoint import CheckpointConfig
        except ImportError:
            try:
                from ray.anyscale.data.checkpoint import CheckpointConfig
            except ImportError:
                CheckpointConfig = None
        if CheckpointConfig is not None and "row_id" in ds.schema().names:
            ray.data.DataContext.get_current().checkpoint_config = CheckpointConfig(
                id_column="row_id",
                checkpoint_path=output_path.rstrip("/") + "_checkpoint",
            )
            print("[embed] Ray Data job-level checkpointing enabled (row_id)")
    if gpus_per_worker is None:
        gpus_per_worker = 1.0
    ds = ds.map_batches(
        EmbeddingExtractor,
        fn_constructor_kwargs={"checkpoint_dir": checkpoint_dir, "pooling": pooling},
        batch_size=batch_size,
        # max_tasks_in_flight: queue several batches per actor (default 4 is
        # sized for slow models) so the fast FM forward is never input-starved.
        compute=ray.data.ActorPoolStrategy(size=num_workers, max_tasks_in_flight_per_actor=16),
        num_gpus=gpus_per_worker if use_gpu else 0,
        # GPU replicas shouldn't also hold CPU slots — those belong to the
        # upstream tokenizer tasks feeding this stage.
        num_cpus=0 if use_gpu else 1,
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

    df = ray.data.read_parquet(output_path, columns=["embedding"]).limit(sample).to_pandas()
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
