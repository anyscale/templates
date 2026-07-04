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


def _truncate_to_last(batch: dict, K: int) -> dict:
    """Keep each window's last K real tokens (BOS + most-recent txns … EOS).

    Windows are right-padded [BOS, txn…, EOS, pad…]. For a window longer than K
    we keep BOS + the final K-1 real tokens (which end in EOS/target), so the
    embedded context is the most-recent ~K/13 transactions — matching NVIDIA's
    NB04 MAX_LENGTH=128. Shorter windows are returned unchanged (re-padded to K).
    """
    ids = np.stack(batch["input_ids"]) if batch["input_ids"].dtype == object else np.asarray(batch["input_ids"])
    am = np.stack(batch["attention_mask"]) if batch["attention_mask"].dtype == object else np.asarray(batch["attention_mask"])
    B = ids.shape[0]
    n = am.sum(1).astype(int)
    out_ids = np.zeros((B, K), ids.dtype)
    out_am = np.zeros((B, K), am.dtype)
    for r in range(B):
        nr = n[r]
        if nr <= K:
            out_ids[r, :nr] = ids[r, :nr]; out_am[r, :nr] = 1
        else:
            out_ids[r, 0] = ids[r, 0]                       # BOS
            out_ids[r, 1:K] = ids[r, nr - (K - 1):nr]       # last K-1 real (…EOS)
            out_am[r, :K] = 1
    b = dict(batch)
    b["input_ids"] = out_ids
    b["attention_mask"] = out_am
    return b


class EmbeddingExtractor:
    def __init__(self, checkpoint_dir: str, pooling: str = "last", max_ctx: int | None = None):
        from .model import build_model

        self.pooling = pooling
        # NVIDIA embeds each transaction from only its most-recent context
        # (NB04 MAX_LENGTH=128 tokens ≈ 10 txns), even though pretraining uses
        # 4096. max_ctx truncates each window to its last `max_ctx` real tokens
        # so the customer vector reflects recent behavior, not a 314-txn average.
        self.max_ctx = max_ctx

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
        print(f"[embed] loaded TransactionFM (Llama) on {self.device}")

    def __call__(self, batch: dict) -> dict:
        # Flat token stream: the model needs only input_ids + attention_mask.
        def to_tensor(k):
            v = np.stack(batch[k]) if batch[k].dtype == object else batch[k]
            return torch.as_tensor(v, dtype=torch.long, device=self.device)

        if self.max_ctx:
            batch = _truncate_to_last(batch, self.max_ctx)
        tensors = {k: to_tensor(k) for k in ("input_ids", "attention_mask")}
        # bf16 autocast mirrors pretraining: at seq_len 4096 the fp32 attention
        # activations OOM even a 24 GiB A10G, and last-token pooling is unaffected
        # by the reduced precision. no_grad drops the autograd tape (inference).
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            emb = self.model.sequence_embedding(tensors, pooling=self.pooling).float().cpu().numpy()
        passthrough = [
            "card_id", "label", "split", "weight",
            "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts",
            # extended raw target features for the NVIDIA-style baseline
            "raw_use_chip", "raw_merchant_state", "raw_merchant_city",
            "raw_zip", "raw_merchant_id", "raw_card_id",
        ]
        out = {k: batch[k] for k in passthrough if k in batch}
        out["embedding"] = [row for row in emb.astype(np.float32)]
        return out


def balanced_eval_sample(tokenized_eval_path: str = None, balanced_train: int = 1_000_000,
                         seed: int = 1, ds=None):
    """Balanced train sample + full holdout test, drawn from tokenized eval.

    Matches NVIDIA NB04: sample BEFORE embedding, so we embed ~``balanced_train`` +
    the holdout (a few million windows) rather than all ~24M. The train sample keeps
    every fraud + a random ~10%-fraud mix of normals (``scale_pos_weight=1.0``
    downstream); the test split is kept whole for a stable held-out metric. Uses
    ``random_sample`` (cheap per-row Bernoulli). Pass ``ds`` (an in-memory eval
    Dataset, e.g. from the fused pipeline) or ``tokenized_eval_path`` (read from disk).
    Returns a Ray Dataset ready for ``extract_embeddings(ds=...)``.
    """
    import ray

    full = ds.materialize() if ds is not None else ray.data.read_parquet(tokenized_eval_path)

    def cnt(s, lab=None):
        d = full.filter(expr=f"split == '{s}'")
        if lab is not None:
            d = d.filter(expr=f"label == {lab}")
        return d.count()

    tr_fraud, tr_norm = cnt("train", 1), cnt("train", 0)
    nf = min(tr_fraud, int(balanced_train * 0.1))
    frac_tn = min(1.0, (balanced_train - nf) / max(tr_norm, 1))
    print(f"[embed] balanced train: {nf:,} fraud + ~{int(frac_tn * tr_norm):,} normal "
          f"(~{balanced_train:,}); test kept whole ({cnt('test'):,})", flush=True)

    tr = full.filter(expr="split == 'train'")
    train_ds = tr.filter(expr="label == 1").union(
        tr.filter(expr="label == 0").random_sample(frac_tn, seed=seed))
    return train_ds.union(full.filter(expr="split == 'test'"))


def extract_embeddings(
    tokenized_path: str | None = None,
    checkpoint_dir: str = "",
    output_path: str = "",
    num_workers: int = 2,
    use_gpu: bool = False,
    batch_size: int = 256,
    pooling: str = "last",
    max_ctx: int | None = None,
    ds=None,
) -> str:
    """Run distributed batch embedding extraction and write Parquet.

    ``ds`` may be any Ray Dataset of tokenized eval windows — including a lazy
    one, so upstream CPU tokenization streams straight into the GPU actors
    through the object store. Falls back to reading ``tokenized_path``.
    """
    import ray

    if ds is None:
        ds = ray.data.read_parquet(tokenized_path)
    ds = ds.map_batches(
        EmbeddingExtractor,
        fn_constructor_kwargs={"checkpoint_dir": checkpoint_dir, "pooling": pooling, "max_ctx": max_ctx},
        batch_size=batch_size,
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=num_workers),
        num_gpus=1 if use_gpu else 0,
        # Give each actor an explicit, non-zero footprint. On the CPU path
        # (num_gpus=0) an actor with no num_cpus has *zero* min scheduling
        # resources, so the autoscaler's bundle-count estimate is infinite and
        # it asserts during scale-up. num_cpus=1 makes the footprint finite.
        num_cpus=1,
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
