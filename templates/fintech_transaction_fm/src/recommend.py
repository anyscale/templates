"""Next-merchant recommendation eval — the second consumer of the backbone.

Same pretrained FM, same batch-inference shape as ``embed.py``, different head:
instead of pooling an embedding for fraud, we **mask the target transaction and
rank the next merchant** (BERT4Rec-style — masked-item prediction is SOTA-
competitive with causal models for sequential recommendation). Scores come from
the InfoNCE-tied merchant embedding table, so this only runs on the *learned*
merchant-vocab path (the hashed path has no merchant table to rank against).

For each eval window (which ends at the target transaction, right-aligned):

* mask **every dynamic field** at the last position — so the model predicts the
  next merchant from prior history only, not from the target's own amount/MCC
  (which would leak: MCC ≈ merchant category). The model saw fully-masked
  positions throughout MLM training, so this is in-distribution.
* score the projected last-position state against the full merchant table,
  exclude reserved ids, and record the rank of the true merchant.

We report HR@K / NDCG@K on the test split, next to a **frequency baseline**
(recommend the card's most-frequent prior merchants) computed from the same
window — the honest "is the FM beating 'just show them what they always buy?'"
comparison. De-risk floor on TabFormer: top-10 historical merchants ≈ 65%.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

K_VALUES = (1, 5, 10, 20)


class NextMerchantScorer:
    """Ray Data callable: score the masked next merchant for each eval window."""

    def __init__(self, checkpoint_dir: str):
        from .model import build_model
        from .tokenizer import MASK, _RESERVED

        with open(os.path.join(checkpoint_dir, "model_config.json")) as f:
            mcfg = json.load(f)
        with open(os.path.join(checkpoint_dir, "vocab.json")) as f:
            self.vocab = json.load(f)
        if "merchant_bucket" not in self.vocab.get("infonce_fields", []):
            raise RuntimeError(
                "recommendation eval needs the learned merchant vocab + InfoNCE "
                "head — re-run with tokenize.merchant_vocab: learned"
            )

        self.MASK = MASK
        self.reserved = _RESERVED
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(
            os.path.join(checkpoint_dir, "vocab.json"),
            arch=mcfg["arch"],
            max_len=mcfg["max_len"],
        )
        state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        # base id of the first dedicated (top-K) merchant — targets at/above this
        # but below base+top_k are "dedicated"; the rest are aggregate buckets.
        mv = self.vocab["merchant_vocab"]
        self.dedicated_hi = mv["base"] + mv["top_k"]
        print(f"[recommend] scoring next-merchant on {self.device}")

    def __call__(self, batch: dict) -> dict:
        dyn = self.vocab["dynamic_fields"]
        cols = [f"d_{f}" for f in dyn] + [f"s_{f}" for f in self.vocab["static_fields"]]
        cols += ["attention_mask"]
        if self.vocab.get("time_aware"):
            cols.append("time_bucket")

        def t(k, dtype):
            v = np.stack(batch[k]) if batch[k].dtype == object else batch[k]
            return torch.as_tensor(v, dtype=dtype, device=self.device)

        tensors = {k: t(k, torch.long) for k in cols}
        if self.vocab.get("amount_mode") == "soft":
            tensors["d_amount_frac"] = t("d_amount_frac", torch.float32)

        # True next merchant = target (last) position before masking.
        true_merch = tensors["d_merchant_bucket"][:, -1].clone()
        # Mask the whole target position so prediction uses history only.
        for f in dyn:
            tensors[f"d_{f}"][:, -1] = self.MASK
        if "d_amount_frac" in tensors:
            tensors["d_amount_frac"][:, -1] = 0.0

        with torch.no_grad():
            hidden = self.model.encode(tensors)[:, -1, :]              # (B, D)
            proj = self.model.infonce_proj["merchant_bucket"](hidden)  # (B, D)
            E = self.model.dyn_emb["merchant_bucket"].weight           # (V, D)
            scores = proj @ E.t()                                      # (B, V)
            scores[:, : self.reserved] = float("-inf")                 # drop PAD/MASK/OOV
            true_score = scores.gather(1, true_merch[:, None]).squeeze(1)
            rank = (scores > true_score[:, None]).sum(dim=1)           # 0-indexed rank

        rank = rank.cpu().numpy().astype(np.int64)
        true_np = true_merch.cpu().numpy()

        # Frequency baseline: rank the card's prior merchants by recurrence.
        merch_hist = (
            np.stack(batch["d_merchant_bucket"])
            if batch["d_merchant_bucket"].dtype == object
            else batch["d_merchant_bucket"]
        )
        attn = (
            np.stack(batch["attention_mask"])
            if batch["attention_mask"].dtype == object
            else batch["attention_mask"]
        )
        B = len(true_np)
        base_hit = {k: np.zeros(B, bool) for k in K_VALUES}
        for i in range(B):
            valid = attn[i, :-1] == 1
            h = merch_hist[i, :-1][valid]
            if h.size == 0:
                continue
            vals, cnts = np.unique(h, return_counts=True)
            ranked = vals[np.argsort(cnts)[::-1]]
            for k in K_VALUES:
                base_hit[k][i] = true_np[i] in ranked[:k]

        out = {
            "split": batch["split"],
            "model_rank": rank,
            "dedicated": true_np < self.dedicated_hi,
        }
        for k in K_VALUES:
            out[f"base_hit_{k}"] = base_hit[k]
        return out


def run_recommend(
    checkpoint_dir: str,
    output_dir: str,
    tokenized_path: str | None = None,
    ds=None,
    num_workers: int = 2,
    use_gpu: bool = False,
    batch_size: int = 256,
    split: str = "test",
) -> dict:
    """Score next-merchant over eval windows; report HR@K / NDCG@K on ``split``.

    ``ds`` is a Ray Dataset of tokenized eval windows (lazy is fine — it streams
    CPU tokenize → GPU scoring like the embedding pass). The per-window outputs
    are a handful of ints, so collecting the chosen split to the driver is cheap
    even at full scale (unlike the embeddings).
    """
    import ray

    if ds is None:
        ds = ray.data.read_parquet(tokenized_path)
    scored = ds.map_batches(
        NextMerchantScorer,
        fn_constructor_kwargs={"checkpoint_dir": checkpoint_dir},
        batch_size=batch_size,
        compute=ray.data.ActorPoolStrategy(size=num_workers),
        num_gpus=1 if use_gpu else 0,
        batch_format="numpy",
    )
    cols = ["split", "model_rank", "dedicated"] + [f"base_hit_{k}" for k in K_VALUES]
    df = scored.select_columns(cols).to_pandas()
    df = df[df["split"] == split]
    if len(df) == 0:
        raise RuntimeError(f"no '{split}'-split eval windows to score")

    rank = df["model_rank"].to_numpy()
    summary = {
        "split": split,
        "n_samples": int(len(df)),
        "dedicated_target_rate": float(df["dedicated"].mean()),
        "model": {
            **{f"hr@{k}": float((rank < k).mean()) for k in K_VALUES},
            **{
                f"ndcg@{k}": float(
                    np.where(rank < k, 1.0 / np.log2(rank + 2), 0.0).mean()
                )
                for k in K_VALUES
            },
        },
        "frequency_baseline": {
            f"hr@{k}": float(df[f"base_hit_{k}"].mean()) for k in K_VALUES
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "recommend_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary: dict) -> None:
    print(
        f"next-merchant recommendation ({summary['split']} split, "
        f"{summary['n_samples']:,} windows; "
        f"{summary['dedicated_target_rate']:.1%} of targets are dedicated merchants)"
    )
    print(f"{'metric':<10} {'FM':>10} {'freq-base':>12}")
    print("-" * 34)
    m, b = summary["model"], summary["frequency_baseline"]
    for k in K_VALUES:
        print(f"{'HR@' + str(k):<10} {m[f'hr@{k}']:>10.4f} {b[f'hr@{k}']:>12.4f}")
    for k in K_VALUES:
        print(f"{'NDCG@' + str(k):<10} {m[f'ndcg@{k}']:>10.4f} {'—':>12}")
