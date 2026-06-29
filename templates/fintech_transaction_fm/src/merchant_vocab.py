"""Distributed merchant-vocabulary construction with long-tail aggregation.

InfoNCE — and any genuine merchant modeling (the recommendation head) — needs a
real vocabulary, not the hashing trick. TabFormer has ~100k unique merchants; at
Visa scale it's 150M+. We keep the **top-K** most frequent merchants as dedicated
ids and fold the long tail + inference-time OOV into a small set of shared
**aggregate buckets** — TREASURE's strategy for "mapping infrequent or new
entities to shared aggregated identifiers" to control vocab size.

The frequency count is a distributed Ray Data aggregation over the full table —
the precondition InfoNCE needs, and the thing NVIDIA's single-GPU RAPIDS path
can't do past one node. De-risk on TabFormer: top-10k merchants cover 95.5% of
transactions, so K≈10-20k captures almost everything while keeping the embedding
table (and the InfoNCE negative pool) a manageable size.

Vocab id layout (contiguous, after the tokenizer's reserved PAD/MASK/OOV ids):

    [0 .. base)               reserved ids (owned by the tokenizer)
    [base .. base+K)          the top-K merchants, by descending frequency
    [base+K .. base+K+A)      A aggregate buckets; tail + OOV merchants land here

So a learned merchant field has ``base + K + A`` embedding rows. The mapping is
fully deterministic given the (small) top-K table, so it broadcasts to the
tokenize workers and applies as a stateless per-card map.
"""

from __future__ import annotations

import numpy as np


def build_merchant_vocab(
    ds,
    top_k: int,
    n_aggregate: int,
    base: int,
    merchant_col: str = "merchant_id",
) -> dict:
    """Count merchant frequencies across the dataset and keep the top-K.

    ``ds`` is a Ray Dataset of raw transactions. The groupby-count is the
    distributed step; the resulting per-merchant table (≈100k rows on TabFormer)
    is small enough to bring to the driver to pick the top-K.
    """
    counts = ds.groupby(merchant_col).count().to_pandas()
    count_col = "count()" if "count()" in counts.columns else "count"
    counts = counts.sort_values(count_col, ascending=False)

    top = counts[merchant_col].to_numpy()[:top_k].astype(np.int64)
    total = int(counts[count_col].sum())
    covered = int(counts[count_col].to_numpy()[:top_k].sum())
    return {
        # JSON keys must be strings; merchant ids are int64 (can be negative).
        "top_merchants": {str(int(m)): i for i, m in enumerate(top)},
        "top_k": int(len(top)),
        "n_aggregate": int(n_aggregate),
        "base": int(base),
        "n_unique": int(len(counts)),
        "coverage": covered / max(total, 1),
    }


def _top_lookup(vocab: dict) -> dict:
    """Rebuild the {merchant_id(int) -> rank} dict from the serialized vocab."""
    return {int(m): r for m, r in vocab["top_merchants"].items()}


def merchant_to_id(merchant_id: np.ndarray, vocab: dict, lookup: dict | None = None) -> np.ndarray:
    """Map raw merchant ids to embedding ids (top-K dedicated, rest aggregated).

    Tail and out-of-vocabulary merchants hash into the aggregate-bucket range, so
    new merchants seen only at inference always get a valid (shared) id — the OOV
    handling TREASURE notes is required in production.
    """
    lookup = lookup if lookup is not None else _top_lookup(vocab)
    base, k, a = vocab["base"], vocab["top_k"], vocab["n_aggregate"]
    m = merchant_id.astype(np.int64)
    # Tail/OOV bucket: abs() because TabFormer merchant ids are signed 64-bit hashes.
    agg = base + k + (np.abs(m) % a)
    out = agg.copy()
    if lookup:
        ranks = np.array([lookup.get(int(x), -1) for x in m], dtype=np.int64)
        hit = ranks >= 0
        out[hit] = base + ranks[hit]
    return out


def merchant_vocab_rows(vocab: dict) -> int:
    """Total embedding rows for a learned merchant field (incl. reserved base)."""
    return vocab["base"] + vocab["top_k"] + vocab["n_aggregate"]
