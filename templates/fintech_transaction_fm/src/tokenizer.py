"""Static / dynamic transaction tokenizer — the heart of the template.

NVIDIA's blueprint flattens every transaction into ~12 tokens in one shared
vocabulary, so a sequence of N transactions costs ~12N positions. We instead use
the **static/dynamic split** (FATA-Trans / Visa TREASURE):

* STATIC fields (issuer, card type, BIN region, home state) describe the *card*
  and never change within a sequence. They are embedded **once** and added to
  every position — they never spend sequence length.
* DYNAMIC fields (amount, merchant, MCC, time) describe each *transaction*. Each
  field has its own embedding table; a transaction occupies **one** position
  whose vector is the sum of its field embeddings.

So a sequence of N transactions is N positions, not 12N — cheaper *and* a
stronger inductive bias. The vocabulary is fully deterministic (fixed buckets +
hashing), so tokenization is a stateless, embarrassingly parallel `map_batches`
with no global aggregation — exactly the workload Ray Data is built for.

Set ``SPLIT_FIELDS = False`` to recover the NVIDIA-style flat baseline (variant
A) for an A/B comparison — left as a documented extension point.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .generate_data import (
    BIN_REGIONS,
    CARD_TYPES,
    ISSUERS,
    MCC_BY_CATEGORY,
    MERCHANT_CATEGORIES,
    STATES,
)

SPLIT_FIELDS = True  # False -> NVIDIA-style flat tokenization (extension point)
AMOUNT_MODE = "hard"  # "hard" (default, verified) | "soft" (blend adjacent bins)

# Reserved token ids (shared across every field table).
PAD = 0
MASK = 1
_RESERVED = 2

SEQ_LEN_BY_SCALE = {"smoke": 32, "small": 64, "medium": 64}

# --- Deterministic field vocabularies (no data scan needed) ---
N_AMOUNT_BUCKETS = 16
N_MERCHANT_BUCKETS = 2000
_ALL_MCC = sorted({m for codes in MCC_BY_CATEGORY.values() for m in codes})

# log10(amount) bucket edges spanning ~$0.10 .. ~$100k.
#
# Amount representation is pluggable. We bucket (robust to the heavy tail, plays
# nicely with the masked-classification objective). Alternatives, worst→best for
# money: (1) raw log1p+z-score scalar via Linear(1,d) — simple but weak and forks
# the objective into regression; (2) hard buckets (here); (3) *soft* binning
# (interpolate the two adjacent bin embeddings — removes hard boundaries); and
# (4) learned numerical embeddings — piecewise-linear (PLE) or periodic/Fourier
# features (Gorishniy et al., "On Embeddings for Numerical Features"). See
# DESIGN.md for the tradeoffs; bucketing is the default, PLE/periodic the upgrade.
_AMOUNT_EDGES = np.linspace(-1.0, 5.0, N_AMOUNT_BUCKETS - 1)

# Time-aware positions: bucket the log10(hours) gap since the previous
# transaction, embedded and added alongside the ordinal position. Transactions
# care about *when* (the inter-event gap), not just ordinal slot.
TIME_AWARE = True
N_TIME_BUCKETS = 16
_TIME_EDGES = np.linspace(-2.0, 4.0, N_TIME_BUCKETS - 1)  # ~0.01h .. ~1yr


def _index_map(values: list) -> dict:
    """Map a fixed list of category values to ids starting after reserved ids."""
    return {v: i + _RESERVED for i, v in enumerate(values)}


STATIC_VOCAB = {
    "issuer": _index_map(ISSUERS),
    "card_type": _index_map(CARD_TYPES),
    "bin_region": _index_map(BIN_REGIONS),
    "home_state": _index_map(STATES),
}

DYNAMIC_CATEGORICAL_VOCAB = {
    "merchant_category": _index_map(MERCHANT_CATEGORIES),
    "mcc": _index_map(_ALL_MCC),
    "hour": _index_map(list(range(24))),
    "day_of_week": _index_map(list(range(7))),
}

# Order of dynamic fields fed to the model (one embedding table each).
DYNAMIC_FIELDS = [
    "amount_bucket",
    "merchant_bucket",
    "merchant_category",
    "mcc",
    "hour",
    "day_of_week",
]
STATIC_FIELDS = ["issuer", "card_type", "bin_region", "home_state"]


def field_vocab_sizes() -> dict:
    """Number of embedding rows per field (including reserved ids)."""
    sizes = {
        "amount_bucket": N_AMOUNT_BUCKETS + _RESERVED,
        "merchant_bucket": N_MERCHANT_BUCKETS + _RESERVED,
    }
    for f, v in DYNAMIC_CATEGORICAL_VOCAB.items():
        sizes[f] = len(v) + _RESERVED
    for f, v in STATIC_VOCAB.items():
        sizes[f] = len(v) + _RESERVED
    return sizes


def _amount_to_bucket(amount: np.ndarray) -> np.ndarray:
    logs = np.log10(np.clip(amount, 0.1, None))
    return np.digitize(logs, _AMOUNT_EDGES) + _RESERVED


def _amount_to_soft(amount: np.ndarray):
    """Soft binning: return (lower_bin_id, frac) where the embedding is
    (1-frac)*emb[lower] + frac*emb[lower+1]. `frac` is the position between the
    two nearest bin edges, so adjacent amounts get near-identical vectors."""
    logs = np.log10(np.clip(amount, 0.1, None))
    cont = np.interp(logs, _AMOUNT_EDGES, np.arange(len(_AMOUNT_EDGES), dtype=np.float64))
    lower = np.floor(cont).astype(np.int64)
    frac = (cont - lower).astype(np.float32)
    return lower + _RESERVED, frac


def _merchant_to_bucket(merchant_id: np.ndarray) -> np.ndarray:
    # Hashing trick: stable hash into a fixed bucket space (no global vocab, no
    # cold-start). Collisions are intentional — unrelated merchants share a row,
    # disambiguated by the other fields. Raise N_MERCHANT_BUCKETS (or use
    # multi-hash) in production to cut the collision rate.
    return (merchant_id.astype(np.int64) % N_MERCHANT_BUCKETS) + _RESERVED


def _delta_to_bucket(delta_hours: np.ndarray) -> np.ndarray:
    logs = np.log10(np.clip(delta_hours, 1e-2, None))
    return np.digitize(logs, _TIME_EDGES) + _RESERVED


def n_time_bucket_rows() -> int:
    """Embedding rows for the time-gap table (incl. reserved + headroom)."""
    return N_TIME_BUCKETS + _RESERVED + 2


def write_vocab(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path.rstrip("/")), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "split_fields": SPLIT_FIELDS,
                "dynamic_fields": DYNAMIC_FIELDS,
                "static_fields": STATIC_FIELDS,
                "field_vocab_sizes": field_vocab_sizes(),
                "time_aware": TIME_AWARE,
                "n_time_buckets": n_time_bucket_rows(),
                "amount_mode": AMOUNT_MODE,
                "pad": PAD,
                "mask": MASK,
            },
            f,
            indent=2,
        )


def make_tokenize_group_fn(seq_len: int):
    """Build a `map_groups` UDF that turns one card's transactions into one
    padded, per-field token sequence.

    A closure (not a callable class) because Ray Data's ``map_groups`` keys off
    ``fn.__name__``. The function is stateless — safe at high concurrency.
    """

    def tokenize_group(group: dict) -> dict:
        n = len(group["card_id"])
        order = np.argsort(group["timestamp"])
        if n > seq_len:
            order = order[-seq_len:]  # keep most recent
        L = len(order)
        pad = seq_len - L

        def padded(values):
            arr = np.asarray(values)[order]
            return np.concatenate([np.full(pad, PAD, dtype=np.int64), arr.astype(np.int64)])

        def padded_f(values):
            arr = np.asarray(values)[order]
            return np.concatenate([np.zeros(pad, dtype=np.float32), arr.astype(np.float32)])

        dyn = {
            "amount_bucket": padded(_amount_to_bucket(np.asarray(group["amount"]))),
            "merchant_bucket": padded(_merchant_to_bucket(np.asarray(group["merchant_id"]))),
            "merchant_category": padded(
                [DYNAMIC_CATEGORICAL_VOCAB["merchant_category"][c] for c in group["merchant_category"]]
            ),
            "mcc": padded([DYNAMIC_CATEGORICAL_VOCAB["mcc"][int(m)] for m in group["mcc"]]),
            "hour": padded([DYNAMIC_CATEGORICAL_VOCAB["hour"][int(h)] for h in group["hour"]]),
            "day_of_week": padded(
                [DYNAMIC_CATEGORICAL_VOCAB["day_of_week"][int(d)] for d in group["day_of_week"]]
            ),
        }
        attn = np.concatenate([np.zeros(pad, dtype=np.int64), np.ones(L, dtype=np.int64)])

        # Time-aware position signal: hours since the previous transaction,
        # log-bucketed. First event in the window has gap 0.
        ts_sorted = np.asarray(group["timestamp"])[order]
        t = np.array(ts_sorted, dtype="datetime64[s]")
        deltas = np.zeros(L, dtype=np.float64)
        if L > 1:
            deltas[1:] = (t[1:] - t[:-1]) / np.timedelta64(1, "h")
        time_vals = _delta_to_bucket(deltas).astype(np.int64)
        time_padded = np.concatenate([np.full(pad, PAD, dtype=np.int64), time_vals])

        # Static fields are scalars (first row — constant within the card).
        out = {
            "card_id": np.array([int(group["card_id"][0])]),
            "length": np.array([L]),
            "attention_mask": [attn],
            "time_bucket": [time_padded],
            "label": np.array([int(np.max(group["is_fraud"]))]),
        }
        for f in STATIC_FIELDS:
            out[f"s_{f}"] = np.array([STATIC_VOCAB[f][group[f][0]]])
        for f in DYNAMIC_FIELDS:
            out[f"d_{f}"] = [dyn[f]]

        if AMOUNT_MODE == "soft":
            # Overwrite the hard amount id with the lower bin + emit the weight.
            lower, frac = _amount_to_soft(np.asarray(group["amount"]))
            out["d_amount_bucket"] = [padded(lower)]
            out["d_amount_frac"] = [padded_f(frac)]
        return out

    return tokenize_group


def tokenize_dataset(ds, seq_len: int):
    """Apply the tokenizer over a Ray Dataset grouped by card."""
    return ds.groupby("card_id").map_groups(
        make_tokenize_group_fn(seq_len), batch_format="numpy"
    )
