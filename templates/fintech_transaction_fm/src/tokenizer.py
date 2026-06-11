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
hashing + a reserved OOV id), so tokenization is a stateless, embarrassingly
parallel ``map_groups`` with no global aggregation — exactly the workload Ray
Data is built for.

The tokenizer emits TWO kinds of samples per card (``kind`` column), following
the evaluation protocol of NVIDIA's transaction-FM blueprint on TabFormer:

* ``pretrain`` — non-overlapping windows of the card's **train-period** history
  (newest first), used for masked-feature-modeling pretraining. The FM never
  sees val/test-period transactions.
* ``eval``     — one window per *target transaction* (the window ends at it),
  labeled with that transaction's ``is_fraud`` and tagged ``split`` ∈
  {train,val,test} by the target's timestamp vs the temporal 80/10/10 cutoffs.
  All frauds are kept; normals are downsampled (deterministically per card).

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
    MERCHANT_CATEGORIES,
)

SPLIT_FIELDS = True  # False -> NVIDIA-style flat tokenization (extension point)
AMOUNT_MODE = "hard"  # "hard" (default, verified) | "soft" (blend adjacent bins)

# Reserved token ids (shared across every field table). OOV absorbs categorical
# values outside the fixed vocab — real-world data always has some.
PAD = 0
MASK = 1
OOV = 2
_RESERVED = 3

SEQ_LEN_BY_SCALE = {"smoke": 32, "small": 64, "medium": 64}

# --- Deterministic field vocabularies (no data scan needed) ---
N_AMOUNT_BUCKETS = 16
N_MERCHANT_BUCKETS = 2000
N_MCC_BUCKETS = 128  # hashed, like merchants — real data has open-ended MCC sets

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

# Vocab lists are supersets of the synthetic generator's palettes so the same
# tokenizer covers both synthetic and real (TabFormer) data. card_type gains the
# TabFormer transaction channels; home_state covers all US states plus the
# ONLINE/FOREIGN buckets the loader emits.
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
]


def _index_map(values: list) -> dict:
    """Map a fixed list of category values to ids starting after reserved ids."""
    return {v: i + _RESERVED for i, v in enumerate(values)}


STATIC_VOCAB = {
    "issuer": _index_map(ISSUERS + ["UNKNOWN"]),
    "card_type": _index_map(CARD_TYPES + ["swipe", "chip", "online", "UNKNOWN"]),
    "bin_region": _index_map(BIN_REGIONS + ["UNKNOWN"]),
    "home_state": _index_map(US_STATES + ["ONLINE", "FOREIGN", "UNKNOWN"]),
}

DYNAMIC_CATEGORICAL_VOCAB = {
    "merchant_category": _index_map(MERCHANT_CATEGORIES + ["other"]),
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
        "mcc": N_MCC_BUCKETS + _RESERVED,
    }
    for f, v in DYNAMIC_CATEGORICAL_VOCAB.items():
        sizes[f] = len(v) + _RESERVED
    for f, v in STATIC_VOCAB.items():
        sizes[f] = len(v) + _RESERVED
    return sizes


def _amount_to_bucket(amount: np.ndarray) -> np.ndarray:
    # abs(): refunds/credits are negative in real data; magnitude carries the signal.
    logs = np.log10(np.clip(np.abs(amount), 0.1, None))
    return np.digitize(logs, _AMOUNT_EDGES) + _RESERVED


def _amount_to_soft(amount: np.ndarray):
    """Soft binning: return (lower_bin_id, frac) where the embedding is
    (1-frac)*emb[lower] + frac*emb[lower+1]. `frac` is the position between the
    two nearest bin edges, so adjacent amounts get near-identical vectors."""
    logs = np.log10(np.clip(np.abs(amount), 0.1, None))
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


def _mcc_to_bucket(mcc: np.ndarray) -> np.ndarray:
    # Same hashing trick: MCC sets are open-ended in real data (TabFormer has
    # ~109 distinct codes). Coarse semantics live in merchant_category.
    return (mcc.astype(np.int64) % N_MCC_BUCKETS) + _RESERVED


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
                "oov": OOV,
            },
            f,
            indent=2,
        )


# Raw target-transaction features carried on eval samples so the downstream
# stage can fit the NVIDIA-style raw-feature baseline without re-reading data.
RAW_FEATURE_COLS = ["raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts"]


def _stack_rows(rows: list, seq_len: int, soft: bool) -> dict:
    """Stack per-row dicts into a numpy batch; emit dtype-correct empties."""
    if rows:
        out = {}
        for k, v0 in rows[0].items():
            if isinstance(v0, np.ndarray):
                out[k] = np.stack([r[k] for r in rows])
            elif isinstance(v0, str):
                out[k] = np.array([r[k] for r in rows], dtype=object)
            else:
                out[k] = np.array([r[k] for r in rows])
        return out
    out = {
        "card_id": np.zeros(0, np.int64),
        "length": np.zeros(0, np.int64),
        "kind": np.array([], dtype=object),
        "split": np.array([], dtype=object),
        "label": np.zeros(0, np.int64),
        "weight": np.zeros(0, np.float64),
        "attention_mask": np.zeros((0, seq_len), np.int64),
        "time_bucket": np.zeros((0, seq_len), np.int64),
        "raw_amount": np.zeros(0, np.float32),
        "raw_hour": np.zeros(0, np.int64),
        "raw_dow": np.zeros(0, np.int64),
        "raw_mcc": np.zeros(0, np.int64),
        "raw_ts": np.zeros(0, np.int64),
    }
    for f in STATIC_FIELDS:
        out[f"s_{f}"] = np.zeros(0, np.int64)
    for f in DYNAMIC_FIELDS:
        out[f"d_{f}"] = np.zeros((0, seq_len), np.int64)
    if soft:
        out["d_amount_frac"] = np.zeros((0, seq_len), np.float32)
    return out


def make_tokenize_group_fn(
    seq_len: int,
    train_end: str | None = None,
    val_end: str | None = None,
    normal_keep: float = 1.0,
    max_pretrain_windows: int | None = None,
):
    """Build a ``map_groups`` UDF turning one card's transactions into padded,
    per-field token sequences (pretrain windows + per-transaction eval samples).

    A closure (not a callable class) because Ray Data's ``map_groups`` keys off
    ``fn.__name__``. The function is stateless — safe at high concurrency; the
    eval-sample downsampling RNG is seeded per card, so output is deterministic.
    """
    t_train = np.datetime64(train_end, "s") if train_end else None
    t_val = np.datetime64(val_end, "s") if val_end else None
    soft = AMOUNT_MODE == "soft"

    def tokenize_group(group: dict) -> dict:
        n = len(group["card_id"])
        order = np.argsort(group["timestamp"])
        ts = np.asarray(group["timestamp"])[order].astype("datetime64[s]")
        fraud = np.asarray(group["is_fraud"]).astype(np.int64)[order]
        amounts = np.asarray(group["amount"]).astype(np.float64)[order]
        hours = np.asarray(group["hour"]).astype(np.int64)[order]
        dows = np.asarray(group["day_of_week"]).astype(np.int64)[order]
        mccs = np.asarray(group["mcc"]).astype(np.int64)[order]
        card_id = int(group["card_id"][0])

        mc_v = DYNAMIC_CATEGORICAL_VOCAB["merchant_category"]
        hr_v = DYNAMIC_CATEGORICAL_VOCAB["hour"]
        dw_v = DYNAMIC_CATEGORICAL_VOCAB["day_of_week"]
        # Tokenize the full sorted history once; windows below just slice it.
        full = {
            "amount_bucket": _amount_to_bucket(amounts),
            "merchant_bucket": _merchant_to_bucket(np.asarray(group["merchant_id"])[order]),
            "merchant_category": np.array(
                [mc_v.get(c, OOV) for c in np.asarray(group["merchant_category"])[order]],
                dtype=np.int64,
            ),
            "mcc": _mcc_to_bucket(mccs),
            "hour": np.array([hr_v.get(int(h), OOV) for h in hours], dtype=np.int64),
            "day_of_week": np.array([dw_v.get(int(d), OOV) for d in dows], dtype=np.int64),
        }
        if soft:
            soft_lo, soft_frac = _amount_to_soft(amounts)
        statics = {f: int(STATIC_VOCAB[f].get(group[f][0], OOV)) for f in STATIC_FIELDS}

        rows = []

        def emit(lo: int, hi: int, kind: str, split: str, label: int, target):
            L = hi - lo
            pad = seq_len - L
            # Importance weight undoing the normal-downsampling, so downstream
            # metrics estimate full-population (natural-prevalence) values.
            weight = 1.0 if (label == 1 or kind != "eval") else 1.0 / max(normal_keep, 1e-9)
            r = {
                "card_id": card_id, "length": L, "kind": kind, "split": split,
                "label": label, "weight": np.float64(weight),
            }
            for f, arr in full.items():
                r[f"d_{f}"] = np.concatenate([np.full(pad, PAD, np.int64), arr[lo:hi]])
            if soft:
                r["d_amount_bucket"] = np.concatenate(
                    [np.full(pad, PAD, np.int64), soft_lo[lo:hi]]
                )
                r["d_amount_frac"] = np.concatenate(
                    [np.zeros(pad, np.float32), soft_frac[lo:hi]]
                ).astype(np.float32)
            for f, v in statics.items():
                r[f"s_{f}"] = v
            r["attention_mask"] = np.concatenate(
                [np.zeros(pad, np.int64), np.ones(L, np.int64)]
            )
            # Inter-transaction gap, recomputed within the window (first gap 0).
            deltas = np.zeros(L, dtype=np.float64)
            if L > 1:
                deltas[1:] = (ts[lo + 1 : hi] - ts[lo : hi - 1]) / np.timedelta64(1, "h")
            r["time_bucket"] = np.concatenate(
                [np.full(pad, PAD, np.int64), _delta_to_bucket(deltas).astype(np.int64)]
            )
            if target is None:
                r.update(raw_amount=np.float32(0.0), raw_hour=0, raw_dow=0, raw_mcc=0, raw_ts=0)
            else:
                r.update(
                    raw_amount=np.float32(amounts[target]),
                    raw_hour=int(hours[target]),
                    raw_dow=int(dows[target]),
                    raw_mcc=int(mccs[target]),
                    raw_ts=int(ts[target].astype(np.int64)),
                )
            rows.append(r)

        # --- Pretraining windows: train-period only, newest first, non-overlapping.
        n_train = n if t_train is None else int(np.searchsorted(ts, t_train))
        hi, made = n_train, 0
        while hi >= 2 and (max_pretrain_windows is None or made < max_pretrain_windows):
            lo = max(0, hi - seq_len)
            emit(lo, hi, "pretrain", "train", 0, None)
            made += 1
            hi = lo

        # --- Eval samples: window ends at the target transaction; label = its
        # is_fraud; split by the target's timestamp. All frauds kept, normals
        # downsampled deterministically.
        rng = np.random.default_rng(card_id + 1)
        keep = (fraud == 1) | (rng.random(n) < normal_keep)
        if not rows and not keep.any():
            # Guarantee at least one output row per card: empty (0, seq_len)
            # batches break Ray's Arrow tensor conversion, and a card should
            # not silently vanish from the pipeline anyway.
            keep[n - 1] = True
        for t in np.nonzero(keep)[0]:
            if t_train is None or ts[t] < t_train:
                split = "train"
            elif t_val is None or ts[t] < t_val:
                split = "val"
            else:
                split = "test"
            emit(max(0, t + 1 - seq_len), t + 1, "eval", split, int(fraud[t]), int(t))

        return _stack_rows(rows, seq_len, soft)

    return tokenize_group


def tokenize_dataset(
    ds,
    seq_len: int,
    train_end: str | None = None,
    val_end: str | None = None,
    normal_keep: float = 1.0,
    max_pretrain_windows: int | None = None,
):
    """Apply the tokenizer over a Ray Dataset grouped by card."""
    return ds.groupby("card_id").map_groups(
        make_tokenize_group_fn(
            seq_len,
            train_end=train_end,
            val_end=val_end,
            normal_keep=normal_keep,
            max_pretrain_windows=max_pretrain_windows,
        ),
        batch_format="numpy",
    )
