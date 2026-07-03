"""NVIDIA-blueprint-faithful flat tokenizer for causal-LM pretraining.

Replaces the field-split scheme with NVIDIA's: every transaction becomes **12
semantic tokens** in ONE shared vocabulary, and a card's history is a flat token
stream

    <bos> AMT_3 MERCH_1498 CAT_retail MCC_57 HOUR_09 DOW_2 MONTH_01 CARD_1
          CHIP_swipe ZIP3_802 STATE_CO CUST_42 <sep> ...next txn... <eos>

padded to ``seq_length`` TOKENS. A Llama decoder is pretrained on this by
next-token prediction (see src/model.py). The vocab is fully deterministic
(fixed integer ranges + hashing for open-ended fields), so tokenization stays a
stateless, embarrassingly-parallel ``map_groups`` — no global fit, exactly like
the field-split tokenizer it replaces.

Per-transaction field order (matches NVIDIA's financial_pipeline.py):
    AMT, MERCH, CAT, MCC, HOUR, DOW, MONTH, CARD, CHIP, ZIP3, STATE, CUST
"""

from __future__ import annotations

import json
import os
import zlib

import numpy as np

from .generate_data import MERCHANT_CATEGORIES

# --- special tokens (ids match NVIDIA's config.json) ---
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, SEP_ID, UNK_ID = 0, 1, 2, 3, 4

# --- field vocabularies ---
# Amount: NVIDIA "fixed" strategy — 7 bins from these edges.
_AMOUNT_EDGES = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]  # -> bins 0..6
MERCH_HASH = 2000       # merchant hashing trick (NVIDIA merchant_hash_size)
MCC_HASH = 128          # mcc hashing (open-ended code set)
CATEGORIES = list(MERCHANT_CATEGORIES) + ["other"]
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "ONLINE", "FOREIGN", "UNKNOWN",
]
CHIPS = ["swipe", "chip", "online"]

# (field name, local vocab size) in per-transaction emission order.
FIELD_SPECS = [
    ("AMT", 7),
    ("MERCH", MERCH_HASH),
    ("CAT", len(CATEGORIES)),
    ("MCC", MCC_HASH),
    ("HOUR", 24),
    ("DOW", 7),
    ("MONTH", 12),
    ("CARD", 10),
    ("CHIP", len(CHIPS)),
    ("ZIP3", 1000),
    ("STATE", len(STATES)),
    ("CUST", 3000),
]
TOKENS_PER_TXN = len(FIELD_SPECS)

# Global-vocab offsets: specials first, then each field's block.
OFFSETS: dict = {}
_o = len(SPECIALS)
for _name, _size in FIELD_SPECS:
    OFFSETS[_name] = _o
    _o += _size
VOCAB_SIZE = _o

_CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
_STATE_IDX = {s: i for i, s in enumerate(STATES)}
_CHIP_IDX = {c: i for i, c in enumerate(CHIPS)}


def _g(field: str, local: int) -> int:
    """Global token id for a field's local index."""
    return OFFSETS[field] + local


def _hash(value, n: int) -> int:
    return int(zlib.crc32(str(value).encode()) % n)


def encode_transactions(
    amount, merchant_id, merchant_category, mcc, hour, dow, month,
    card_id, use_chip, zip_, merchant_state,
) -> np.ndarray:
    """Vectorized: arrays of per-transaction fields -> (N, 12) global token ids,
    in the fixed field order. Missing/out-of-vocab categoricals -> <unk>."""
    n = len(amount)
    out = np.empty((n, TOKENS_PER_TXN), dtype=np.int64)
    amt = np.abs(np.asarray(amount, dtype=np.float64))
    out[:, 0] = OFFSETS["AMT"] + np.clip(np.digitize(amt, _AMOUNT_EDGES), 0, 6)
    out[:, 1] = [_g("MERCH", _hash(m, MERCH_HASH)) for m in merchant_id]
    out[:, 2] = [_g("CAT", _CAT_IDX[c]) if c in _CAT_IDX else UNK_ID for c in merchant_category]
    out[:, 3] = [_g("MCC", _hash(m, MCC_HASH)) for m in mcc]
    out[:, 4] = OFFSETS["HOUR"] + np.clip(np.asarray(hour, dtype=np.int64), 0, 23)
    out[:, 5] = OFFSETS["DOW"] + np.clip(np.asarray(dow, dtype=np.int64), 0, 6)
    out[:, 6] = OFFSETS["MONTH"] + np.clip(np.asarray(month, dtype=np.int64) - 1, 0, 11)
    cid = np.asarray(card_id, dtype=np.int64)
    out[:, 7] = OFFSETS["CARD"] + np.clip(cid % 100, 0, 9)
    out[:, 8] = [_g("CHIP", _CHIP_IDX[str(c).lower()]) if str(c).lower() in _CHIP_IDX else UNK_ID
                 for c in use_chip]
    z = np.asarray(zip_, dtype=np.float64)
    z = np.where(np.isnan(z) | (z < 0), 0, z).astype(np.int64) // 100
    out[:, 9] = OFFSETS["ZIP3"] + np.clip(z, 0, 999)
    out[:, 10] = [_g("STATE", _STATE_IDX[s]) if s in _STATE_IDX else UNK_ID for s in merchant_state]
    out[:, 11] = OFFSETS["CUST"] + np.clip(cid // 100, 0, 2999)
    return out


def build_sequence(txn_tokens: np.ndarray, seq_length: int) -> tuple:
    """Flatten per-txn token rows into one <bos> … <sep> … <eos> sequence,
    right-padded/truncated to ``seq_length``. Returns (input_ids, attention_mask)
    int32 arrays of length ``seq_length``. Right-padded (last real token is the
    most recent transaction) so last-token pooling reads the latest state."""
    toks = [BOS_ID]
    for i, row in enumerate(txn_tokens):
        if i:
            toks.append(SEP_ID)
        toks.extend(int(t) for t in row)
    toks.append(EOS_ID)
    toks = toks[:seq_length]
    ids = np.full(seq_length, PAD_ID, dtype=np.int32)
    ids[: len(toks)] = toks
    mask = np.zeros(seq_length, dtype=np.int32)
    mask[: len(toks)] = 1
    return ids, mask


def write_vocab(output_path: str, seq_length: int) -> None:
    """Persist the flat-tokenizer spec the model + embed stages read."""
    os.makedirs(os.path.dirname(output_path.rstrip("/")), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "scheme": "flat_ntp",           # vs the old "split_fields"
                "vocab_size": VOCAB_SIZE,
                "tokens_per_txn": TOKENS_PER_TXN,
                "field_order": [n for n, _ in FIELD_SPECS],
                "seq_length": seq_length,
                "pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "sep": SEP_ID, "unk": UNK_ID,
            },
            f,
            indent=2,
        )


# --- Ray map_groups wrapper: pretrain sequences + eval windows per card ---
# Columns eval rows carry beyond the model inputs; pretrain rows fill defaults
# then drop these (same pattern as the old field-split tokenizer).
_RAW_COLS = [
    "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts", "raw_use_chip",
    "raw_merchant_state", "raw_merchant_city", "raw_zip", "raw_merchant_id", "raw_card_id",
]
# card_id dropped too: pretrain feeds iter_torch_batches, which tensorizes every
# remaining column — leaving only input_ids/attention_mask keeps it clean.
PRETRAIN_DROP = ["kind", "split", "label", "weight", "card_id"] + _RAW_COLS


def eval_normal_keep(splits: dict, target_eval_samples: int) -> float:
    """Keep-probability for normal txns in the eval set (unchanged from before)."""
    n_txn = splits["n_transactions"]
    n_fraud = splits["fraud_rate"] * n_txn
    normals_target = max(target_eval_samples - n_fraud, 4 * n_fraud)
    return float(min(1.0, normals_target / max(n_txn - n_fraud, 1.0)))


def _raw_features(i, amounts, hours, dows, mccs, ts, use_chips, states, cities, zips, merch, card_id):
    """Raw target-transaction features for the downstream baseline (unchanged schema)."""
    return {
        "raw_amount": np.float32(amounts[i]), "raw_hour": int(hours[i]), "raw_dow": int(dows[i]),
        "raw_mcc": int(mccs[i]), "raw_ts": int(ts[i].astype(np.int64)),
        "raw_use_chip": _CHIP_IDX.get(str(use_chips[i]).lower(), -1),
        "raw_merchant_state": _STATE_IDX.get(states[i], -1),
        "raw_merchant_city": _hash(cities[i], 100000) if str(cities[i]) else -1,
        "raw_zip": (-1 if (zips[i] != zips[i]) else int(zips[i])),
        "raw_merchant_id": int(merch[i]) % 100000, "raw_card_id": int(card_id),
    }


def _empty_raw(card_id):
    d = {c: 0 for c in _RAW_COLS}
    d["raw_amount"] = np.float32(0.0)
    d["raw_card_id"] = int(card_id)
    return d


def make_tokenize_group_fn(seq_len, train_end=None, val_end=None, normal_keep=1.0,
                           holdout_keep=None, max_pretrain_windows=None, emit="both"):
    """Build a ``map_groups`` UDF: one card's transactions -> flat token sequences.

    ``seq_len`` is now in TOKENS. Pretrain = non-overlapping blocks of the card's
    train-period history (causal LM). Eval = one window per target transaction
    (history ending at it) + label/split/weight + raw_* passthrough.
    """
    t_train = np.datetime64(train_end, "s") if train_end else None
    t_val = np.datetime64(val_end, "s") if val_end else None
    holdout_keep = normal_keep if holdout_keep is None else holdout_keep
    max_txns = max(1, (seq_len - 2) // (TOKENS_PER_TXN + 1))

    def tokenize_group(group: dict) -> dict:
        n = len(group["card_id"])
        order = np.argsort(group["timestamp"])
        ts = np.asarray(group["timestamp"])[order].astype("datetime64[s]")
        fraud = np.asarray(group["is_fraud"]).astype(np.int64)[order]
        amounts = np.asarray(group["amount"]).astype(np.float64)[order]
        merch = np.asarray(group["merchant_id"]).astype(np.int64)[order]
        cats = np.asarray(group["merchant_category"])[order]
        mccs = np.asarray(group["mcc"]).astype(np.int64)[order]
        hours = np.asarray(group["hour"]).astype(np.int64)[order]
        dows = np.asarray(group["day_of_week"]).astype(np.int64)[order]
        months = (ts.astype("datetime64[M]").astype(np.int64) % 12) + 1
        card_id = int(group["card_id"][0])
        use_chips = np.asarray(group["use_chip"])[order] if "use_chip" in group else np.array([""] * n)
        states = np.asarray(group["merchant_state"])[order] if "merchant_state" in group else np.array([""] * n)
        cities = np.asarray(group["merchant_city"])[order] if "merchant_city" in group else np.array([""] * n)
        zips = np.asarray(group["zip"]).astype(np.float64)[order] if "zip" in group else np.full(n, np.nan)

        tok = encode_transactions(amounts, merch, cats, mccs, hours, dows, months,
                                  np.full(n, card_id), use_chips, zips, states)
        rows = []

        def emit_row(lo, hi, kind, split, label, target, weight=1.0):
            ids, mask = build_sequence(tok[lo:hi], seq_len)
            r = {"card_id": card_id, "kind": kind, "split": split,
                 "label": int(label), "weight": np.float64(weight),
                 "input_ids": ids, "attention_mask": mask}
            r.update(_raw_features(target, amounts, hours, dows, mccs, ts, use_chips,
                                   states, cities, zips, merch, card_id)
                     if target is not None else _empty_raw(card_id))
            rows.append(r)

        n_train = n if t_train is None else int(np.searchsorted(ts, t_train))

        # Pretrain: non-overlapping blocks of train-period history.
        if emit in ("both", "pretrain"):
            made = 0
            for lo in range(0, n_train, max_txns):
                if max_pretrain_windows is not None and made >= max_pretrain_windows:
                    break
                hi = min(lo + max_txns, n_train)
                if hi - lo >= 1:
                    emit_row(lo, hi, "pretrain", "train", 0, None)
                    made += 1

        # Eval: one window per kept target transaction.
        rng = np.random.default_rng(card_id + 1)
        keep_p = np.full(n, normal_keep)
        if t_train is not None:
            keep_p[int(np.searchsorted(ts, t_train)):] = holdout_keep
        keep = (fraud == 1) | (rng.random(n) < keep_p)
        if emit == "pretrain":
            keep[:] = False
        if not rows and not keep.any():
            keep[n - 1] = True
        if emit in ("both", "eval"):
            for t in np.nonzero(keep)[0]:
                if t_train is None or ts[t] < t_train:
                    split = "train"
                elif t_val is None or ts[t] < t_val:
                    split = "val"
                else:
                    split = "test"
                w = 1.0 if fraud[t] == 1 else 1.0 / max(float(keep_p[t]), 1e-9)
                emit_row(max(0, t + 1 - max_txns), t + 1, "eval", split, int(fraud[t]), int(t), weight=w)

        return _stack(rows, seq_len)

    return tokenize_group


def _stack(rows, seq_len):
    if not rows:
        out = {"card_id": np.zeros(0, np.int64), "kind": np.array([], dtype=object),
               "split": np.array([], dtype=object), "label": np.zeros(0, np.int64),
               "weight": np.zeros(0, np.float64),
               "input_ids": np.zeros((0, seq_len), np.int32),
               "attention_mask": np.zeros((0, seq_len), np.int32)}
        for c in _RAW_COLS:
            out[c] = np.zeros(0, np.float32 if c == "raw_amount" else np.int64)
        return out
    out = {}
    for k, v0 in rows[0].items():
        if isinstance(v0, np.ndarray):
            out[k] = np.stack([r[k] for r in rows])
        elif isinstance(v0, str):
            out[k] = np.array([r[k] for r in rows], dtype=object)
        else:
            out[k] = np.array([r[k] for r in rows])
    return out


def tokenize_dataset(ds, seq_len, train_end=None, val_end=None, normal_keep=1.0,
                     holdout_keep=None, max_pretrain_windows=None, num_partitions=None, emit="both"):
    """Apply the flat tokenizer over a Ray Dataset grouped by card."""
    return ds.groupby("card_id", num_partitions=num_partitions).map_groups(
        make_tokenize_group_fn(
            seq_len, train_end=train_end, val_end=val_end, normal_keep=normal_keep,
            holdout_keep=holdout_keep, max_pretrain_windows=max_pretrain_windows, emit=emit,
        ),
        batch_format="numpy",
    )


def _seq_len_by_scale() -> dict:
    from .scale_config import load_scales

    return {name: cfg["tokenize"]["seq_len"] for name, cfg in load_scales().items()}


SEQ_LEN_BY_SCALE = _seq_len_by_scale()


# --- display helper: token id -> readable string ("AMT_3", "<bos>", ...) ---
_ID_TO_TOKEN = {i: s for i, s in enumerate(SPECIALS)}
for _n, _sz in FIELD_SPECS:
    for _l in range(_sz):
        _ID_TO_TOKEN[OFFSETS[_n] + _l] = f"{_n}_{_l}"


def decode_tokens(ids) -> list:
    """Map token ids back to readable strings (for walkthrough display)."""
    return [_ID_TO_TOKEN.get(int(t), f"?{int(t)}") for t in ids]
