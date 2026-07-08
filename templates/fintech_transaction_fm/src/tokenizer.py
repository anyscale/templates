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
    ERROR_CATEGORIES,
    ISSUERS,
    MERCHANT_CATEGORIES,
)
from .merchant_vocab import _top_lookup, merchant_to_id, merchant_vocab_rows

SPLIT_FIELDS = True  # False -> NVIDIA-style flat tokenization (extension point)
AMOUNT_MODE = "hard"  # "hard" (default, verified) | "soft" (blend adjacent bins)

# Reserved token ids (shared across every field table). OOV absorbs categorical
# values outside the fixed vocab — real-world data always has some.
PAD = 0
MASK = 1
OOV = 2
_RESERVED = 3

# One position per transaction makes long histories cheap: `full`'s 512
# positions cover 512 transactions — vs ~315 in the NVIDIA blueprint's entire
# 4096-token context (~12 tokens per txn). Going past 512 on T4s wants
# flash-attention (O(S^2) buffers) — left as the documented extension.
# Derived from configs/<scale>.yaml; kept here because the README imports it.
def _seq_len_by_scale() -> dict:
    from .scale_config import load_scales

    return {name: cfg["tokenize"]["seq_len"] for name, cfg in load_scales().items()}


SEQ_LEN_BY_SCALE = _seq_len_by_scale()

# --- Deterministic field vocabularies (no data scan needed) ---
N_AMOUNT_BUCKETS = 16
N_MERCHANT_BUCKETS = 2000

# RUN-2 (TEARDOWN.md): exact MCC vocabulary. TabFormer's MCC set is CLOSED
# (109 codes, list from NVIDIA's blueprint tokenizer, Apache-2.0) — the old
# `mcc % 128` hash guaranteed collisions for zero benefit. Unknown codes -> OOV.
KNOWN_MCCS = [
    -1, 1711, 3000, 3001, 3005, 3006, 3007, 3008, 3009, 3058, 3066,
    3075, 3132, 3144, 3174, 3256, 3260, 3359, 3387, 3389, 3390, 3393,
    3395, 3405, 3504, 3509, 3596, 3640, 3684, 3722, 3730, 3771, 3775,
    3780, 4111, 4112, 4121, 4131, 4214, 4411, 4511, 4722, 4784, 4814,
    4829, 4899, 4900, 5045, 5094, 5192, 5193, 5211, 5251, 5261, 5300,
    5310, 5311, 5411, 5499, 5533, 5541, 5621, 5651, 5655, 5661, 5712,
    5719, 5722, 5732, 5733, 5812, 5813, 5814, 5815, 5816, 5912, 5921,
    5932, 5941, 5942, 5947, 5970, 5977, 6300, 7011, 7210, 7230, 7276,
    7349, 7393, 7531, 7538, 7542, 7549, 7801, 7802, 7832, 7922, 7995,
    7996, 8011, 8021, 8041, 8043, 8049, 8062, 8099, 8111, 8931, 9402,
]
_MCC_MAP = {c: i + _RESERVED for i, c in enumerate(KNOWN_MCCS)}
N_ZIP3 = 1000  # per-txn 3-digit zip prefix (RUN-2; parity with NVIDIA's ZIP3)

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
    # Transaction channel ("Use Chip"): a per-event INPUT, known at auth time.
    "channel": _index_map(["swipe", "chip", "online"]),
    # RUN-2 (TEARDOWN.md #2/#7): per-transaction fields the old tokenizer
    # deleted. merchant_state is THE geographic-novelty fraud axis (a per-card
    # modal static cannot express "suddenly transacting in a new state");
    # month adds seasonality; direction restores the refund sign abs() threw
    # away; prev_error is the PREVIOUS txn's network signal (shifted by one,
    # so it is known at auth time — never the target's own outcome).
    "merchant_state": _index_map(US_STATES + ["ONLINE", "FOREIGN"]),
    "month": _index_map(list(range(1, 13))),
    "direction": _index_map(["debit", "credit"]),
    "prev_error": _index_map(ERROR_CATEGORIES),
}

# Order of dynamic fields fed to the model (one embedding table each).
DYNAMIC_FIELDS = [
    "amount_bucket",
    "merchant_bucket",
    "merchant_category",
    "mcc",
    "hour",
    "day_of_week",
    "channel",
    "merchant_state",
    "zip3",
    "month",
    "direction",
    "prev_error",
    "merchant_recency",  # RUN-2b: time since THIS card last hit THIS merchant
]

# merchant_recency vocab: the log-hour gap buckets + a dedicated FIRST_VISIT
# id — "new merchant for this card" is a classic novelty fraud signal the
# per-card time delta cannot express.
MERCHANT_FIRST_VISIT = N_TIME_BUCKETS + _RESERVED

# Continuous input channels (RUN-2b, G2/G3): float columns emitted alongside
# the bucketed tokens; the model consumes them only when its periodic_* flags
# are on, so the schema is a strict superset of the old one.
CONTINUOUS_FIELDS = ["amount_log", "delta_log"]
# RUN-2 (TEARDOWN.md #5): user/card identity REMOVED from the FM input.
# Broadcasting identity into every position made the pooled embedding and
# its PCA identity-dominated (and gave the seq-CL task a trivial shortcut);
# identity is already a raw feature in the downstream fusion, so the FM's
# job is the behavioral/contextual signal identity can't carry.
N_USERS, N_CARDS = 3000, 10  # kept for reference; no longer static fields
STATIC_FIELDS = ["issuer", "card_type", "bin_region", "home_state"]

# Payment-network signals (TREASURE's distinguishing pillar): predicted as
# OUTPUTS, never fed as inputs (they're only known after a txn is processed, so
# using them as features would leak the label). Each is a per-position
# classification target carried as a ``y_<field>`` column.
SIGNAL_VOCAB = {"error": _index_map(ERROR_CATEGORIES)}
SIGNAL_FIELDS = ["error"]


def signal_vocab_sizes() -> dict:
    return {f: len(v) + _RESERVED for f, v in SIGNAL_VOCAB.items()}


def field_vocab_sizes(merchant_vocab: dict | None = None) -> dict:
    """Number of embedding rows per field (including reserved ids).

    With ``merchant_vocab`` (the learned-vocab path), the merchant table is sized
    to top-K + aggregate buckets instead of the fixed hash-bucket count.
    """
    merchant_rows = (
        merchant_vocab_rows(merchant_vocab)
        if merchant_vocab
        else N_MERCHANT_BUCKETS + _RESERVED
    )
    sizes = {
        "amount_bucket": N_AMOUNT_BUCKETS + _RESERVED,
        "merchant_bucket": merchant_rows,
        "mcc": len(KNOWN_MCCS) + _RESERVED,  # exact closed vocab (RUN-2)
        "zip3": N_ZIP3 + _RESERVED,
        "merchant_recency": N_TIME_BUCKETS + 1 + _RESERVED,  # + FIRST_VISIT
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
    # Exact vocabulary (RUN-2): TabFormer's MCC set is closed; unknown -> OOV.
    return np.array([_MCC_MAP.get(int(m), OOV) for m in mcc], dtype=np.int64)


def _delta_to_bucket(delta_hours: np.ndarray) -> np.ndarray:
    logs = np.log10(np.clip(delta_hours, 1e-2, None))
    return np.digitize(logs, _TIME_EDGES) + _RESERVED


def n_time_bucket_rows() -> int:
    """Embedding rows for the time-gap table (incl. reserved + headroom)."""
    return N_TIME_BUCKETS + _RESERVED + 2


def write_vocab(output_path: str, merchant_vocab: dict | None = None) -> None:
    """Persist the model-facing vocab spec.

    ``merchant_vocab`` (built by ``merchant_vocab.build_merchant_vocab`` from a
    distributed frequency scan) switches the merchant field to learned ids and
    flags it for InfoNCE. Without it, the merchant field stays hash-bucketed and
    ``infonce_fields`` is empty — the CI/smoke path is byte-for-byte unchanged.
    """
    os.makedirs(os.path.dirname(output_path.rstrip("/")), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "split_fields": SPLIT_FIELDS,
                "dynamic_fields": DYNAMIC_FIELDS,
                "static_fields": STATIC_FIELDS,
                "signal_fields": SIGNAL_FIELDS,
                "field_vocab_sizes": field_vocab_sizes(merchant_vocab),
                "signal_vocab_sizes": signal_vocab_sizes(),
                "time_aware": TIME_AWARE,
                "n_time_buckets": n_time_bucket_rows(),
                "amount_mode": AMOUNT_MODE,
                "continuous_fields": CONTINUOUS_FIELDS,
                "merchant_vocab_mode": "learned" if merchant_vocab else "hashed",
                "merchant_vocab": merchant_vocab,
                # InfoNCE only when the merchant vocab is genuinely large; the
                # hash-bucketed path stays full-softmax cross-entropy.
                "infonce_fields": ["merchant_bucket"] if merchant_vocab else [],
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
        "row_id": np.zeros(0, np.int64),
        "attention_mask": np.zeros((0, seq_len), np.int32),
        "time_bucket": np.zeros((0, seq_len), np.int32),
        "d_amount_log": np.zeros((0, seq_len), np.float32),
        "d_delta_log": np.zeros((0, seq_len), np.float32),
        "y_error": np.zeros((0, seq_len), np.int32),
        "raw_amount": np.zeros(0, np.float32),
        "raw_hour": np.zeros(0, np.int64),
        "raw_dow": np.zeros(0, np.int64),
        "raw_mcc": np.zeros(0, np.int64),
        "raw_ts": np.zeros(0, np.int64),
    }
    for f in STATIC_FIELDS:
        out[f"s_{f}"] = np.zeros(0, np.int64)
    for f in DYNAMIC_FIELDS:
        out[f"d_{f}"] = np.zeros((0, seq_len), np.int32)
    if soft:
        out["d_amount_frac"] = np.zeros((0, seq_len), np.float32)
    return out


# Columns only the eval samples need — dropped from the pretrain windows
# before they reach the trainer (string columns break torch batch conversion).
PRETRAIN_DROP = [
    "kind", "split", "label", "weight", "row_id",
    "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts",
]


def eval_normal_keep(splits: dict, target_eval_samples: int) -> float:
    """Keep-probability for normal txns in the eval set.

    Keeps enough normals to hit the eval-size target, never fewer than 4x the
    fraud count (importance weights undo the downsampling in the metrics).
    """
    n_txn = splits["n_transactions"]
    n_fraud = splits["fraud_rate"] * n_txn
    normals_target = max(target_eval_samples - n_fraud, 4 * n_fraud)
    return float(min(1.0, normals_target / max(n_txn - n_fraud, 1.0)))


def make_tokenize_group_fn(
    seq_len: int,
    train_end: str | None = None,
    val_end: str | None = None,
    normal_keep: float = 1.0,
    holdout_keep: float | None = None,
    max_pretrain_windows: int | None = None,
    emit: str = "both",
    merchant_vocab: dict | None = None,
    eval_targets: dict | None = None,
):
    """Build a ``map_groups`` UDF turning one card's transactions into padded,
    per-field token sequences (pretrain windows + per-transaction eval samples).

    A closure (not a callable class) because Ray Data's ``map_groups`` keys off
    ``fn.__name__``. The function is stateless — safe at high concurrency; the
    eval-sample downsampling RNG is seeded per card, so output is deterministic.

    ``emit`` ("both" | "pretrain" | "eval") skips computing the other kind's
    windows — a pure compute optimization for single-consumer pipelines. Kept
    rows are identical to a "both" run filtered on ``kind`` (the one-row-per-
    card guard may still emit the other kind, so consumers filter regardless).

    ``merchant_vocab`` (learned-vocab path) maps merchant ids to top-K/aggregate
    embedding ids; the small top-K lookup is built once and captured. Without it,
    merchants are hash-bucketed (the default).

    ``eval_targets`` (card_id -> array of target timestamps, epoch seconds)
    switches eval-window selection to EXACTLY those transactions — the
    NVIDIA-protocol benchmark rows written by stage 01 — with weight 1.0 (no
    downsampling to undo). ``normal_keep``/``holdout_keep`` are ignored then.
    """
    t_train = np.datetime64(train_end, "s") if train_end else None
    t_val = np.datetime64(val_end, "s") if val_end else None
    # Keep-probability for normals in the val/test periods; None -> same as
    # train. 1.0 = score every holdout transaction (exact full-data metrics).
    holdout_keep = normal_keep if holdout_keep is None else holdout_keep
    soft = AMOUNT_MODE == "soft"
    merch_lookup = _top_lookup(merchant_vocab) if merchant_vocab else None

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
        ch_v = DYNAMIC_CATEGORICAL_VOCAB["channel"]
        er_v = SIGNAL_VOCAB["error"]
        # Tokenize the full sorted history once; windows below just slice it.
        merch_ids = np.asarray(group["merchant_id"])[order]
        merchant_col = (
            merchant_to_id(merch_ids, merchant_vocab, merch_lookup)
            if merchant_vocab
            else _merchant_to_bucket(merch_ids)
        )
        full = {
            "amount_bucket": _amount_to_bucket(amounts),
            "merchant_bucket": merchant_col,
            "merchant_category": np.array(
                [mc_v.get(c, OOV) for c in np.asarray(group["merchant_category"])[order]],
                dtype=np.int64,
            ),
            "mcc": _mcc_to_bucket(mccs),
            "hour": np.array([hr_v.get(int(h), OOV) for h in hours], dtype=np.int64),
            "day_of_week": np.array([dw_v.get(int(d), OOV) for d in dows], dtype=np.int64),
            "channel": np.array(
                [ch_v.get(str(c), OOV) for c in np.asarray(group["channel"])[order]],
                dtype=np.int64,
            ),
        }
        # Network-signal target (output-only): tokenized per position, never in `full`.
        error_ids = np.array(
            [er_v.get(str(e), OOV) for e in np.asarray(group["error"])[order]], dtype=np.int64
        )
        # RUN-2 per-transaction fields (TEARDOWN.md #2/#7). Synthetic data has
        # no geo columns -> constant OOV (schema stays fixed across sources).
        ms_v = DYNAMIC_CATEGORICAL_VOCAB["merchant_state"]
        if "merchant_state_raw" in group:
            st = np.asarray(group["merchant_state_raw"]).astype(str)[order]
            full["merchant_state"] = np.array(
                [ms_v.get(s if len(s) == 2 else ("ONLINE" if s == "" else "FOREIGN"), OOV)
                 for s in st],
                dtype=np.int64,
            )
        else:
            full["merchant_state"] = np.full(n, OOV, dtype=np.int64)
        if "zip" in group:
            z = np.asarray(group["zip"], dtype=np.float64)[order]
            zip3 = np.where(
                np.isnan(z), OOV,
                np.clip(np.nan_to_num(z) // 100, 0, N_ZIP3 - 1) + _RESERVED,
            ).astype(np.int64)
            full["zip3"] = zip3
        else:
            full["zip3"] = np.full(n, OOV, dtype=np.int64)
        mo_v = DYNAMIC_CATEGORICAL_VOCAB["month"]
        months = (ts.astype("datetime64[M]").astype(np.int64) % 12) + 1
        full["month"] = np.array([mo_v[int(m)] for m in months], dtype=np.int64)
        dir_v = DYNAMIC_CATEGORICAL_VOCAB["direction"]
        full["direction"] = np.where(
            np.asarray(group["amount"], dtype=np.float64)[order] < 0,
            dir_v["credit"], dir_v["debit"],
        ).astype(np.int64)
        # Previous txn's network signal: shifted one position, so it is known
        # at auth time — the target's own outcome never enters the input.
        full["prev_error"] = np.concatenate([[np.int64(OOV)], error_ids[:-1]])
        # RUN-2b: per-merchant recency — hours since THIS card last hit THIS
        # merchant (log-bucketed; dedicated FIRST_VISIT id for new merchants).
        # Vectorized: sort by (merchant, time), diff within merchant runs.
        mr = np.full(n, MERCHANT_FIRST_VISIT, dtype=np.int64)
        ts_i = ts.astype(np.int64)
        o2 = np.lexsort((ts_i, merch_ids))
        same = merch_ids[o2][1:] == merch_ids[o2][:-1]
        gaps_h = (ts_i[o2][1:] - ts_i[o2][:-1]) / 3600.0
        repeat_idx = o2[1:][same]
        mr[repeat_idx] = _delta_to_bucket(gaps_h[same])
        full["merchant_recency"] = mr
        # RUN-2b continuous channels (G2/G3): signed log-amount and the
        # log-hour gap as floats, alongside their bucketed tokens.
        amount_log_full = (np.sign(amounts) * np.log1p(np.abs(amounts))).astype(np.float32)
        deltas_full = np.zeros(n, dtype=np.float64)
        if n > 1:
            deltas_full[1:] = (ts_i[1:] - ts_i[:-1]) / 3600.0
        delta_log_full = np.log10(np.clip(deltas_full, 1e-2, None)).astype(np.float32)
        if soft:
            soft_lo, soft_frac = _amount_to_soft(amounts)
        # String-categorical statics via their vocab. RUN-2: user/card identity
        # is NOT an FM input anymore (TEARDOWN.md #5) — it dominated the pooled
        # embedding while the downstream fusion already carries it raw.
        statics = {f: int(STATIC_VOCAB[f].get(group[f][0], OOV)) for f in STATIC_VOCAB}

        rows = []

        def emit_row(lo: int, hi: int, kind: str, split: str, label: int, target, weight: float = 1.0):
            L = hi - lo
            pad = seq_len - L
            r = {
                "card_id": card_id, "length": L, "kind": kind, "split": split,
                "label": label, "weight": np.float64(weight),
                # Unique per eval row (card_id, target index) — the id column
                # Ray Data job-level checkpointing keys on (see src/embed.py).
                "row_id": card_id * 1_000_000_000 + (int(target) if target is not None else hi),
            }
            # Sequence columns are int32: vocabularies are tiny, and at long
            # seq_len the eval set is millions of windows — int64 doubles GBs
            # of intermediates for nothing. The model casts to long on input.
            for f, arr in full.items():
                r[f"d_{f}"] = np.concatenate(
                    [np.full(pad, PAD, np.int32), arr[lo:hi].astype(np.int32)]
                )
            # Network-signal target column (PAD at padded positions; supervised
            # at every valid position, masked or not — it's never an input).
            r["y_error"] = np.concatenate(
                [np.full(pad, PAD, np.int32), error_ids[lo:hi].astype(np.int32)]
            )
            if soft:
                r["d_amount_bucket"] = np.concatenate(
                    [np.full(pad, PAD, np.int32), soft_lo[lo:hi].astype(np.int32)]
                )
                r["d_amount_frac"] = np.concatenate(
                    [np.zeros(pad, np.float32), soft_frac[lo:hi]]
                ).astype(np.float32)
            for f, v in statics.items():
                r[f"s_{f}"] = v
            r["attention_mask"] = np.concatenate(
                [np.zeros(pad, np.int32), np.ones(L, np.int32)]
            )
            # Continuous channels (G2/G3), PAD -> 0.0.
            r["d_amount_log"] = np.concatenate(
                [np.zeros(pad, np.float32), amount_log_full[lo:hi]]
            ).astype(np.float32)
            r["d_delta_log"] = np.concatenate(
                [np.zeros(pad, np.float32), delta_log_full[lo:hi]]
            ).astype(np.float32)
            # Inter-transaction gap, recomputed within the window (first gap 0).
            deltas = np.zeros(L, dtype=np.float64)
            if L > 1:
                deltas[1:] = (ts[lo + 1 : hi] - ts[lo : hi - 1]) / np.timedelta64(1, "h")
            r["time_bucket"] = np.concatenate(
                [np.full(pad, PAD, np.int32), _delta_to_bucket(deltas).astype(np.int32)]
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

        # --- Pretraining windows: train-period only, newest first. RUN-2:
        # stride seq_len//2 (50% overlap) — frozen non-overlapping boundaries
        # showed the model the same windows every epoch (TEARDOWN.md #6);
        # overlap doubles effective samples for free.
        n_train = n if t_train is None else int(np.searchsorted(ts, t_train))
        stride = max(1, seq_len // 2)
        if emit in ("both", "pretrain"):
            hi, made = n_train, 0
            while hi >= 2 and (max_pretrain_windows is None or made < max_pretrain_windows):
                lo = max(0, hi - seq_len)
                emit_row(lo, hi, "pretrain", "train", 0, None)
                made += 1
                hi = hi - stride if lo > 0 else 0

        # --- Eval samples: window ends at the target transaction; label = its
        # is_fraud; split by the target's timestamp.
        if eval_targets is not None:
            # Benchmark mode: exactly the stage-01 sampled rows, weight 1.0.
            tgt = eval_targets.get(card_id)
            keep_p = np.ones(n)  # no downsampling -> unit importance weights
            keep = (
                np.isin(ts.astype(np.int64), tgt)
                if tgt is not None
                else np.zeros(n, dtype=bool)
            )
        else:
            # Heuristic mode (synthetic data): all frauds kept, normals
            # downsampled deterministically.
            rng = np.random.default_rng(card_id + 1)
            keep_p = np.full(n, normal_keep)
            if t_train is not None:
                keep_p[int(np.searchsorted(ts, t_train)):] = holdout_keep
            keep = (fraud == 1) | (rng.random(n) < keep_p)
        if emit == "pretrain":
            keep[:] = False  # eval windows not wanted; guard below still applies
        # Guarantee at least one output row per card: empty (0, seq_len)
        # batches break Ray's Arrow tensor conversion, and a card should not
        # silently vanish from the pipeline anyway. In eval-only mode the
        # guard must mirror the combined run exactly — a card whose pretrain
        # windows would have prevented it firing gets a pretrain-kind marker
        # (dropped by the consumer's kind filter) instead of an extra eval
        # row, otherwise the eval set would differ between the two pipelines.
        would_emit_pretrain = n_train >= 2 and (
            max_pretrain_windows is None or max_pretrain_windows > 0
        )
        if not rows and not keep.any():
            if emit == "eval" and would_emit_pretrain:
                emit_row(max(0, n_train - seq_len), n_train, "pretrain", "train", 0, None)
            else:
                keep[n - 1] = True
        for t in np.nonzero(keep)[0]:
            if t_train is None or ts[t] < t_train:
                split = "train"
            elif t_val is None or ts[t] < t_val:
                split = "val"
            else:
                split = "test"
            # Importance weight undoing the normal-downsampling, so downstream
            # metrics estimate full-population (natural-prevalence) values.
            w = 1.0 if fraud[t] == 1 else 1.0 / max(float(keep_p[t]), 1e-9)
            emit_row(max(0, t + 1 - seq_len), t + 1, "eval", split, int(fraud[t]), int(t), weight=w)

        return _stack_rows(rows, seq_len, soft)

    return tokenize_group


def tokenize_dataset(
    ds,
    seq_len: int,
    train_end: str | None = None,
    val_end: str | None = None,
    normal_keep: float = 1.0,
    holdout_keep: float | None = None,
    max_pretrain_windows: int | None = None,
    num_partitions: int | None = None,
    emit: str = "both",
    merchant_vocab: dict | None = None,
    eval_targets: dict | None = None,
):
    """Apply the tokenizer over a Ray Dataset grouped by card.

    ``num_partitions`` sizes the hash shuffle; Ray's default (200) is tuned for
    much larger clusters/datasets and produces hundreds of tiny output files at
    demo scales. ``emit`` (see ``make_tokenize_group_fn``) skips the other
    kind's window computation when the pipeline only consumes one.
    ``merchant_vocab`` switches the merchant field to learned ids.
    ``eval_targets`` selects benchmark-exact eval windows (see above).
    """
    return ds.groupby("card_id", num_partitions=num_partitions).map_groups(
        make_tokenize_group_fn(
            seq_len,
            train_end=train_end,
            val_end=val_end,
            normal_keep=normal_keep,
            holdout_keep=holdout_keep,
            max_pretrain_windows=max_pretrain_windows,
            emit=emit,
            merchant_vocab=merchant_vocab,
            eval_targets=eval_targets,
        ),
        batch_format="numpy",
    )
