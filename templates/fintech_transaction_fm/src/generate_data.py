"""Synthetic transaction generator.

Schema mirrors IBM TabFormer (the de-facto public benchmark for transaction
foundation models) so the template transfers cleanly to the real dataset. Each
*card* has a set of STATIC fields (issuer, card type, BIN region, customer home
state) that never change, plus a time-ordered stream of transactions with
DYNAMIC fields (amount, merchant, MCC, timestamp). A small fraction of
transactions are fraudulent, following a few realistic patterns.

``generate_transactions`` is plain numpy/pandas — it stands in for "your
transactions already sitting in S3/Parquet" at small scale. For the scaling
deep-dive we also expose ``generate_dataset_distributed``, which maps the same
per-card generator over the cluster with Ray Data: single-node generation OOMs
well before the billions of rows you need to actually stress multi-node, so
generating the data is itself the first thing that has to go distributed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# --- Field vocabularies (kept compact for a fast demo) ---
ISSUERS = [f"ISSUER_{i}" for i in range(12)]
CARD_TYPES = ["credit", "debit", "prepaid"]
BIN_REGIONS = ["US_EAST", "US_WEST", "US_CENTRAL", "EU", "APAC", "LATAM"]
STATES = ["MA", "NY", "CA", "TX", "FL", "IL", "WA", "GA", "CO", "AZ"]

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "travel", "retail",
    "entertainment", "utilities", "healthcare", "online_services",
]
# MCC codes roughly aligned to categories (a coarse, realistic mapping).
MCC_BY_CATEGORY = {
    "grocery": [5411, 5422, 5451],
    "restaurant": [5812, 5814],
    "travel": [3000, 4511, 7011],
    "retail": [5651, 5732, 5999],
    "entertainment": [7832, 7922, 7995],
    "utilities": [4900, 4814],
    "healthcare": [8011, 8062, 5912],
    "online_services": [5817, 5818, 4816],
}

FRAUD_RATE = 0.018


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


_COLUMNS = [
    # static (card-level)
    "card_id", "issuer", "card_type", "bin_region", "home_state",
    # dynamic (per-transaction)
    "timestamp", "amount", "merchant_id", "merchant_category", "mcc",
    "hour", "day_of_week",
    # label
    "is_fraud",
]


def generate_cards(card_ids, seed: int = 42) -> pd.DataFrame:
    """Generate transactions for an arbitrary set of card ids.

    Each card is seeded independently (``[seed, card_id]``), so generation is
    deterministic per card and embarrassingly parallel — ``generate_dataset_
    distributed`` maps this over blocks of ids with Ray Data, and the result
    doesn't depend on how the ids are partitioned.
    """
    base_ts = np.datetime64("2023-01-01T00:00:00")
    chunks: dict = {c: [] for c in _COLUMNS}

    for c in np.asarray(card_ids).tolist():
        rng = np.random.default_rng([int(seed), int(c)])
        issuer = rng.choice(ISSUERS)
        card_type = rng.choice(CARD_TYPES, p=[0.6, 0.35, 0.05])
        bin_region = rng.choice(BIN_REGIONS)
        home_state = rng.choice(STATES)
        log_mu = rng.normal(3.4, 0.5)            # ~ $30 median spend
        log_sigma = rng.uniform(0.4, 0.9)
        n_txn = max(3, int(rng.gamma(shape=2.0, scale=15.0)))   # txns/card

        # Inter-transaction gaps in hours (heavy-tailed).
        gaps = rng.exponential(scale=36.0, size=n_txn).cumsum()
        ts = base_ts + (gaps * 3600).astype("timedelta64[s]")
        cats = rng.choice(MERCHANT_CATEGORIES, n_txn)
        mcc = np.array([rng.choice(MCC_BY_CATEGORY[cat]) for cat in cats])
        merchant_id = rng.integers(0, 8000, n_txn)
        amount = np.round(np.exp(rng.normal(log_mu, log_sigma, n_txn)), 2)
        is_fraud = np.zeros(n_txn, dtype=np.int64)

        # --- Plant fraud patterns ---
        if rng.random() < 0.25:  # this card experiences fraud at all
            pattern = rng.integers(0, 4)
            k = rng.integers(1, max(2, n_txn // 8))
            idx = rng.choice(n_txn, size=min(k, n_txn), replace=False)
            if pattern == 0:        # amount anomaly
                amount[idx] = amount[idx] * rng.uniform(8, 25, len(idx))
            elif pattern == 1:      # card testing — many tiny txns
                amount[idx] = rng.uniform(0.5, 3.0, len(idx))
            elif pattern == 2:      # off-hours online services
                cats[idx] = "online_services"
                mcc[idx] = 5817
            else:                   # velocity spike
                amount[idx] = amount[idx] * rng.uniform(3, 8, len(idx))
            is_fraud[idx] = 1

        tsi = pd.DatetimeIndex(ts)
        chunks["card_id"].append(np.full(n_txn, int(c), np.int64))
        chunks["issuer"].append(np.full(n_txn, issuer, object))
        chunks["card_type"].append(np.full(n_txn, card_type, object))
        chunks["bin_region"].append(np.full(n_txn, bin_region, object))
        chunks["home_state"].append(np.full(n_txn, home_state, object))
        chunks["timestamp"].append(ts.astype("datetime64[s]"))
        chunks["amount"].append(amount.astype(np.float64))
        chunks["merchant_id"].append(merchant_id.astype(np.int64))
        chunks["merchant_category"].append(cats.astype(object))
        chunks["mcc"].append(mcc.astype(np.int64))
        chunks["hour"].append(tsi.hour.to_numpy().astype(np.int64))
        chunks["day_of_week"].append(tsi.dayofweek.to_numpy().astype(np.int64))
        chunks["is_fraud"].append(is_fraud)

    return pd.DataFrame({c: np.concatenate(v) for c, v in chunks.items()})


def generate_transactions(num_cards: int, seed: int = 42) -> pd.DataFrame:
    """Single-node generation of cards ``0..num_cards-1`` (sorted).

    Composes the same per-card generator the distributed path uses, so the two
    produce identical data for the same card ids.
    """
    df = generate_cards(np.arange(num_cards), seed=seed)
    return df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)


def generate_dataset_distributed(
    output_path: str,
    splits_out: str,
    num_cards: int,
    seed: int = 42,
    num_blocks: int = 200,
) -> str:
    """Generate a large dataset with Ray Data and write sharded Parquet.

    Maps the per-card generator over blocks of card ids across the cluster, so
    the data never has to fit in one process. Then derives the temporal split
    from two slim columns. Returns ``output_path``.
    """
    import ray

    from .paths import write_splits_meta

    def _gen(batch: dict) -> dict:
        df = generate_cards(batch["id"], seed=seed)
        return {col: df[col].to_numpy() for col in _COLUMNS}

    (
        ray.data.range(int(num_cards), override_num_blocks=num_blocks)
        .map_batches(_gen, batch_format="numpy")
        .write_parquet(output_path)
    )

    slim = ray.data.read_parquet(output_path, columns=["timestamp", "is_fraud"]).to_pandas()
    write_splits_meta(
        splits_out,
        slim["timestamp"].to_numpy().astype("datetime64[s]"),
        slim["is_fraud"].to_numpy(),
        source="synthetic",
        n_cards=int(num_cards),
    )
    print(
        f"[generate_data] {len(slim):,} transactions / {num_cards:,} cards "
        f"({slim['is_fraud'].mean() * 100:.2f}% fraud) -> {output_path}",
        flush=True,
    )
    return output_path


def save_dataset(output_path: str, num_cards: int, seed: int = 42) -> str:
    """Generate and write the dataset to a single Parquet file."""
    from .paths import write_splits_meta

    os.makedirs(os.path.dirname(output_path.rstrip("/")), exist_ok=True)
    df = generate_transactions(num_cards, seed=seed)
    df.to_parquet(output_path, index=False)
    write_splits_meta(
        os.path.join(os.path.dirname(output_path), "splits.json"),
        df["timestamp"].to_numpy().astype("datetime64[s]"),
        df["is_fraud"].to_numpy(),
        source="synthetic",
        n_cards=num_cards,
    )
    print(
        f"[generate_data] {len(df):,} transactions / {num_cards:,} cards "
        f"({df['is_fraud'].mean() * 100:.2f}% fraud) -> {output_path}"
    )
    return output_path


# Static vs dynamic field declarations — imported by the tokenizer and model.
STATIC_FIELDS = ["issuer", "card_type", "bin_region", "home_state"]
DYNAMIC_CATEGORICAL_FIELDS = ["merchant_category", "mcc", "hour", "day_of_week"]
DYNAMIC_NUMERIC_FIELDS = ["amount"]
