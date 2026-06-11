"""Synthetic transaction generator.

Schema mirrors IBM TabFormer (the de-facto public benchmark for transaction
foundation models) so the template transfers cleanly to the real dataset. Each
*card* has a set of STATIC fields (issuer, card type, BIN region, customer home
state) that never change, plus a time-ordered stream of transactions with
DYNAMIC fields (amount, merchant, MCC, timestamp). A small fraction of
transactions are fraudulent, following a few realistic patterns.

This is intentionally generated with numpy/pandas (not Ray) — it stands in for
"your transactions already sitting in S3/Parquet". The interesting distributed
work starts at the tokenizer.
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


def generate_transactions(num_cards: int, seed: int = 42) -> pd.DataFrame:
    """Generate a card-level transaction dataset.

    Returns one row per transaction with both static (card-level) and dynamic
    (per-transaction) columns plus an `is_fraud` label.
    """
    rng = _rng(seed)

    # --- Static per-card profile ---
    card_ids = np.arange(num_cards)
    issuer = rng.choice(ISSUERS, num_cards)
    card_type = rng.choice(CARD_TYPES, num_cards, p=[0.6, 0.35, 0.05])
    bin_region = rng.choice(BIN_REGIONS, num_cards)
    home_state = rng.choice(STATES, num_cards)
    # Spending profile drives amount distribution per card.
    log_amount_mu = rng.normal(3.4, 0.5, num_cards)      # ~ $30 median
    log_amount_sigma = rng.uniform(0.4, 0.9, num_cards)
    activity = rng.gamma(shape=2.0, scale=15.0, size=num_cards)  # txns/card

    rows = []
    base_ts = np.datetime64("2023-01-01T00:00:00")

    for c in card_ids:
        n_txn = max(3, int(activity[c]))
        # Inter-transaction gaps in hours (heavy-tailed).
        gaps = rng.exponential(scale=36.0, size=n_txn).cumsum()
        ts = base_ts + (gaps * 3600).astype("timedelta64[s]")

        cats = rng.choice(MERCHANT_CATEGORIES, n_txn)
        mcc = np.array([rng.choice(MCC_BY_CATEGORY[cat]) for cat in cats])
        merchant_id = rng.integers(0, 8000, n_txn)
        amount = np.exp(rng.normal(log_amount_mu[c], log_amount_sigma[c], n_txn))
        amount = np.round(amount, 2)
        is_fraud = np.zeros(n_txn, dtype=np.int64)

        # --- Plant fraud patterns ---
        if rng.random() < 0.25:  # this card experiences fraud at all
            pattern = rng.integers(0, 4)
            k = rng.integers(1, max(2, n_txn // 8))
            idx = rng.choice(n_txn, size=min(k, n_txn), replace=False)
            if pattern == 0:        # amount anomaly
                amount[idx] = amount[idx] * rng.uniform(8, 25, len(idx))
            elif pattern == 1:      # card testing — many tiny txns, bursty
                amount[idx] = rng.uniform(0.5, 3.0, len(idx))
            elif pattern == 2:      # geographic / off-hours online services
                cats[idx] = "online_services"
                mcc[idx] = 5817
            else:                   # velocity spike (cluster gaps tight)
                amount[idx] = amount[idx] * rng.uniform(3, 8, len(idx))
            is_fraud[idx] = 1

        hour = pd.DatetimeIndex(ts).hour.to_numpy()
        dow = pd.DatetimeIndex(ts).dayofweek.to_numpy()

        for i in range(n_txn):
            rows.append(
                (
                    int(c), issuer[c], card_type[c], bin_region[c], home_state[c],
                    str(ts[i]), float(amount[i]), int(merchant_id[i]),
                    str(cats[i]), int(mcc[i]), int(hour[i]), int(dow[i]),
                    int(is_fraud[i]),
                )
            )

    df = pd.DataFrame(
        rows,
        columns=[
            # static (card-level)
            "card_id", "issuer", "card_type", "bin_region", "home_state",
            # dynamic (per-transaction)
            "timestamp", "amount", "merchant_id", "merchant_category", "mcc",
            "hour", "day_of_week",
            # label
            "is_fraud",
        ],
    )
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    return df


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
