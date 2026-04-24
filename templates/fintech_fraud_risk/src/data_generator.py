"""
Synthetic financial transaction generator with realistic fraud patterns.
Writes Parquet under the demo base dir (see ``src.paths.get_demo_base_dir``).
"""
import argparse
import os
import uuid

import numpy as np
import pandas as pd

SCALE_MAP = {"small": 100_000, "medium": 1_000_000, "large": 10_000_000}

MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "restaurant", "gas", "online", "ATM"]
CARD_TYPES = ["visa", "mastercard", "amex"]
COUNTRIES = ["US", "US", "US", "US", "UK", "CA", "DE", "FR", "JP", "BR", "MX", "IN", "AU", "NG"]  # weighted toward US
DEVICE_TYPES = ["mobile", "desktop", "POS", "ATM"]

# Country risk scores (higher = riskier)
COUNTRY_RISK = {
    "US": 0.1, "UK": 0.12, "CA": 0.11, "DE": 0.13, "FR": 0.14,
    "JP": 0.08, "BR": 0.25, "MX": 0.22, "IN": 0.20, "AU": 0.09, "NG": 0.35,
}


def generate_transactions(
    num_transactions: int = 1_000_000,
    num_users: int = 50_000,
    num_merchants: int = 10_000,
    fraud_rate: float = 0.02,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic transaction data with injected fraud patterns.

    Returns (transactions_df, user_aggregates_df, merchant_aggregates_df).
    """
    rng = np.random.default_rng(seed)

    # --- Base transactions (legitimate) ---
    user_ids = [f"user_{i:06d}" for i in range(num_users)]
    merchant_ids = [f"merch_{i:05d}" for i in range(num_merchants)]

    # Assign each user a "home country" and typical spending pattern
    user_home_country = rng.choice(COUNTRIES, size=num_users)
    user_mean_amount = rng.lognormal(mean=3.5, sigma=1.0, size=num_users).clip(10, 5000)
    user_std_amount = user_mean_amount * rng.uniform(0.3, 0.8, size=num_users)

    # Generate transaction-level fields
    user_idx = rng.integers(0, num_users, size=num_transactions)
    txn_user_ids = np.array(user_ids)[user_idx]
    txn_merchant_ids = rng.choice(merchant_ids, size=num_transactions)
    txn_categories = rng.choice(MERCHANT_CATEGORIES, size=num_transactions)
    txn_card_types = rng.choice(CARD_TYPES, size=num_transactions, p=[0.5, 0.35, 0.15])
    txn_device_types = rng.choice(DEVICE_TYPES, size=num_transactions, p=[0.35, 0.25, 0.30, 0.10])

    # Amounts drawn from each user's distribution
    txn_amounts = np.abs(rng.normal(user_mean_amount[user_idx], user_std_amount[user_idx]))
    txn_amounts = txn_amounts.clip(0.01, 50_000).round(2)

    # Timestamps: spread over 90 days, mostly business hours
    base_ts = pd.Timestamp("2024-01-01")
    hours_offset = rng.exponential(scale=24 * 45, size=num_transactions)  # cluster toward recent
    hours_offset = hours_offset.clip(0, 24 * 90)
    timestamps = (pd.to_datetime(base_ts) + pd.to_timedelta(hours_offset, unit="h")).to_series().reset_index(drop=True)

    # Countries: mostly user's home country
    txn_countries = np.where(
        rng.random(num_transactions) < 0.92,
        user_home_country[user_idx],
        rng.choice(COUNTRIES, size=num_transactions),
    )
    is_international = txn_countries != user_home_country[user_idx]

    # --- Inject fraud patterns ---
    num_fraud = int(num_transactions * fraud_rate)
    fraud_indices = rng.choice(num_transactions, size=num_fraud, replace=False)
    is_fraud = np.zeros(num_transactions, dtype=bool)
    is_fraud[fraud_indices] = True

    # Split fraud indices by pattern
    rng.shuffle(fraud_indices)
    splits = np.cumsum([0.30, 0.25, 0.20, 0.15])
    split_points = (splits * num_fraud).astype(int)
    velocity_idx = fraud_indices[: split_points[0]]
    amount_idx = fraud_indices[split_points[0] : split_points[1]]
    geo_idx = fraud_indices[split_points[1] : split_points[2]]
    offhours_idx = fraud_indices[split_points[2] : split_points[3]]
    cardtest_idx = fraud_indices[split_points[3] :]

    # Velocity fraud: cluster timestamps (multiple txns within minutes)
    for idx in velocity_idx:
        cluster_base = timestamps[idx]
        nearby = rng.choice(fraud_indices, size=min(5, len(fraud_indices)), replace=False)
        for n in nearby:
            timestamps[n] = cluster_base + pd.Timedelta(minutes=rng.integers(1, 10))

    # Amount anomaly: set amount to >3 std above user mean
    for idx in amount_idx:
        uid = user_idx[idx]
        txn_amounts[idx] = user_mean_amount[uid] + user_std_amount[uid] * rng.uniform(3.5, 8.0)

    # Geographic anomaly: different country from home
    foreign = [c for c in COUNTRIES if c not in ("US",)]
    for idx in geo_idx:
        uid = user_idx[idx]
        home = user_home_country[uid]
        options = [c for c in foreign if c != home]
        txn_countries[idx] = rng.choice(options)
        is_international[idx] = True

    # Off-hours fraud: set time to 1am-5am
    for idx in offhours_idx:
        ts = timestamps[idx]
        new_hour = rng.integers(1, 5)
        timestamps[idx] = ts.replace(hour=int(new_hour), minute=int(rng.integers(0, 59)))

    # Card testing: small amounts
    txn_amounts[cardtest_idx] = rng.uniform(1.0, 5.0, size=len(cardtest_idx)).round(2)

    # Clip amounts
    txn_amounts = txn_amounts.clip(0.01, 50_000).round(2)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(num_transactions)],
        "timestamp": timestamps,
        "user_id": txn_user_ids,
        "amount": txn_amounts,
        "merchant_id": txn_merchant_ids,
        "merchant_category": txn_categories,
        "card_type": txn_card_types,
        "country": txn_countries,
        "device_type": txn_device_types,
        "is_international": is_international,
        "is_fraud": is_fraud,
    })

    # Sort by timestamp for realistic ordering
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Build user aggregates (pre-computed feature store) ---
    user_agg = df.groupby("user_id").agg(
        user_avg_amount=("amount", "mean"),
        user_std_amount=("amount", "std"),
        user_txn_count_24h=("transaction_id", "count"),  # simplified: total txns as proxy
        user_unique_merchants_7d=("merchant_id", "nunique"),
        user_unique_countries_7d=("country", "nunique"),
    ).reset_index()

    # Derive 1h count as fraction of 24h (simplified)
    user_agg["user_txn_count_1h"] = (user_agg["user_txn_count_24h"] / 24).clip(lower=1).astype(int)
    user_agg["user_std_amount"] = user_agg["user_std_amount"].fillna(1.0)

    # --- Build merchant aggregates ---
    merchant_agg = df.groupby("merchant_id").agg(
        merchant_avg_amount=("amount", "mean"),
        merchant_fraud_rate=("is_fraud", "mean"),
    ).reset_index()

    print(f"Generated {len(df):,} transactions ({is_fraud.sum():,} fraud, {fraud_rate*100:.1f}% rate)")
    print(f"  Users: {num_users:,}  |  Merchants: {num_merchants:,}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    return df, user_agg, merchant_agg


def save_dataset(
    output_dir: str,
    num_transactions: int = 1_000_000,
    seed: int = 42,
) -> dict:
    """Generate and save all artifacts to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    txn_path = os.path.join(output_dir, "transactions.parquet")
    user_agg_path = os.path.join(output_dir, "user_aggregates.parquet")
    merchant_agg_path = os.path.join(output_dir, "merchant_aggregates.parquet")

    df, user_agg, merchant_agg = generate_transactions(
        num_transactions=num_transactions, seed=seed,
    )

    df.to_parquet(txn_path, index=False)
    user_agg.to_parquet(user_agg_path, index=False)
    merchant_agg.to_parquet(merchant_agg_path, index=False)

    print(f"  Saved transactions     → {txn_path}")
    print(f"  Saved user aggregates  → {user_agg_path}")
    print(f"  Saved merchant aggregates → {merchant_agg_path}")

    return {
        "transactions": txn_path,
        "user_aggregates": user_agg_path,
        "merchant_aggregates": merchant_agg_path,
        "num_transactions": len(df),
    }


if __name__ == "__main__":
    from src.paths import get_demo_base_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(get_demo_base_dir(), "raw", "medium"),
    )
    args = parser.parse_args()
    save_dataset(args.output_dir, num_transactions=SCALE_MAP[args.scale])
