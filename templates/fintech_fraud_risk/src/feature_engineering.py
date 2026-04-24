"""
Feature engineering for fraud detection pipeline.
Implements all 21 features as batch functions for Ray Data map_batches.
"""
import numpy as np
import pandas as pd

MERCHANT_CATEGORY_MAP = {
    "grocery": 0, "electronics": 1, "travel": 2, "restaurant": 3,
    "gas": 4, "online": 5, "ATM": 6,
}
CARD_TYPE_MAP = {"visa": 0, "mastercard": 1, "amex": 2}
DEVICE_TYPE_MAP = {"mobile": 0, "desktop": 1, "POS": 2, "ATM": 3}
COUNTRY_RISK = {
    "US": 0.1, "UK": 0.12, "CA": 0.11, "DE": 0.13, "FR": 0.14,
    "JP": 0.08, "BR": 0.25, "MX": 0.22, "IN": 0.20, "AU": 0.09, "NG": 0.35,
}

FEATURE_COLUMNS = [
    "log_amount", "is_round_amount", "hour_of_day", "day_of_week", "is_off_hours",
    "user_avg_amount", "user_std_amount", "amount_zscore",
    "user_txn_count_1h", "user_txn_count_24h",
    "user_unique_merchants_7d", "user_unique_countries_7d",
    "merchant_fraud_rate", "merchant_avg_amount",
    "merchant_category_encoded", "card_type_encoded", "device_type_encoded",
    "country_risk_score", "is_international_flag",
    "amount_x_velocity", "zscore_x_international",
]


def join_user_features(batch: dict, user_lookup: pd.DataFrame) -> dict:
    """Join pre-computed user aggregates onto transaction batch via broadcast lookup."""
    user_ids = batch["user_id"]
    if isinstance(user_ids, np.ndarray):
        user_ids_list = user_ids.tolist()
    else:
        user_ids_list = list(user_ids)

    lookup_indexed = user_lookup.set_index("user_id")
    user_cols = ["user_avg_amount", "user_std_amount", "user_txn_count_1h",
                 "user_txn_count_24h", "user_unique_merchants_7d", "user_unique_countries_7d"]

    matched = lookup_indexed.reindex(user_ids_list)
    for col in user_cols:
        vals = matched[col].values.astype(np.float64)
        batch[col] = np.nan_to_num(vals, nan=0.0)

    return batch


def join_merchant_features(batch: dict, merchant_lookup: pd.DataFrame) -> dict:
    """Join pre-computed merchant aggregates onto transaction batch."""
    merchant_ids = batch["merchant_id"]
    if isinstance(merchant_ids, np.ndarray):
        merchant_ids_list = merchant_ids.tolist()
    else:
        merchant_ids_list = list(merchant_ids)

    lookup_indexed = merchant_lookup.set_index("merchant_id")
    matched = lookup_indexed.reindex(merchant_ids_list)

    batch["merchant_fraud_rate"] = np.nan_to_num(matched["merchant_fraud_rate"].values.astype(np.float64), nan=0.0)
    batch["merchant_avg_amount"] = np.nan_to_num(matched["merchant_avg_amount"].values.astype(np.float64), nan=0.0)

    return batch


def compute_features(batch: dict) -> dict:
    """Compute all 21 engineered features from a transaction batch.

    Expects user and merchant aggregate columns to already be joined.
    """
    amount = batch["amount"].astype(np.float64)
    timestamp = batch["timestamp"]

    # Convert timestamps to pandas for extraction
    if not isinstance(timestamp, pd.Series):
        timestamp = pd.to_datetime(timestamp)
    else:
        timestamp = pd.to_datetime(timestamp)

    # --- Transaction-level features ---
    batch["log_amount"] = np.log1p(amount)
    batch["is_round_amount"] = (np.mod(amount, 1.0) < 0.01).astype(np.float64)
    batch["hour_of_day"] = timestamp.hour.values.astype(np.float64) if hasattr(timestamp.hour, 'values') else np.array([t.hour for t in timestamp], dtype=np.float64)
    batch["day_of_week"] = timestamp.dayofweek.values.astype(np.float64) if hasattr(timestamp.dayofweek, 'values') else np.array([t.dayofweek for t in timestamp], dtype=np.float64)
    hours = batch["hour_of_day"]
    batch["is_off_hours"] = ((hours >= 1) & (hours <= 5)).astype(np.float64)

    # --- User-level features (already joined) ---
    user_avg = batch["user_avg_amount"].astype(np.float64)
    user_std = np.maximum(batch["user_std_amount"].astype(np.float64), 1e-6)
    batch["amount_zscore"] = (amount - user_avg) / user_std

    # --- Categorical encodings ---
    mc = batch["merchant_category"]
    if isinstance(mc, np.ndarray):
        batch["merchant_category_encoded"] = np.array([MERCHANT_CATEGORY_MAP.get(str(x), 0) for x in mc], dtype=np.float64)
    else:
        batch["merchant_category_encoded"] = np.array([MERCHANT_CATEGORY_MAP.get(x, 0) for x in mc], dtype=np.float64)

    ct = batch["card_type"]
    if isinstance(ct, np.ndarray):
        batch["card_type_encoded"] = np.array([CARD_TYPE_MAP.get(str(x), 0) for x in ct], dtype=np.float64)
    else:
        batch["card_type_encoded"] = np.array([CARD_TYPE_MAP.get(x, 0) for x in ct], dtype=np.float64)

    dt = batch["device_type"]
    if isinstance(dt, np.ndarray):
        batch["device_type_encoded"] = np.array([DEVICE_TYPE_MAP.get(str(x), 0) for x in dt], dtype=np.float64)
    else:
        batch["device_type_encoded"] = np.array([DEVICE_TYPE_MAP.get(x, 0) for x in dt], dtype=np.float64)

    country = batch["country"]
    if isinstance(country, np.ndarray):
        batch["country_risk_score"] = np.array([COUNTRY_RISK.get(str(x), 0.15) for x in country], dtype=np.float64)
    else:
        batch["country_risk_score"] = np.array([COUNTRY_RISK.get(x, 0.15) for x in country], dtype=np.float64)

    is_intl = batch["is_international"]
    if isinstance(is_intl, np.ndarray):
        batch["is_international_flag"] = is_intl.astype(np.float64)
    else:
        batch["is_international_flag"] = np.array(is_intl, dtype=np.float64)

    # --- Interaction features ---
    velocity = batch["user_txn_count_1h"].astype(np.float64)
    batch["amount_x_velocity"] = amount * velocity
    batch["zscore_x_international"] = batch["amount_zscore"] * batch["is_international_flag"]

    return batch
