"""
Step 2: Feature-engineer training data and train XGBoost model.

Usage:
  python scripts/02_train_model.py --scale medium
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.feature_engineering import (
    join_user_features, join_merchant_features, compute_features,
    FEATURE_COLUMNS,
)
from src.model_training import train_fraud_model
from src.paths import get_demo_base_dir

BASE_DIR = get_demo_base_dir()


def prepare_training_features(raw_dir: str, output_path: str) -> str:
    """Load raw data, compute features, save feature-engineered Parquet."""
    print("Loading raw data...")
    txn = pd.read_parquet(os.path.join(raw_dir, "transactions.parquet"))
    user_agg = pd.read_parquet(os.path.join(raw_dir, "user_aggregates.parquet"))
    merchant_agg = pd.read_parquet(os.path.join(raw_dir, "merchant_aggregates.parquet"))

    print(f"  Transactions: {len(txn):,}")
    print("Computing features on training data...")

    # Convert to dict-of-arrays (same format as Ray Data batches)
    batch = {col: txn[col].values for col in txn.columns}

    # Join aggregates
    batch = join_user_features(batch, user_agg)
    batch = join_merchant_features(batch, merchant_agg)

    # Compute features
    batch = compute_features(batch)

    # Build output DataFrame with features + label
    keep_cols = FEATURE_COLUMNS + ["is_fraud", "transaction_id"]
    df_features = pd.DataFrame({col: batch[col] for col in keep_cols})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    print(f"  Feature-engineered data saved → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--model-output", default=None)
    args = parser.parse_args()

    raw_dir = args.raw_dir or f"{BASE_DIR}/raw/{args.scale}"
    model_output = args.model_output or f"{BASE_DIR}/model/fraud_model.json"
    features_path = f"{BASE_DIR}/features/{args.scale}/train_features.parquet"

    # Step 1: Feature-engineer training data
    prepare_training_features(raw_dir, features_path)

    # Step 2: Train model
    metrics = train_fraud_model(features_path, model_output)
    print(f"\nTraining complete. AUC-ROC: {metrics['auc_roc']}")


if __name__ == "__main__":
    main()
