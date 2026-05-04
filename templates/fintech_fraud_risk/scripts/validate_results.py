"""
Validate scored fraud detection output.

Usage:
  python scripts/validate_results.py --input demo_data/scored/medium/
  # workspace cluster storage: --input /mnt/cluster_storage/fintech-demo/scored/medium/
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def load_scored_data(input_path: str) -> pd.DataFrame:
    """Load scored Parquet output (directory of part files or single file)."""
    if os.path.isdir(input_path):
        parts = [f for f in os.listdir(input_path) if f.endswith(".parquet")]
        dfs = [pd.read_parquet(os.path.join(input_path, p)) for p in sorted(parts)]
        return pd.concat(dfs, ignore_index=True)
    return pd.read_parquet(input_path)


def print_risk_distribution(df: pd.DataFrame):
    """Print risk tier distribution."""
    dist = df["risk_tier"].value_counts()
    total = len(df)
    print(f"\n{'=' * 50}")
    print("  RISK TIER DISTRIBUTION")
    print(f"{'=' * 50}")
    for tier in ["low", "medium", "high", "critical"]:
        count = dist.get(tier, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {tier:<10} {count:>10,}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<10} {total:>10,}")
    print(f"{'=' * 50}")

    alerts = df["should_alert"].sum() if "should_alert" in df.columns else 0
    print(f"\n  Transactions flagged for review (critical): {alerts:,}")


def print_model_accuracy(df: pd.DataFrame):
    """Print model accuracy if ground-truth labels are present."""
    if "is_fraud" not in df.columns or "fraud_probability" not in df.columns:
        print("\n  (Ground-truth labels not in output — skipping accuracy metrics)")
        return

    y_true = df["is_fraud"].values.astype(int)
    y_prob = df["fraud_probability"].values

    auc = roc_auc_score(y_true, y_prob)
    print(f"\n{'=' * 50}")
    print("  MODEL ACCURACY ON SCORED DATA")
    print(f"{'=' * 50}")
    print(f"  AUC-ROC:  {auc:.4f}")

    for threshold in [0.1, 0.3, 0.5, 0.8]:
        y_pred = (y_prob > threshold).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        print(f"  threshold={threshold:.1f}  precision={p:.3f}  recall={r:.3f}  flagged={y_pred.sum():,}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to scored output directory")
    args = parser.parse_args()

    df = load_scored_data(args.input)
    print(f"Loaded {len(df):,} scored transactions")
    print_risk_distribution(df)
    print_model_accuracy(df)


if __name__ == "__main__":
    main()
