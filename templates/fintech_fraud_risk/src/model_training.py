"""
XGBoost fraud detection model training.
Trains on feature-engineered data, saves model artifact.
"""
import argparse
import os
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_curve,
)

from src.feature_engineering import FEATURE_COLUMNS


def train_fraud_model(
    train_features_path: str,
    model_output_path: str,
    fraud_rate: float = 0.02,
    xgb_params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    tune_report: bool = False,
    tune_report_frequency: int = 5,
) -> dict:
    """Train XGBoost fraud detection model.

    Args:
        train_features_path: Path to feature-engineered training Parquet.
        model_output_path: Path to save the trained model (.json).
        fraud_rate: Expected fraud rate for scale_pos_weight.
        xgb_params: Optional XGBoost parameter overrides.
        num_boost_round: Number of boosting rounds.
        early_stopping_rounds: Stop early if validation metric stalls.
        tune_report: Report per-iteration metrics/checkpoints to Ray Tune.
        tune_report_frequency: Report checkpoint frequency for Ray Tune.

    Returns:
        Dictionary of evaluation metrics.
    """
    print(f"Loading training data from {train_features_path}...")
    df = pd.read_parquet(train_features_path)

    # Ensure all feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df["is_fraud"].values.astype(np.float64)

    # Train/validation split (80/20)
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"  Training:   {len(X_train):,} samples ({y_train.sum():,.0f} fraud)")
    print(f"  Validation: {len(X_val):,} samples ({y_val.sum():,.0f} fraud)")

    # XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": (1 - fraud_rate) / fraud_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
    }
    if xgb_params:
        params.update(xgb_params)

    callbacks = []
    if tune_report:
        try:
            from ray.tune.integration.xgboost import TuneReportCheckpointCallback
        except ImportError as exc:
            raise ImportError(
                "ray[tune] is required for tune_report=True. "
                "Install with: pip install 'ray[tune]>=2.9'"
            ) from exc

        callbacks.append(
            TuneReportCheckpointCallback(
                metrics={"auc": "val-auc"},
                filename="model.json",
                frequency=tune_report_frequency,
            )
        )

    print("\nTraining XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=50,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
    )

    # Evaluate
    y_prob = model.predict(dval)
    y_pred = (y_prob > 0.5).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    metrics = {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "val_fraud_count": int(y_val.sum()),
        "best_iteration": model.best_iteration,
    }

    print(f"\n{'=' * 50}")
    print("  MODEL EVALUATION")
    print(f"{'=' * 50}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"{'=' * 50}")

    # Threshold analysis
    print("\nPrecision/Recall at various thresholds:")
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
        preds = (y_prob > threshold).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        flagged = preds.sum()
        print(f"  threshold={threshold:.1f}  precision={p:.3f}  recall={r:.3f}  flagged={flagged:,}")

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save_model(model_output_path)
    print(f"\nModel saved → {model_output_path}")

    # Save metrics alongside model
    metrics_path = model_output_path.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    return metrics


if __name__ == "__main__":
    from src.paths import get_demo_base_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Path to feature-engineered training Parquet")
    parser.add_argument(
        "--model-output",
        default=os.path.join(get_demo_base_dir(), "model", "fraud_model.json"),
    )
    args = parser.parse_args()
    train_fraud_model(args.train_data, args.model_output)
