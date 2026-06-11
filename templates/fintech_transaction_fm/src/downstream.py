"""Downstream fraud classification — the headline result.

We compare three feature sets with the SAME XGBoost recipe so the only variable
is the representation:

1. ``raw``    — hand-crafted aggregate features (the "what you have today" baseline)
2. ``fm``     — the FM embedding only (no hand-crafted features at all)
3. ``fusion`` — embedding concatenated with raw features (Nubank's joint fusion)

The lift of (2) and (3) over (1) is the story: a pretrained transaction FM lets
you drop or augment a hand-tuned feature pipeline.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _raw_features(raw_path: str) -> pd.DataFrame:
    """Cheap hand-crafted per-card aggregates — a stand-in for a feature pipeline."""
    df = pd.read_parquet(raw_path)
    g = df.groupby("card_id")
    feats = pd.DataFrame(
        {
            "n_txn": g.size(),
            "amount_mean": g["amount"].mean(),
            "amount_std": g["amount"].std().fillna(0.0),
            "amount_max": g["amount"].max(),
            "amount_p95": g["amount"].quantile(0.95),
            "n_categories": g["merchant_category"].nunique(),
            "n_merchants": g["merchant_id"].nunique(),
            "night_frac": g["hour"].apply(lambda h: float((h < 6).mean())),
            "weekend_frac": g["day_of_week"].apply(lambda d: float((d >= 5).mean())),
            "label": g["is_fraud"].max(),
        }
    ).reset_index()
    return feats


def _fit_eval(X_train, y_train, X_val, y_val) -> dict:
    import xgboost as xgb

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=(neg / max(pos, 1.0)),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "auc_roc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
    }


def run_downstream(embeddings_path: str, raw_path: str, output_dir: str) -> dict:
    """Train + evaluate all three feature sets; persist a metrics summary."""
    emb_df = pd.read_parquet(embeddings_path)
    emb_mat = np.vstack(emb_df["embedding"].to_numpy())
    emb_df = emb_df[["card_id", "label"]].copy()

    raw = _raw_features(raw_path)
    merged = emb_df.merge(raw.drop(columns=["label"]), on="card_id", how="inner")

    raw_cols = [c for c in raw.columns if c not in ("card_id", "label")]
    # Re-align embedding rows to merged order.
    emb_lookup = {cid: emb_mat[i] for i, cid in enumerate(emb_df["card_id"].to_numpy())}
    emb_aligned = np.vstack([emb_lookup[c] for c in merged["card_id"].to_numpy()])

    y = merged["label"].to_numpy()
    X_raw = merged[raw_cols].to_numpy(dtype=np.float32)
    X_fm = emb_aligned.astype(np.float32)
    X_fusion = np.hstack([X_fm, X_raw])

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.25, random_state=0, stratify=y)

    results = {}
    for name, X in [("raw", X_raw), ("fm", X_fm), ("fusion", X_fusion)]:
        results[name] = _fit_eval(X[tr], y[tr], X[va], y[va])

    summary = {
        "n_cards": int(len(y)),
        "fraud_rate": float(y.mean()),
        "embedding_dim": int(X_fm.shape[1]),
        "results": results,
        "fm_lift_pr_auc": results["fm"]["pr_auc"] - results["raw"]["pr_auc"],
        "fusion_lift_pr_auc": results["fusion"]["pr_auc"] - results["raw"]["pr_auc"],
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "downstream_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary: dict) -> None:
    print(f"{'feature set':<10} {'AUC-ROC':>10} {'PR-AUC':>10}")
    print("-" * 32)
    for name, r in summary["results"].items():
        print(f"{name:<10} {r['auc_roc']:>10.4f} {r['pr_auc']:>10.4f}")
    print("-" * 32)
    print(f"FM-only PR-AUC lift vs raw:  {summary['fm_lift_pr_auc']:+.4f}")
    print(f"Fusion PR-AUC lift vs raw:   {summary['fusion_lift_pr_auc']:+.4f}")
