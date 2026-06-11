"""Downstream fraud classification — the headline result.

Evaluation protocol matches NVIDIA's transaction-FM blueprint on TabFormer so
the numbers are directly comparable when run on the real data:

* **Temporal 80/10/10 split** by transaction time (cutoffs from splits.json —
  the same ones the tokenizer used). Train on the past, early-stop on val,
  report on the most recent 10%. No temporal leakage.
* **Per-transaction, last-event labels**: each sample is one target transaction
  scored from the window of history ending at it.
* **Metrics: AUC-ROC and PR-AUC (Average Precision)** — at ~0.1% fraud, AUC-ROC
  saturates and PR-AUC is the operationally meaningful number (NVIDIA frames it
  the same way).

We compare three feature sets with the SAME XGBoost recipe so the only variable
is the representation:

1. ``raw``    — the target transaction's tabular fields (the "what you have today" baseline)
2. ``fm``     — the FM embedding of the history window only (no raw fields at all)
3. ``fusion`` — embedding concatenated with raw fields (Nubank's joint fusion)

The lift of (2) and (3) over (1) is the story: a pretrained transaction FM lets
you drop or augment a hand-tuned feature pipeline.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, w_te) -> dict:
    import xgboost as xgb

    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        scale_pos_weight=(neg / max(pos, 1.0)),
        random_state=0,  # pinned: subsampled fits at 0.1% prevalence vary a LOT
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    # Weighted metrics undo the normal-downsampling: they estimate performance
    # at the natural fraud prevalence (what NVIDIA's blueprint reports), not on
    # the fraud-enriched sample we kept for compute reasons.
    return {
        "auc_roc": float(roc_auc_score(y_te, proba, sample_weight=w_te)),
        "pr_auc": float(average_precision_score(y_te, proba, sample_weight=w_te)),
        "pr_auc_sampled": float(average_precision_score(y_te, proba)),
    }


def run_downstream(embeddings_path: str, output_dir: str) -> dict:
    """Train + evaluate all three feature sets; persist a metrics summary."""
    df = pd.read_parquet(embeddings_path)
    X_fm = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    df = df.drop(columns=["embedding"])  # free the object column (GBs at `full`)

    # Raw target-transaction features (carried through tokenize -> embed).
    amt = df["raw_amount"].to_numpy(np.float64)
    X_raw = np.column_stack(
        [
            np.sign(amt) * np.log1p(np.abs(amt)),
            df["raw_hour"].to_numpy(np.float32),
            df["raw_dow"].to_numpy(np.float32),
            df["raw_mcc"].to_numpy(np.float32),
        ]
    ).astype(np.float32)
    y = df["label"].to_numpy(np.int64)
    w = df["weight"].to_numpy(np.float64)

    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}
    for s, m in masks.items():
        if m.sum() == 0:
            raise RuntimeError(
                f"split '{s}' is empty — re-run 01/02 so splits.json temporal "
                "cutoffs are written and applied during tokenization"
            )
    tr, va, te = masks["train"], masks["val"], masks["test"]

    # Assemble matrices per split instead of full-dataset (fusion would
    # otherwise duplicate the entire embedding matrix — ~8GB at `full`).
    feature_sets = {
        "raw": lambda m: X_raw[m],
        "fm": lambda m: X_fm[m],
        "fusion": lambda m: np.hstack([X_fm[m], X_raw[m]]),
    }
    results = {}
    for name, fx in feature_sets.items():
        results[name] = _fit_eval(fx(tr), y[tr], fx(va), y[va], fx(te), y[te], w[te])

    summary = {
        "protocol": (
            "temporal 80/10/10 split by transaction time; per-transaction "
            "last-event fraud labels; prevalence-weighted metrics on the "
            "held-out most-recent 10% (NVIDIA transaction-FM blueprint protocol)"
        ),
        "n_samples": {s: int(m.sum()) for s, m in masks.items()},
        "fraud_rate": {s: float(y[m].mean()) for s, m in masks.items()},
        "natural_fraud_rate": {
            s: float((w[m] * y[m]).sum() / w[m].sum()) for s, m in masks.items()
        },
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
    n = summary["n_samples"]
    nfr = summary["natural_fraud_rate"]
    print(f"protocol: {summary['protocol']}")
    print(
        f"samples  train={n['train']:,}  val={n['val']:,}  test={n['test']:,} "
        f"(natural test fraud rate {nfr['test']:.4%})"
    )
    print(f"{'feature set':<10} {'AUC-ROC':>10} {'PR-AUC':>10}")
    print("-" * 32)
    for name, r in summary["results"].items():
        print(f"{name:<10} {r['auc_roc']:>10.4f} {r['pr_auc']:>10.4f}")
    print("-" * 32)
    print(f"FM-only PR-AUC lift vs raw:  {summary['fm_lift_pr_auc']:+.4f}")
    print(f"Fusion PR-AUC lift vs raw:   {summary['fusion_lift_pr_auc']:+.4f}")
