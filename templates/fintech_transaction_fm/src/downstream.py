"""Downstream fraud classification — the headline result, NVIDIA's blueprint recipe.

Faithful reproduction of NVIDIA's transaction-FM downstream (their notebook 05,
``05_xgboost_fraud_detection``), run on our Ray pipeline. Three feature sets:

1. ``raw``    — the target transaction's NVIDIA-13 tabular fields (the "what you
   have today" baseline).
2. ``fm``     — the pretrained transaction-FM embedding of that transaction.
3. ``fusion`` — raw fields concatenated with the embedding (the product claim).

The lift of ``fusion`` over ``raw`` is the whole point: does the foundation model
add signal on top of hand-built features? On real TabFormer it does (fusion > raw).

The recipe MIRRORS NB05 exactly — this file must not invent modeling variants:

* **Three per-feature-set XGBoost param sets** (``XGB_PARAMS_RAW/EMBED/COMBINED``),
  copied verbatim from NB05. They are separately HPO-tuned; the combined set is
  deliberately regularized (``gamma=4.8``, ``min_child_weight=25.85``, low ``lr``,
  512 rounds) so the raw+embedding feature space does not overfit. Using one shared
  recipe (an earlier divergence here) collapses ``fusion`` below ``raw`` — that was a
  reimplementation bug, not a property of the data.
* **Early stopping on a validation split** — ``early_stopping_rounds=20``,
  ``eval_metric='auc'``, ``eval_set=[(X_val, y_val)]`` — exactly as NB05.
* **Balanced train sample + ``scale_pos_weight=1.0``** (the embed stage produces the
  ~10%-fraud train sample and the val/test splits, matching NB01/NB04).
* **PCA 512 -> 64** on the embedding before XGBoost (fit on train), for ``fm`` and the
  embedding half of ``fusion``.
* **OrdinalEncoder** for the string categoricals, fit on train.
* **100K stratified eval** — val/test are subsampled to ``EVAL_SAMPLES`` preserving the
  natural fraud rate (NB01's ``stratified_subsample``), so we score NVIDIA's exact
  ``val_eval``/``test_eval`` protocol, not the full holdout.

Metrics: **AUC-ROC and PR-AUC (Average Precision)**. At ~0.1% fraud AUC saturates, so
PR-AUC — and specifically the fusion-minus-raw lift — is the operative number.

The heavy fit + score runs inside a single Ray task pinned to a GPU worker, so the
embedding load and XGBoost never touch the head node.
"""

from __future__ import annotations

import json
import os

import ray

_SPLITS = ("train", "val", "test")

# NVIDIA's 13-feature raw baseline. ``MerchantName`` is the full merchant id (their
# "Merchant Name"); User/Card/Year/Month/Day are split out from card_id/timestamp.
RAW_NUMERIC = ("Amount", "User", "Card", "Year", "Month", "Day", "Hour", "MCC", "Zip", "MerchantName")
RAW_CATEGORICAL = ("UseChip", "MerchantState", "MerchantCity")  # OrdinalEncoded (train-fit)
RAW_FEATURES = list(RAW_NUMERIC + RAW_CATEGORICAL)

# NB05's three per-feature-set HPO param sets, verbatim. Do NOT collapse to one shared
# recipe: the combined set's regularization (gamma, min_child_weight, low lr) is what
# keeps fusion from overfitting. scale_pos_weight=1.0 (train sample is ~10% fraud).
XGB_PARAMS_RAW = dict(
    n_estimators=400, max_depth=8, learning_rate=0.0023, colsample_bytree=0.95,
    min_child_weight=12, subsample=0.673, reg_alpha=0.01, reg_lambda=0.001, random_state=42,
)
XGB_PARAMS_EMBED = dict(
    n_estimators=435, max_depth=12, learning_rate=0.03774, colsample_bytree=0.587,
    min_child_weight=2.61, subsample=0.569, reg_alpha=0.01364, reg_lambda=9.7e-05,
    gamma=1.7, random_state=42,
)
XGB_PARAMS_COMBINED = dict(
    n_estimators=512, max_depth=12, learning_rate=0.00305, colsample_bytree=0.768,
    min_child_weight=25.85, subsample=0.65, reg_alpha=0.01, reg_lambda=0.0001,
    gamma=4.8, random_state=42,
)
XGB_PARAMS = {"raw": XGB_PARAMS_RAW, "fm": XGB_PARAMS_EMBED, "fusion": XGB_PARAMS_COMBINED}

# NB01 evaluates on 100K stratified subsamples of val/test (natural fraud rate preserved),
# not the full holdout. eval_metric/early-stopping use val_eval; results report test_eval.
EVAL_SAMPLES = 100_000

# Columns read from the embeddings Parquet (never the token arrays).
_READ_COLS = [
    "embedding", "label", "split",
    "raw_amount", "raw_hour", "raw_mcc", "raw_ts", "raw_zip", "raw_merchant_id",
    "raw_card_id", "raw_use_chip", "raw_merchant_state", "raw_merchant_city",
]


def _build_raw(df):
    """NVIDIA's 13 raw features from the raw_* passthrough (numeric part).

    The three string categoricals are returned separately for the caller to
    OrdinalEncode (fit on train only).
    """
    import numpy as np
    import pandas as pd

    out = pd.DataFrame(index=df.index)
    out["Amount"] = df["raw_amount"].astype(np.float64)
    cid = df["raw_card_id"].to_numpy(np.int64)
    out["User"] = cid // 100
    out["Card"] = cid % 100
    ts = pd.to_datetime(df["raw_ts"].to_numpy(np.int64), unit="s")
    out["Year"] = ts.year.to_numpy()
    out["Month"] = ts.month.to_numpy()
    out["Day"] = ts.day.to_numpy()
    out["Hour"] = df["raw_hour"].astype(np.int32)
    out["MCC"] = df["raw_mcc"].astype(np.int64)
    out["Zip"] = df["raw_zip"].astype(np.float64)
    out["MerchantName"] = df["raw_merchant_id"].astype(np.int64)  # full id, passthrough
    cat = df[["raw_use_chip", "raw_merchant_state", "raw_merchant_city"]].astype(str).fillna("")
    cat.columns = list(RAW_CATEGORICAL)
    return out, cat


@ray.remote(num_cpus=2)
def _fit_and_score(embeddings_path: str, pca_dim: int, use_gpu: bool, output_dir: str) -> dict:
    """Fit raw/fm/fusion on the (already balanced-sampled) embeddings; score test.

    Also persists per-sample test scores (feature_set, label, proba) to
    ``output_dir/test_predictions.parquet`` so ROC/PR curves can be drawn offline.
    """
    import time

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import OrdinalEncoder
    import xgboost as xgb

    df = ray.data.read_parquet(embeddings_path, columns=_READ_COLS).to_pandas()
    counts = df["split"].value_counts().to_dict()
    print(f"[06] loaded {len(df):,} embeddings  splits={counts}", flush=True)
    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    # NB05 early-stops on a real validation split. If the embed stage didn't emit one
    # (mini/CI), carve the last 10% of the shuffled train rows as val so the fit path
    # is identical everywhere.
    if not val_mask.any():
        tr_idx = np.where(train_mask)[0]
        rng = np.random.RandomState(42)
        rng.shuffle(tr_idx)
        cut = tr_idx[: max(1, len(tr_idx) // 10)]
        train_mask = train_mask.copy(); val_mask = np.zeros_like(train_mask)
        train_mask[cut] = False; val_mask[cut] = True
        print(f"[06] no val split — carved {val_mask.sum():,} val rows from train", flush=True)

    # --- embedding -> PCA(pca_dim), fit on train ---
    emb = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    if pca_dim and pca_dim < emb.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42).fit(emb[train_mask])
        emb = pca.transform(emb).astype(np.float32)
        print(f"[06] PCA {512}->{pca_dim}  explained_var={pca.explained_variance_ratio_.sum():.3f}", flush=True)
    EMB = [f"emb_{i}" for i in range(emb.shape[1])]
    Xemb = pd.DataFrame(emb, columns=EMB, index=df.index)

    # --- raw features, OrdinalEncode the categoricals (fit on train) ---
    num, cat = _build_raw(df)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cat[train_mask])
    cat_enc = pd.DataFrame(enc.transform(cat), columns=list(RAW_CATEGORICAL), index=df.index)
    Xraw = pd.concat([num, cat_enc], axis=1)[RAW_FEATURES]

    full = pd.concat([Xemb, Xraw], axis=1)
    y = df["label"].astype(int).to_numpy()
    cols = {"raw": RAW_FEATURES, "fm": EMB, "fusion": EMB + RAW_FEATURES}

    # NB01's stratified_subsample: reduce val/test to EVAL_SAMPLES rows, preserving the
    # natural fraud rate (sklearn train_test_split, stratify on label, rs=42). Train is
    # the full balanced sample. This matches NVIDIA's val_eval/test_eval exactly.
    from sklearn.model_selection import train_test_split

    def _eval_subsample(mask):
        idx = np.where(mask)[0]
        if EVAL_SAMPLES >= len(idx):
            return mask
        _, keep = train_test_split(idx, test_size=EVAL_SAMPLES, stratify=y[idx],
                                   random_state=42)
        m = np.zeros_like(mask)
        m[keep] = True
        return m

    val_mask = _eval_subsample(val_mask)
    test_mask = _eval_subsample(test_mask)
    print(f"[06] eval subsample -> val={val_mask.sum():,} ({y[val_mask].mean():.4%} fraud)  "
          f"test={test_mask.sum():,} ({y[test_mask].mean():.4%} fraud)", flush=True)

    device = "cuda" if use_gpu else "cpu"
    ytr, yval, yte = y[train_mask], y[val_mask], y[test_mask]
    results, preds = {}, {}
    for name in ("raw", "fm", "fusion"):
        c = cols[name]
        # NB05's train_xgb: per-feature-set params, early stopping on val, eval_metric='auc'.
        clf = xgb.XGBClassifier(**XGB_PARAMS[name], scale_pos_weight=1.0, tree_method="hist",
                                device=device, early_stopping_rounds=20, eval_metric="auc")
        t0 = time.time()
        clf.fit(full.loc[train_mask, c], ytr,
                eval_set=[(full.loc[val_mask, c], yval)], verbose=False)
        p = clf.predict_proba(full.loc[test_mask, c])[:, 1]
        preds[name] = p
        results[name] = {
            "auc_roc": float(roc_auc_score(yte, p)),
            "pr_auc": float(average_precision_score(yte, p)),
            "best_iteration": int(clf.best_iteration),
            "s": round(time.time() - t0, 1),
        }
        print(f"[06] {name:6} AUC-ROC={results[name]['auc_roc']:.4f}  "
              f"PR-AUC={results[name]['pr_auc']:.4f}  best_iter={clf.best_iteration}  "
              f"({results[name]['s']}s)", flush=True)

    # per-sample test scores (for offline ROC/PR curves)
    os.makedirs(output_dir, exist_ok=True)
    names = list(preds)
    pd.DataFrame({
        "feature_set": np.repeat(names, len(yte)),
        "label": np.tile(yte, len(names)),
        "proba": np.concatenate([preds[n] for n in names]),
    }).to_parquet(os.path.join(output_dir, "test_predictions.parquet"), index=False)

    return {
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_fraud_rate": float(ytr.mean()),
        "test_fraud_rate": float(yte.mean()),
        "embedding_dim": int(emb.shape[1]),
        "results": results,
        "fm_lift_pr_auc": results["fm"]["pr_auc"] - results["raw"]["pr_auc"],
        "fusion_lift_pr_auc": results["fusion"]["pr_auc"] - results["raw"]["pr_auc"],
    }


def run_downstream(embeddings_path: str, output_dir: str, pca_dim: int = 64,
                   use_gpu: bool = True, **_) -> dict:
    """Fit + evaluate raw/fm/fusion on the cluster; persist a summary. The fit runs in
    one GPU worker task (embeddings load + XGBoost off the head node).

    ``embeddings_path`` must already hold the balanced train sample + eval splits
    (the embed stage does the sampling, matching NVIDIA NB04). Extra kwargs are
    accepted and ignored so stage scripts can pass legacy args (num_workers, etc.).
    """
    ray.init(ignore_reinit_error=True)
    # Pin to a GPU worker (and give it memory headroom) when use_gpu; on CPU (mini/CI)
    # request no GPU so it schedules anywhere. Either way it runs OFF the head node.
    opts = ({"num_gpus": 1, "num_cpus": 6, "memory": 48 * 1024 ** 3} if use_gpu
            else {"num_cpus": 2})
    summary = ray.get(
        _fit_and_score.options(**opts).remote(embeddings_path, pca_dim, use_gpu, output_dir))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "downstream_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary: dict) -> None:
    r = summary["results"]
    print(f"train={summary['n_train']:,} ({summary['train_fraud_rate']:.2%} fraud)  "
          f"test={summary['n_test']:,} ({summary['test_fraud_rate']:.4%} fraud)  "
          f"emb_dim={summary['embedding_dim']}")
    print(f"{'feature set':<10} {'AUC-ROC':>10} {'PR-AUC':>10}")
    print("-" * 32)
    for name, m in r.items():
        print(f"{name:<10} {m['auc_roc']:>10.4f} {m['pr_auc']:>10.4f}")
    print("-" * 32)
    print(f"FM-only PR-AUC lift vs raw:  {summary['fm_lift_pr_auc']:+.4f}")
    print(f"Fusion  PR-AUC lift vs raw:  {summary['fusion_lift_pr_auc']:+.4f}   "
          f"({'FM adds signal' if summary['fusion_lift_pr_auc'] > 0 else 'no lift'})")
