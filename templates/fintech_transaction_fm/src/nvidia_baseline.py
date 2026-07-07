"""NVIDIA transaction-FM blueprint protocol — the single source of truth.

The exact split / sampling / features / XGBoost hyperparameters of NVIDIA's
transaction-foundation-model notebooks 01 (baseline) and 05 (fraud detection
with embeddings), transcribed from their repo. The baseline recipe is
validated to reproduce their published Test ROC-AUC 0.9885 / AP 0.1238 (we
measured 0.9873 / 0.1469 on the raw CSV). Consumers import from here so the
pipeline and the repro can never diverge:

* ``scripts/baseline_repro.py`` — standalone CSV validation
* stage 01 — writes ``benchmark.parquet``: the sampled train/val/test rows
* stage 05 — trains baseline (and, with embeddings, embed-only + combined)
  on those SAME rows
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "User", "Card", "Year", "Month", "Day", "Hour", "Amount",
    "Use Chip", "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
]
RANDOM_STATE = 42
TRAIN_N = 1_000_000  # balanced train sample (10% fraud)
HOLDOUT_N = 100_000  # stratified val/test samples (natural ~0.1% fraud)
PCA_DIM = 64  # notebook 05 reduces 512d embeddings -> 64d before XGBoost

# Shared trainer settings (their train_xgb): no loss reweighting, GPU hist,
# early stopping on val AUC. Per-model params below are their Optuna results.
_XGB_COMMON = dict(
    scale_pos_weight=1.0, tree_method="hist", eval_metric="auc",
    early_stopping_rounds=20,
)
# notebook 01 baseline == notebook 05 "raw" model.
XGB_PARAMS_RAW = dict(
    n_estimators=400, max_depth=8, learning_rate=0.0023, colsample_bytree=0.95,
    min_child_weight=12, subsample=0.673, reg_alpha=0.01, reg_lambda=0.001,
    random_state=RANDOM_STATE, **_XGB_COMMON,
)
XGB_PARAMS_EMBED = dict(
    n_estimators=435, max_depth=12, learning_rate=0.03774, colsample_bytree=0.587,
    min_child_weight=2.61, subsample=0.569, reg_alpha=0.01364, reg_lambda=9.7e-05,
    gamma=1.7, random_state=RANDOM_STATE, **_XGB_COMMON,
)
XGB_PARAMS_COMBINED = dict(
    n_estimators=512, max_depth=12, learning_rate=0.00305, colsample_bytree=0.768,
    min_child_weight=25.85, subsample=0.65, reg_alpha=0.01, reg_lambda=0.0001,
    gamma=4.8, random_state=RANDOM_STATE, **_XGB_COMMON,
)
XGB_PARAMS = XGB_PARAMS_RAW  # back-compat alias (baseline_repro)

# Their published test numbers (blog).
NVIDIA_REFERENCE = {
    "baseline": {"auc_roc": 0.9885, "ap": 0.1238},
    "combined": {"auc_roc": 0.9925, "ap": 0.1755},
}


def find_cutoff_date(df, ratio):
    daily = df.groupby("date").size().sort_index().cumsum()
    total = int(daily.iloc[-1])
    return daily[daily >= total * ratio].index[0]


def balanced_train_sample(df, n=TRAIN_N, rs=RANDOM_STATE):
    np.random.seed(rs)  # match NVIDIA's legacy global RNG exactly
    fraud = df.index[df["_target"] == 1].to_numpy()
    normal = df.index[df["_target"] == 0].to_numpy()
    n_fraud = min(len(fraud), int(n * 0.1))
    n_normal = min(len(normal), n - n_fraud)
    idx = np.concatenate([np.random.choice(fraud, n_fraud, replace=False),
                          np.random.choice(normal, n_normal, replace=False)])
    np.random.shuffle(idx)
    return df.loc[idx]


def stratified(df, n, rs=RANDOM_STATE):
    from sklearn.model_selection import train_test_split

    if n >= len(df):
        return df
    _, sub = train_test_split(df, test_size=n, stratify=df["_target"], random_state=rs)
    return sub


def frame_from_normalized(n: pd.DataFrame) -> pd.DataFrame:
    """Canonical (``_normalize``d) columns -> NVIDIA FEATURE_COLS frame.

    Adds ``_target``, ``date`` (split key) and ``card_id``/``_ts`` (seconds) so
    benchmark rows can be joined back to FM embeddings on (card_id, raw_ts).
    """
    ts = pd.to_datetime(n["timestamp"])
    df = pd.DataFrame({
        "User": n["user"].to_numpy(), "Card": n["card"].to_numpy(),
        "Year": ts.dt.year.to_numpy(), "Month": ts.dt.month.to_numpy(), "Day": ts.dt.day.to_numpy(),
        "Hour": n["hour"].to_numpy(), "Amount": n["amount"].to_numpy(),
        "Use Chip": n["use_chip"].to_numpy(), "Merchant Name": n["merchant_id"].to_numpy(),
        "Merchant City": n["merchant_city"].to_numpy(), "Merchant State": n["merchant_state_raw"].to_numpy(),
        "Zip": n["zip"].to_numpy(), "MCC": n["mcc"].to_numpy(),
        "_target": n["is_fraud"].to_numpy(),
        "card_id": n["card_id"].to_numpy(),
        "_ts": ts.to_numpy().astype("datetime64[s]").astype("int64"),
    })
    df["date"] = ts.dt.normalize().to_numpy()
    return df


def split_and_sample(df, train_n=TRAIN_N, holdout_n=HOLDOUT_N) -> pd.DataFrame:
    """Temporal 80/10/10 + 1M-balanced train + stratified holdouts, exactly as
    the notebook. Returns one frame with a ``split`` column."""
    tr_cut = find_cutoff_date(df, 0.8)
    te_cut = find_cutoff_date(df, 0.9)
    train = df[df["date"] < tr_cut]
    val = df[(df["date"] >= tr_cut) & (df["date"] < te_cut)]
    test = df[df["date"] >= te_cut]
    for nm, d in [("train", train), ("val", val), ("test", test)]:
        print(f"  {nm:5s} {len(d):>12,}  fraud {d['_target'].mean():.4%}")
    train_s = balanced_train_sample(train, n=train_n, rs=RANDOM_STATE)
    val_s = stratified(val, holdout_n, RANDOM_STATE)
    test_s = stratified(test, holdout_n, RANDOM_STATE)
    print(f"train sample {len(train_s):,} (fraud {train_s['_target'].mean():.2%}), "
          f"val {len(val_s):,}, test {len(test_s):,} (fraud {test_s['_target'].mean():.4%})")
    out = pd.concat([train_s.assign(split="train"), val_s.assign(split="val"),
                     test_s.assign(split="test")], ignore_index=True)
    return out


def make_encoder():
    """OrdinalEncoder for object columns, numeric passthrough — as the notebook
    (their cuDF read infers Merchant Name / Zip numeric)."""
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.preprocessing import OrdinalEncoder

    return make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         make_column_selector(dtype_include=["object", "category"])),
        remainder="passthrough",
    )


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, params=None, device="cpu") -> dict:
    """Their train_xgb: fit w/ val early stopping, score val + test."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    model = xgb.XGBClassifier(**(params or XGB_PARAMS_RAW), device=device)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    va = model.predict_proba(Xva)[:, 1]
    te = model.predict_proba(Xte)[:, 1]
    return {
        "val_auc_roc": float(roc_auc_score(yva, va)),
        "val_ap": float(average_precision_score(yva, va)),
        "auc_roc": float(roc_auc_score(yte, te)),
        "ap": float(average_precision_score(yte, te)),
    }


def pca_embeddings(X_train, X_val, X_test, dim=PCA_DIM):
    """Notebook 05: PCA the FM embeddings before XGBoost (fit on train only)."""
    from sklearn.decomposition import PCA

    dim = min(dim, X_train.shape[1], len(X_train))
    pca = PCA(n_components=dim, random_state=RANDOM_STATE)
    Xtr = pca.fit_transform(X_train)
    print(f"  PCA {X_train.shape[1]}d -> {dim}d "
          f"(explained variance {pca.explained_variance_ratio_.sum():.2%})")
    return Xtr, pca.transform(X_val), pca.transform(X_test)
