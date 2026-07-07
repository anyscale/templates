"""Faithful reproduction of NVIDIA's 01_dataset_baseline.ipynb XGBoost baseline.

Reads the raw TabFormer CSV directly (NOT our normalized prep) with NVIDIA's
exact 13 FEATURE_COLS, ordinal encoding, temporal 80/10/10 split, 1M balanced
training sample, 100k stratified holdout, and their XGBoost HPO params. The
point: confirm we're on the same data by matching their Test ROC-AUC 0.9885 /
AP 0.1238 before we touch pretraining or our own encoding.

    python scripts/baseline_repro.py --csv /mnt/user_storage/transaction-fm/source/card_transaction.v1.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

FEATURE_COLS = [
    "User", "Card", "Year", "Month", "Day", "Hour", "Amount",
    "Use Chip", "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
]
RANDOM_STATE = 42
XGB_PARAMS = dict(
    n_estimators=400, max_depth=8, learning_rate=0.0023, colsample_bytree=0.95,
    min_child_weight=12, subsample=0.673, reg_alpha=0.01, reg_lambda=0.001,
    random_state=RANDOM_STATE, tree_method="hist", eval_metric="auc",
    early_stopping_rounds=20, scale_pos_weight=1.0,
)


def find_cutoff_date(df, ratio):
    daily = df.groupby("date").size().sort_index().cumsum()
    total = int(daily.iloc[-1])
    return daily[daily >= total * ratio].index[0]


def balanced_train_sample(df, n=1_000_000, rs=42):
    np.random.seed(rs)  # match NVIDIA's legacy global RNG exactly
    fraud = df.index[df["_target"] == 1].to_numpy()
    normal = df.index[df["_target"] == 0].to_numpy()
    n_fraud = min(len(fraud), int(n * 0.1))
    n_normal = min(len(normal), n - n_fraud)
    idx = np.concatenate([np.random.choice(fraud, n_fraud, replace=False),
                          np.random.choice(normal, n_normal, replace=False)])
    np.random.shuffle(idx)
    return df.loc[idx]


def stratified(df, n, rs=42):
    if n >= len(df):
        return df
    _, sub = train_test_split(df, test_size=n, stratify=df["_target"], random_state=rs)
    return sub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    t0 = time.time()
    # Infer dtypes like NVIDIA's cuDF read: Merchant Name -> int64, Zip ->
    # float64 (numeric passthrough), Merchant City/State/Use Chip -> object
    # (ordinal-encoded). Forcing str made the two high-card numerics ordinal,
    # which tanked AP. Only Amount ("$..") needs to stay string for cleaning.
    df = pd.read_csv(args.csv)
    print(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s")

    df["Hour"] = df["Time"].str.split(":", n=1, expand=True)[0].astype(int)
    df["Amount"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["_target"] = (df["Is Fraud?"].str.strip().str.lower() == "yes").astype(int)
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

    tr_cut = find_cutoff_date(df, 0.8)
    te_cut = find_cutoff_date(df, 0.9)
    train = df[df["date"] < tr_cut]
    val = df[(df["date"] >= tr_cut) & (df["date"] < te_cut)]
    test = df[df["date"] >= te_cut]
    for nm, d in [("train", train), ("val", val), ("test", test)]:
        print(f"  {nm:5s} {len(d):>12,}  fraud {d['_target'].mean():.4%}")

    train_s = balanced_train_sample(train, rs=RANDOM_STATE)
    val_s = stratified(val, 100_000, RANDOM_STATE)
    test_s = stratified(test, 100_000, RANDOM_STATE)
    print(f"train sample {len(train_s):,} (fraud {train_s['_target'].mean():.2%}), "
          f"val {len(val_s):,}, test {len(test_s):,} (fraud {test_s['_target'].mean():.4%})")

    pre = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         make_column_selector(dtype_include=["object", "category"])),
        remainder="passthrough",
    )
    Xtr = pre.fit_transform(train_s[FEATURE_COLS])
    Xva = pre.transform(val_s[FEATURE_COLS])
    Xte = pre.transform(test_s[FEATURE_COLS])

    model = xgb.XGBClassifier(**XGB_PARAMS, device=args.device)
    model.fit(Xtr, train_s["_target"].to_numpy(),
              eval_set=[(Xva, val_s["_target"].to_numpy())], verbose=False)
    proba = model.predict_proba(Xte)[:, 1]
    yte = test_s["_target"].to_numpy()
    print("\n=== BASELINE (NVIDIA 01 recipe) ===")
    print(f"Test ROC-AUC: {roc_auc_score(yte, proba):.4f}   (NVIDIA 0.9885)")
    print(f"Test AP:      {average_precision_score(yte, proba):.4f}   (NVIDIA 0.1238)")


if __name__ == "__main__":
    main()
