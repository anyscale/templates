"""Faithful reproduction of NVIDIA's raw XGBoost baseline (their notebook 01).

Goal: reproduce NVIDIA's reported raw AP 0.1238 / AUC 0.9885 EXACTLY, from the
original TabFormer CSV, to confirm our data/features are sound and isolate which
recipe knobs matter. This mirrors 01_dataset_baseline.ipynb line-for-line:
temporal 80/10/10 split by calendar date, 13 raw features, OrdinalEncoder for
categoricals, a 1M balanced train sample (all fraud + sampled normal),
scale_pos_weight=1.0, their HPO params, and a 100k stratified eval.

Runs the whole thing inside a single Ray task pinned to a WORKER with a GPU, so
the 2.35 GB CSV load and the XGBoost fit never touch the head node.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray

CSV = "/mnt/cluster_storage/transaction-fm/source/card_transaction.v1.csv"

FEATURE_COLS = [
    "User", "Card", "Year", "Month", "Day", "Hour", "Amount",
    "Use Chip", "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
]
XGB_PARAMS_RAW = {
    "n_estimators": 400, "max_depth": 8, "learning_rate": 0.0023,
    "colsample_bytree": 0.95, "min_child_weight": 12, "subsample": 0.673,
    "reg_alpha": 0.01, "reg_lambda": 0.001, "random_state": 42,
}
BALANCED_TRAIN_SIZE = 1_000_000
EVAL_SAMPLES = 100_000
RS = 42


@ray.remote(num_gpus=1, num_cpus=6, memory=48 * 1024 ** 3)
def run():
    import time
    import numpy as np
    import pandas as pd
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.metrics import average_precision_score, roc_auc_score
    import xgboost as xgb

    np.random.seed(RS)
    t0 = time.time()
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    print(f"[repro] loaded {len(df):,} rows in {time.time()-t0:.0f}s cols={list(df.columns)}", flush=True)

    # date for temporal split (their find_cutoff_date on cumulative daily counts)
    date = pd.to_datetime(
        df["Year"].astype(str) + "-"
        + df["Month"].astype(str).str.zfill(2) + "-"
        + df["Day"].astype(str).str.zfill(2), format="%Y-%m-%d")

    def cutoff(ratio):
        dc = date.value_counts().sort_index().cumsum()
        return dc.index[(dc >= dc.iloc[-1] * ratio)][0]

    train_cut, test_cut = cutoff(0.8), cutoff(0.9)
    print(f"[repro] train_cut={train_cut.date()} test_cut={test_cut.date()}", flush=True)

    # feature engineering (verbatim)
    df["Hour"] = df["Time"].str.split(":", n=1, expand=True)[0].astype(int)
    df["Amount"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "").astype(float)
    df["_target"] = ((df["Is Fraud?"] == "Yes") | (df["Is Fraud?"] == "1")).astype(int)

    train_df = df[date < train_cut]
    val_df = df[(date >= train_cut) & (date < test_cut)]
    test_df = df[date >= test_cut]
    for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[repro] {n}: {len(d):,} rows fraud={d['_target'].sum():,} "
              f"({d['_target'].mean():.4%})", flush=True)

    def balanced(d, total):
        f = d.index[d["_target"] == 1].to_numpy()
        nn = d.index[d["_target"] == 0].to_numpy()
        nf = min(len(f), int(total * 0.1))
        nnrm = min(len(nn), total - nf)
        idx = np.concatenate([np.random.choice(f, nf, replace=False),
                              np.random.choice(nn, nnrm, replace=False)])
        np.random.shuffle(idx)
        s = d.loc[idx]
        return s[FEATURE_COLS].reset_index(drop=True), s["_target"].values

    def strat(d, n):
        if n >= len(d):
            return d[FEATURE_COLS], d["_target"].values
        _, X, _, y = train_test_split(d[FEATURE_COLS], d["_target"],
                                      test_size=n, stratify=d["_target"], random_state=RS)
        return X, y.values

    Xtr, ytr = balanced(train_df, BALANCED_TRAIN_SIZE)
    Xva, yva = strat(val_df, EVAL_SAMPLES)
    Xte, yte = strat(test_df, EVAL_SAMPLES)
    print(f"[repro] train={Xtr.shape} fraud={ytr.mean():.4%} | val={Xva.shape} "
          f"fraud={yva.mean():.4%} | test={Xte.shape} fraud={yte.mean():.4%}", flush=True)

    pre = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         make_column_selector(dtype_include=["object", "category"])),
        remainder="passthrough")
    Xtr_e = pre.fit_transform(Xtr)
    Xva_e = pre.transform(Xva)
    Xte_e = pre.transform(Xte)

    clf = xgb.XGBClassifier(**XGB_PARAMS_RAW, scale_pos_weight=1.0, tree_method="hist",
                            device="cuda", early_stopping_rounds=20, eval_metric="auc")
    t0 = time.time()
    clf.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)], verbose=False)
    pte = clf.predict_proba(Xte_e)[:, 1]
    m = {"test_auc": float(roc_auc_score(yte, pte)),
         "test_ap": float(average_precision_score(yte, pte)),
         "best_iter": int(clf.best_iteration), "fit_s": round(time.time() - t0, 1)}
    print(f"[repro] fit {m['fit_s']}s best_iter={m['best_iter']}", flush=True)
    print(f"[repro] RESULT  test AUC={m['test_auc']:.4f}  AP={m['test_ap']:.4f}   "
          f"(NVIDIA raw: AUC 0.9885 / AP 0.1238)", flush=True)
    return m


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    print(ray.get(run.remote()), flush=True)
