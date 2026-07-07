"""Faithful reproduction of NVIDIA's 01_dataset_baseline.ipynb XGBoost baseline.

Reads the raw TabFormer CSV directly (NOT our normalized prep) with NVIDIA's
exact 13 FEATURE_COLS, ordinal encoding, temporal 80/10/10 split, 1M balanced
training sample, 100k stratified holdout, and their XGBoost HPO params — all
imported from ``src/nvidia_baseline.py``, the same module the pipeline uses.
The point: confirm we're on the same data by matching their Test ROC-AUC
0.9885 / AP 0.1238 before we touch pretraining or our own encoding.

    python scripts/baseline_repro.py --csv /mnt/user_storage/transaction-fm/source/card_transaction.v1.csv
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nvidia_baseline import (  # noqa: E402
    FEATURE_COLS,
    NVIDIA_REFERENCE,
    fit_eval,
    frame_from_normalized,
    make_encoder,
    split_and_sample,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--via-normalize", action="store_true",
                   help="build features by running OUR src.tabformer._normalize on the "
                        "CSV (validates our pipeline reproduces the baseline), instead of "
                        "reading the raw CSV columns directly")
    args = p.parse_args()

    t0 = time.time()
    if args.via_normalize:
        from src.tabformer import _CSV_COLUMNS, _normalize

        raw = pd.read_csv(args.csv, usecols=_CSV_COLUMNS, dtype={"Time": str, "Amount": str})
        print(f"loaded {len(raw):,} rows in {time.time()-t0:.0f}s; applying OUR _normalize ...")
        df = frame_from_normalized(_normalize(raw))
    else:
        # Infer dtypes like NVIDIA's cuDF read: Merchant Name -> int64, Zip ->
        # float64 (numeric passthrough); Merchant City/State/Use Chip -> object.
        df = pd.read_csv(args.csv)
        print(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s")
        df["Hour"] = df["Time"].str.split(":", n=1, expand=True)[0].astype(int)
        df["Amount"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
        df["_target"] = (df["Is Fraud?"].str.strip().str.lower() == "yes").astype(int)
        df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

    bench = split_and_sample(df)
    parts = {s: bench[bench["split"] == s] for s in ("train", "val", "test")}

    pre = make_encoder()
    Xtr = pre.fit_transform(parts["train"][FEATURE_COLS])
    Xva = pre.transform(parts["val"][FEATURE_COLS])
    Xte = pre.transform(parts["test"][FEATURE_COLS])
    m = fit_eval(
        Xtr, parts["train"]["_target"].to_numpy(),
        Xva, parts["val"]["_target"].to_numpy(),
        Xte, parts["test"]["_target"].to_numpy(),
        device=args.device,
    )
    ref = NVIDIA_REFERENCE["baseline"]
    print("\n=== BASELINE (NVIDIA 01 recipe) ===")
    print(f"Test ROC-AUC: {m['auc_roc']:.4f}   (NVIDIA {ref['auc_roc']})")
    print(f"Test AP:      {m['ap']:.4f}   (NVIDIA {ref['ap']})")


if __name__ == "__main__":
    main()
