"""Bootstrap confidence intervals for the headline table.

The 100k stratified test set holds only ~112 frauds, so a single-draw AP is
noisy (a lesson from the faithful-repro branch, where fusion AP visibly moved
across eval draws). This resamples the test rows with replacement and reports
each model's AP/ROC as point + 95% CI, plus the probability of clearing
NVIDIA's published fusion headline (0.1755).

Reads the test_predictions.parquet stage 05 writes next to benchmark_metrics.json.

    python scripts/bootstrap_ci.py --base-dir $BASE --scale full
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nvidia_baseline import NVIDIA_REFERENCE  # noqa: E402


def main():
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--scale", default="full")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    path = f"{args.base_dir}/downstream/{args.scale}/test_predictions.parquet"
    df = pd.read_parquet(path)
    y = df.pop("y").to_numpy()
    n = len(y)
    fusion_bar = NVIDIA_REFERENCE["combined"]["ap"]
    print(f"[ci] {n:,} test rows, {int(y.sum())} frauds, {args.draws} bootstrap draws")

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.draws, n))
    out = {}
    beat_col = f"P(AP>{fusion_bar:.4f})"
    print(f"\n{'model':<18} {'AP':>8} {'95% CI':>17} {beat_col:>13} {'ROC-AUC':>9}")
    print("-" * 70)
    for col in df.columns:
        proba = df[col].to_numpy()
        aps = np.empty(args.draws)
        for d in range(args.draws):
            i = idx[d]
            if y[i].sum() == 0:  # a draw with zero frauds has undefined AP
                aps[d] = np.nan
                continue
            aps[d] = average_precision_score(y[i], proba[i])
        aps = aps[~np.isnan(aps)]
        lo, hi = np.percentile(aps, [2.5, 97.5])
        point_ap = average_precision_score(y, proba)
        out[col] = {
            "ap": float(point_ap), "ap_ci95": [float(lo), float(hi)],
            "p_beats_nvidia_fusion": float((aps > fusion_bar).mean()),
            "auc_roc": float(roc_auc_score(y, proba)),
            "n_draws": int(len(aps)),
        }
        print(f"{col:<18} {point_ap:>8.4f} [{lo:>7.4f},{hi:>7.4f}] "
              f"{out[col]['p_beats_nvidia_fusion']:>13.3f} {out[col]['auc_roc']:>9.4f}")

    dest = f"{args.base_dir}/downstream/{args.scale}/bootstrap_ci.json"
    with open(dest, "w") as f:
        json.dump({"draws": args.draws, "seed": args.seed,
                   "nvidia_fusion_bar": fusion_bar, "results": out}, f, indent=2)
    print(f"\n[ci] -> {dest}")


if __name__ == "__main__":
    main()
