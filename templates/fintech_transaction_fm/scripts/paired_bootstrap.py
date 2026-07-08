"""Paired bootstrap across context lengths — the ordering claim, done right.

Every *_fulltest eval scores the IDENTICAL test rows (same seeded benchmark
construction), so differences between context lengths should be tested
paired: resample the same row indices once per draw, score every model on
that draw, and look at the distribution of per-draw DIFFERENCES. Far tighter
than comparing marginal CIs.

    python scripts/paired_bootstrap.py --base-dir $BASE \
        --runs full_fulltest xl_fulltest xxl_fulltest --column embed_xgb
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    import pandas as pd
    from sklearn.metrics import average_precision_score

    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--runs", nargs="+", required=True,
                   help="downstream/<run> dirs holding test_predictions.parquet")
    p.add_argument("--column", default="embed_xgb")
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    proba, y = {}, None
    for r in args.runs:
        df = pd.read_parquet(f"{args.base_dir}/downstream/{r}/test_predictions.parquet")
        yy = df["y"].to_numpy()
        if y is None:
            y = yy
        else:
            assert np.array_equal(y, yy), f"{r}: test rows differ — pairing invalid"
        proba[r] = df[args.column].to_numpy()
    n = len(y)
    print(f"[paired] {n:,} shared test rows, {int(y.sum())} frauds, "
          f"{args.draws} draws, column={args.column}")

    rng = np.random.default_rng(args.seed)
    aps = {r: np.empty(args.draws) for r in args.runs}
    for d in range(args.draws):
        i = rng.integers(0, n, size=n)  # ONE index draw shared by all models
        if y[i].sum() == 0:
            for r in args.runs:
                aps[r][d] = np.nan
            continue
        for r in args.runs:
            aps[r][d] = average_precision_score(y[i], proba[r][i])

    out = {"column": args.column, "n_rows": int(n), "draws": args.draws,
           "point": {r: float(average_precision_score(y, proba[r])) for r in args.runs},
           "pairs": {}}
    print(f"\npoint estimates: " + "  ".join(f"{r}={out['point'][r]:.4f}" for r in args.runs))
    print(f"\n{'pair':<32} {'diff':>8} {'95% CI':>19} {'P(A>B)':>8}")
    print("-" * 70)
    for a, b in itertools.permutations(args.runs, 2):
        if args.runs.index(a) >= args.runs.index(b):
            continue
        diff = aps[a] - aps[b]
        diff = diff[~np.isnan(diff)]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p_gt = float((diff > 0).mean())
        out["pairs"][f"{a} - {b}"] = {
            "mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "p_a_gt_b": p_gt,
        }
        print(f"{a + ' - ' + b:<32} {diff.mean():>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}] {p_gt:>8.3f}")

    dest = f"{args.base_dir}/downstream/paired_bootstrap_{args.column}.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[paired] -> {dest}")


if __name__ == "__main__":
    main()
