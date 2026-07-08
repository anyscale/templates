"""Classical burst/velocity baseline — no FM anywhere (CLEANUP.md #4.2).

How far do cheap card-velocity features get you before you need the FM?
XGBoost on NVIDIA's 13 raw features + causal per-card aggregates (trailing
counts, spend velocity, amount deviation), computed from the full raw
transaction history and evaluated on the exact benchmark rows/protocol.
Every feature sees only the card's transactions at or before the scored row
(fraud labels never enter feature computation).

Reference points: baseline 0.9875 ROC / 0.1421 AP (13 raw), FM embed-only
0.23-0.26 AP. If this recovers most of the FM lift, the headline framing
changes.

    python scripts/velocity_baseline.py --base-dir $BASE --scale full
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nvidia_baseline import (  # noqa: E402
    FEATURE_COLS,
    XGB_PARAMS_RAW,
    fit_eval,
    make_encoder,
)

HOUR, DAY, WEEK = 3600, 86400, 7 * 86400
VELOCITY_COLS = [
    "card_txn_idx", "dt_prev_s", "cnt_1h", "cnt_24h", "cnt_7d",
    "amt_sum_24h", "amt_mean_prior", "amt_z",
]


def velocity_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Causal per-card velocity aggregates for every raw transaction.

    Sorted the same way build_benchmark sorts (chronological within card,
    mergesort for stable ties) so join keys line up deterministically.
    """
    raw = raw.sort_values(
        ["card_id", "timestamp", "amount", "merchant_id"], kind="mergesort"
    ).reset_index(drop=True)
    ts = raw["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    amt = raw["amount"].to_numpy()
    card = raw["card_id"].to_numpy()

    idx = raw.groupby("card_id", sort=False).cumcount().to_numpy()
    dt_prev = np.where(idx > 0, ts - np.roll(ts, 1), np.nan).astype(np.float64)

    # Expanding PRIOR amount stats via per-card cumsums (exclude current row).
    cs = raw.groupby("card_id", sort=False)["amount"].cumsum().to_numpy() - amt
    amt2 = raw["amount"] ** 2
    cs2 = amt2.groupby(raw["card_id"], sort=False).cumsum().to_numpy() - amt**2
    n_prior = idx.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_prior = np.where(idx > 0, cs / np.maximum(n_prior, 1), np.nan)
        var_prior = np.where(
            idx > 1, cs2 / np.maximum(n_prior, 1) - mean_prior**2, np.nan
        )
        amt_z = (amt - mean_prior) / np.sqrt(np.maximum(var_prior, 1e-12))

    # Trailing-window counts + 24h spend, per card (timestamps sorted in-group).
    cnt = {w: np.zeros(len(raw), np.float64) for w in (HOUR, DAY, WEEK)}
    sum_24h = np.zeros(len(raw), np.float64)
    starts = np.flatnonzero(np.r_[True, card[1:] != card[:-1]])
    ends = np.r_[starts[1:], len(raw)]
    for s, e in zip(starts, ends):
        ts_g, amt_g = ts[s:e], amt[s:e]
        pos = np.arange(e - s)
        csg = np.cumsum(amt_g)
        for w in (HOUR, DAY, WEEK):
            left = np.searchsorted(ts_g, ts_g - w, side="left")
            cnt[w][s:e] = pos - left + 1
            if w == DAY:
                sum_24h[s:e] = csg - np.where(left > 0, csg[left - 1], 0.0)

    out = pd.DataFrame({
        "card_id": card, "_ts": ts,
        "_amt_cents": np.round(amt * 100).astype(np.int64),
        "card_txn_idx": idx.astype(np.float64), "dt_prev_s": dt_prev,
        "cnt_1h": cnt[HOUR], "cnt_24h": cnt[DAY], "cnt_7d": cnt[WEEK],
        "amt_sum_24h": sum_24h, "amt_mean_prior": mean_prior, "amt_z": amt_z,
    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--scale", default="full")
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-match", type=float, default=0.999)
    args = p.parse_args()

    import pyarrow.dataset as pads

    raw_path = f"{args.base_dir}/raw/{args.scale}/transactions.parquet"
    bench_path = f"{args.base_dir}/raw/{args.scale}/benchmark.parquet"
    cols = ["card_id", "timestamp", "amount", "merchant_id"]  # labels NEVER read
    raw = pads.dataset(raw_path, format="parquet").to_table(columns=cols).to_pandas()
    print(f"[velocity] {len(raw):,} raw transactions; computing causal aggregates ...")
    feats = velocity_features(raw)
    del raw

    bench = pd.read_parquet(bench_path)
    join_on = ["card_id", "_ts", "_amt_cents"]
    feats = feats.drop_duplicates(join_on, keep="first")
    bench["_amt_cents"] = np.round(bench["Amount"].to_numpy() * 100).astype(np.int64)
    b = bench.drop_duplicates(join_on, keep="first").merge(feats, on=join_on, how="inner")
    matched = len(b) / len(bench)
    assert matched >= args.min_match, f"join only matched {matched:.2%}"
    print(f"[velocity] joined {len(b):,}/{len(bench):,} ({matched:.2%})")

    y = b["_target"].to_numpy().astype(np.int64)
    masks = {s: (b["split"] == s).to_numpy() for s in ("train", "val", "test")}
    pre = make_encoder()
    X13 = {"train": pre.fit_transform(b.loc[masks["train"], FEATURE_COLS])}
    for s in ("val", "test"):
        X13[s] = pre.transform(b.loc[masks[s], FEATURE_COLS])
    V = {s: b.loc[masks[s], VELOCITY_COLS].to_numpy(np.float64) for s in masks}

    results = {}
    for name, X in (
        ("raw13", X13),
        ("raw13_velocity", {s: np.hstack([X13[s], V[s]]) for s in X13}),
    ):
        print(f"[velocity] fitting {name} ({X['train'].shape[1]} features)")
        results[name] = fit_eval(
            X["train"], y[masks["train"]], X["val"], y[masks["val"]],
            X["test"], y[masks["test"]], params=XGB_PARAMS_RAW, device=args.device,
        )
        r = results[name]
        print(f"[velocity]   -> {name}: auc={r['auc_roc']:.4f} ap={r['ap']:.4f}")

    print(f"\n{'model':<18} {'ROC-AUC':>9} {'AP':>9}   "
          "(baseline 0.9875 / 0.1421; FM embed-only AP 0.23-0.26)")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:<18} {r['auc_roc']:>9.4f} {r['ap']:>9.4f}")
    out = f"{args.base_dir}/downstream/{args.scale}_velocity/velocity_metrics.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[velocity] -> {out}")


if __name__ == "__main__":
    main()
