"""Stage 05 — NVIDIA notebook-05 fraud detection on the benchmark rows.

Trains XGBoost exactly as NVIDIA's ``05_xgboost_fraud_detection.ipynb``, on
the exact rows stage 01 sampled with their notebook-01 protocol
(``benchmark.parquet``: 1M balanced train + 100k stratified val/test):

* ``baseline``  — their 13 raw features, ordinal-encoded (XGB_PARAMS_RAW).
  Runs without embeddings; reproduces their published 0.9885 / 0.1238.
* ``embeddings`` — FM embeddings only, PCA'd to 64d (XGB_PARAMS_EMBED).
* ``combined``   — 13 raw + 64d PCA embeddings (XGB_PARAMS_COMBINED).

Embedding rows are joined back to benchmark rows on (card_id, raw_ts) — the
eval windows were emitted for exactly these keys, so the join is ~exact.
When embeddings are used, ALL models run on the matched subset so the
comparison is row-identical.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .nvidia_baseline import (
    FEATURE_COLS,
    NVIDIA_REFERENCE,
    XGB_PARAMS_COMBINED,
    XGB_PARAMS_EMBED,
    XGB_PARAMS_RAW,
    fit_eval,
    make_encoder,
    pca_embeddings,
)


def _load_embeddings(embeddings_path: str, embedding_column: str | None = None):
    """Stream embedding shards into (keys DataFrame, float32 matrix).

    ``embedding_column`` picks a pooling variant (extraction writes one column
    per readout — embedding_last/mean/max — from a single forward pass).
    Default: "embedding" (the extraction's default pooling) when present,
    else "embedding_last".
    """
    import pandas as pd
    import pyarrow.dataset as pads

    dset = pads.dataset(embeddings_path, format="parquet")
    if embedding_column is None:
        names = dset.schema.names
        embedding_column = "embedding" if "embedding" in names else "embedding_last"
    print(f"[05] embedding column: {embedding_column}")
    n = dset.count_rows()
    X = None
    cid = np.empty(n, np.int64)
    ts = np.empty(n, np.int64)
    amt = np.empty(n, np.float64)
    i = 0
    cols = [embedding_column, "card_id", "raw_ts", "raw_amount"]
    for batch in dset.to_batches(columns=cols, batch_size=32_768):
        m = batch.num_rows
        if m == 0:
            continue
        emb = batch.column(embedding_column)
        if hasattr(emb, "storage"):
            emb = emb.storage  # unwrap Ray tensor extension array
        flat = emb.flatten().to_numpy(zero_copy_only=False)
        if X is None:
            X = np.empty((n, len(flat) // m), np.float32)
        X[i : i + m] = flat.reshape(m, -1)
        cid[i : i + m] = batch.column("card_id").to_numpy(zero_copy_only=False)
        ts[i : i + m] = batch.column("raw_ts").to_numpy(zero_copy_only=False)
        amt[i : i + m] = batch.column("raw_amount").to_numpy(zero_copy_only=False)
        i += m
    assert i == n, f"read {i} embedding rows, expected {n}"
    # Amount in cents disambiguates same-card same-minute key collisions
    # (TabFormer time is HH:MM). Those bursts are disproportionately FRAUD, so
    # a (card_id, ts)-only join silently drops the most informative training
    # rows. float32 raw_amount is exact to the cent through ~$40k.
    keys = pd.DataFrame({
        "card_id": cid, "_ts": ts,
        "_amt_cents": np.round(amt * 100).astype(np.int64),
        "_row": np.arange(n),
    })
    return keys, X


def run_benchmark(
    benchmark_path: str,
    output_dir: str,
    embeddings_path: str | None = None,
    device: str = "cpu",
    embedding_column: str | None = None,
) -> dict:
    import pandas as pd

    bench = pd.read_parquet(benchmark_path)
    emb_row = None
    if embeddings_path:
        keys, X_emb = _load_embeddings(embeddings_path, embedding_column)
        join_on = ["card_id", "_ts", "_amt_cents"]
        keys = keys.drop_duplicates(join_on, keep="first")
        n0 = len(bench)
        bench["_amt_cents"] = np.round(bench["Amount"].to_numpy() * 100).astype(np.int64)
        bench = bench.drop_duplicates(join_on, keep="first").merge(
            keys, on=join_on, how="inner"
        )
        matched = len(bench) / n0
        print(f"[05] embeddings joined: {len(bench):,}/{n0:,} benchmark rows ({matched:.2%})")
        if matched < 0.999:
            raise RuntimeError(
                f"only {matched:.2%} of benchmark rows have an embedding — "
                "stage 02/04 must run on the same benchmark.parquet (re-run them)"
            )
        emb_row = bench.pop("_row").to_numpy()
        bench = bench.drop(columns=["_amt_cents"])

    y = {s: g["_target"].to_numpy() for s, g in bench.groupby("split")}
    for s in ("train", "val", "test"):
        if s not in y:
            raise RuntimeError(f"benchmark split '{s}' is empty — re-run stage 01")
    masks = {s: (bench["split"] == s).to_numpy() for s in ("train", "val", "test")}
    print(f"[05] rows: " + "  ".join(
        f"{s}={m.sum():,} (fraud {bench.loc[m, '_target'].mean():.4%})"
        for s, m in masks.items()
    ))

    # Their notebook's encoding: OrdinalEncoder for strings, numeric passthrough.
    pre = make_encoder()
    Xf = {s: bench.loc[m, FEATURE_COLS] for s, m in masks.items()}
    X_enc = {"train": pre.fit_transform(Xf["train"])}
    X_enc.update({s: pre.transform(Xf[s]) for s in ("val", "test")})

    results = {}
    print("[05] training baseline (13 raw features, XGB_PARAMS_RAW) ...")
    results["baseline"] = fit_eval(
        X_enc["train"], y["train"], X_enc["val"], y["val"], X_enc["test"], y["test"],
        params=XGB_PARAMS_RAW, device=device,
    )

    pca_explained = None
    if embeddings_path:
        E = {s: X_emb[emb_row[m]] for s, m in masks.items()}
        Ep = {}
        Ep["train"], Ep["val"], Ep["test"], pca_explained = pca_embeddings(
            E["train"], E["val"], E["test"]
        )
        print("[05] training embeddings-only (64d PCA, XGB_PARAMS_EMBED) ...")
        results["embeddings"] = fit_eval(
            Ep["train"], y["train"], Ep["val"], y["val"], Ep["test"], y["test"],
            params=XGB_PARAMS_EMBED, device=device,
        )
        print("[05] training combined (13 raw + 64d PCA, XGB_PARAMS_COMBINED) ...")
        C = {s: np.hstack([X_enc[s], Ep[s]]) for s in masks}
        results["combined"] = fit_eval(
            C["train"], y["train"], C["val"], y["val"], C["test"], y["test"],
            params=XGB_PARAMS_COMBINED, device=device,
        )

    summary = {
        "protocol": (
            "NVIDIA transaction-FM blueprint notebooks 01+05: temporal 80/10/10, "
            "1M balanced train, 100k stratified val/test, their HPO XGBoost params"
        ),
        "n_samples": {s: int(m.sum()) for s, m in masks.items()},
        "fraud_rate": {s: float(bench.loc[m, "_target"].mean()) for s, m in masks.items()},
        "nvidia_reference": NVIDIA_REFERENCE,
        # Self-documenting for the writeup: the exact recipe behind each number.
        "xgb_params": {
            "baseline": XGB_PARAMS_RAW,
            **({"embeddings": XGB_PARAMS_EMBED, "combined": XGB_PARAMS_COMBINED}
               if embeddings_path else {}),
        },
        "embedding_dim": int(X_emb.shape[1]) if embeddings_path else None,
        "pca_explained_variance": pca_explained,
        "results": results,
    }
    for name in ("embeddings", "combined"):
        if name in results:
            b, r = results["baseline"], results[name]
            summary[f"{name}_lift_ap_pct"] = (r["ap"] - b["ap"]) / b["ap"] * 100
            summary[f"{name}_lift_auc_pct"] = (r["auc_roc"] - b["auc_roc"]) / b["auc_roc"] * 100

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "benchmark_metrics.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[05] metrics -> {out}")
    return summary


def print_benchmark(summary: dict) -> None:
    ref = summary["nvidia_reference"]
    print(f"\n{'model':<12} {'ROC-AUC':>10} {'AP':>10}   (test, 100k stratified)")
    print("-" * 46)
    for name, r in summary["results"].items():
        print(f"{name:<12} {r['auc_roc']:>10.4f} {r['ap']:>10.4f}")
    print(f"{'nvidia base':<12} {ref['baseline']['auc_roc']:>10.4f} {ref['baseline']['ap']:>10.4f}")
    print(f"{'nvidia comb':<12} {ref['combined']['auc_roc']:>10.4f} {ref['combined']['ap']:>10.4f}")
    for name in ("embeddings", "combined"):
        k = f"{name}_lift_ap_pct"
        if k in summary:
            print(f"  {name} vs baseline: AP {summary[k]:+.2f}%  "
                  f"ROC-AUC {summary[f'{name}_lift_auc_pct']:+.2f}%  "
                  f"(NVIDIA combined: AP +41.8%)")
