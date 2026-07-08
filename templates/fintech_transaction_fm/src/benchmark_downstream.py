"""Stage 05 — fraud detection on the benchmark rows: the headline table.

Evaluates on the exact rows stage 01 sampled with NVIDIA's notebook-01
protocol (``benchmark.parquet``: 1M balanced train + 100k stratified
val/test):

* ``baseline``        — their 13 raw features, ordinal-encoded
  (XGB_PARAMS_RAW). Runs without embeddings; reproduces their published
  0.9885 / 0.1238.
* ``embed_pca64_xgb`` — THEIR notebook-05 protocol on OUR embeddings:
  PCA to 64d, then XGBoost (XGB_PARAMS_EMBED).
* ``embed_logistic`` / ``embed_xgb`` — our readout: the raw embedding, no
  PCA, into a linear head / XGBoost. This is the headline row.

Embedding rows are joined back to benchmark rows on (card_id, raw_ts,
amount-cents) — the eval windows were emitted for exactly these keys, so the
join is ~exact. When embeddings are used, ALL models run on the matched
subset so the comparison is row-identical.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .nvidia_baseline import (
    FEATURE_COLS,
    NVIDIA_REFERENCE,
    XGB_PARAMS_EMBED,
    XGB_PARAMS_RAW,
    fit_eval,
    make_encoder,
    pca_embeddings,
)


def _load_embeddings(embeddings_path: str):
    """Stream embedding shards into (keys DataFrame, float32 matrix)."""
    import pandas as pd
    import pyarrow.dataset as pads

    dset = pads.dataset(embeddings_path, format="parquet")
    embedding_column = "embedding"
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


def _logistic_probe(X, y, masks, device="cpu", epochs=6, lr=1e-3, seed=0):
    """Linear head on the raw embedding: z-scored, pos-weighted BCE, best
    epoch selected on val ROC-AUC (mirroring XGBoost's early stopping)."""
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    tr, va, te = (masks[s] for s in ("train", "val", "test"))
    X64 = X.astype(np.float64)
    mu, sd = np.nanmean(X64[tr], axis=0), np.nanstd(X64[tr], axis=0) + 1e-6

    def _z(m):
        z = np.nan_to_num((X64[m] - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(np.clip(z, -10, 10), dtype=torch.float32)

    Xt = {k: _z(m) for k, m in {"tr": tr, "va": va, "te": te}.items()}
    ytr = torch.as_tensor(y[tr], dtype=torch.float32)
    torch.manual_seed(seed)
    net = torch.nn.Linear(X.shape[1], 1).to(device)
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / ytr.sum()], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(ytr)
    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 8192):
            idx = perm[i : i + 8192]
            opt.zero_grad()
            loss = lossf(net(Xt["tr"][idx].to(device)).squeeze(-1), ytr[idx].to(device))
            loss.backward()
            opt.step()
        with torch.no_grad():
            va_p = net(Xt["va"].to(device)).squeeze(-1).cpu().numpy()
        auc = roc_auc_score(y[va], va_p)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        te_p = net(Xt["te"].to(device)).squeeze(-1).cpu().numpy()
    return {
        "auc_roc": float(roc_auc_score(y[te], te_p)),
        "ap": float(average_precision_score(y[te], te_p)),
        "val_auc_roc": float(best_auc),
        "n_features": int(X.shape[1]),
    }


def run_benchmark(
    benchmark_path: str,
    output_dir: str,
    embeddings_path: str | None = None,
    device: str = "cpu",
) -> dict:
    import pandas as pd

    bench = pd.read_parquet(benchmark_path)
    emb_row = None
    if embeddings_path:
        keys, X_emb = _load_embeddings(embeddings_path)
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
        print("[05] training embed_pca64_xgb (their protocol: 64d PCA, XGB_PARAMS_EMBED) ...")
        results["embed_pca64_xgb"] = fit_eval(
            Ep["train"], y["train"], Ep["val"], y["val"], Ep["test"], y["test"],
            params=XGB_PARAMS_EMBED, device=device,
        )
        print("[05] training embed_logistic (raw embedding, no PCA, linear head) ...")
        E_full = X_emb[emb_row]
        yb = bench["_target"].to_numpy().astype(np.int64)
        results["embed_logistic"] = _logistic_probe(E_full, yb, masks, device=device)
        print("[05] training embed_xgb (raw embedding, no PCA, XGB_PARAMS_EMBED) ...")
        Ef = {s: E_full[m] for s, m in masks.items()}
        results["embed_xgb"] = fit_eval(
            Ef["train"], y["train"], Ef["val"], y["val"], Ef["test"], y["test"],
            params=XGB_PARAMS_EMBED, device=device,
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
            **({"embed_pca64_xgb": XGB_PARAMS_EMBED, "embed_xgb": XGB_PARAMS_EMBED}
               if embeddings_path else {}),
        },
        "embedding_dim": int(X_emb.shape[1]) if embeddings_path else None,
        "pca_explained_variance": pca_explained,
        "results": results,
    }
    for name in ("embed_pca64_xgb", "embed_logistic", "embed_xgb"):
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
    print(f"\n{'model':<16} {'ROC-AUC':>10} {'AP':>10}   (test, 100k stratified)")
    print("-" * 50)
    for name, r in summary["results"].items():
        print(f"{name:<16} {r['auc_roc']:>10.4f} {r['ap']:>10.4f}")
    print(f"{'nvidia base':<16} {ref['baseline']['auc_roc']:>10.4f} {ref['baseline']['ap']:>10.4f}")
    print(f"{'nvidia fusion':<16} {ref['combined']['auc_roc']:>10.4f} {ref['combined']['ap']:>10.4f}")
    for name in ("embed_pca64_xgb", "embed_logistic", "embed_xgb"):
        k = f"{name}_lift_ap_pct"
        if k in summary:
            print(f"  {name} vs baseline: AP {summary[k]:+.2f}%  "
                  f"ROC-AUC {summary[f'{name}_lift_auc_pct']:+.2f}%  "
                  f"(NVIDIA's fusion: AP +41.8%)")
