"""Readout probe: is the fraud signal IN the embeddings, or was PCA+GBM blind?

Trains 5 classifiers per embedding set on the NVIDIA benchmark rows — NO PCA
anywhere (the literature never uses it; it was a blueprint idiosyncrasy):

  1. logistic  (torch linear, z-scored, pos-weighted)   embedding only  512d
  2. mlp       (512->256->1, torch, z-scored)           embedding only  512d
  3. xgb       (their XGB_PARAMS_EMBED, raw dims)       embedding only  512d
  4. logistic+ (13 raw features + embedding)            fusion          525d
  5. xgb+      (their XGB_PARAMS_COMBINED, raw dims)    fusion          525d

Same 1M/100k/100k benchmark rows and join as stage 05. Baseline reference:
0.9875 ROC / 0.1421 AP (13 features, XGB_PARAMS_RAW).

    python scripts/probe_embeddings.py --base-dir $BASE --scale full \
        --set navy_last=$BASE/embeddings/full_navy_last:embedding \
        --set cl_last=$BASE/embeddings/full:embedding_last \
        --set cl_mean=$BASE/embeddings/full:embedding_mean
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark_downstream import _load_embeddings  # noqa: E402
from src.nvidia_baseline import (  # noqa: E402
    FEATURE_COLS,
    XGB_PARAMS_COMBINED,
    XGB_PARAMS_EMBED,
    XGB_PARAMS_RAW,
    make_encoder,
)


def _join(bench, embeddings_path, embedding_column, min_match=0.999):
    """Same (card_id, ts, amount-cents) join as run_benchmark."""
    keys, X_emb = _load_embeddings(embeddings_path, embedding_column)
    join_on = ["card_id", "_ts", "_amt_cents"]
    keys = keys.drop_duplicates(join_on, keep="first")
    b = bench.copy()
    b["_amt_cents"] = np.round(b["Amount"].to_numpy() * 100).astype(np.int64)
    b = b.drop_duplicates(join_on, keep="first").merge(keys, on=join_on, how="inner")
    matched = len(b) / len(bench)
    assert matched >= min_match, f"join only matched {matched:.2%} (need {min_match:.2%})"
    print(f"[probe] joined {len(b):,}/{len(bench):,} ({matched:.2%})")
    return b, X_emb[b.pop("_row").to_numpy()]


def _torch_probe(X, y, masks, hidden=None, device="cpu", epochs=6, lr=1e-3, seed=0):
    """Logistic (hidden=None) or 1-hidden-layer MLP, z-scored, pos-weighted."""
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    tr, va, te = masks
    # float64 stats: fusion inputs include int64-scale ids (Merchant Name ~1e18)
    # that overflow float32 moments; Zip carries NaN (XGB tolerates, torch not).
    X64 = X.astype(np.float64)
    mu, sd = np.nanmean(X64[tr], axis=0), np.nanstd(X64[tr], axis=0) + 1e-6
    def _z(m):
        z = np.nan_to_num((X64[m] - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(np.clip(z, -10, 10), dtype=torch.float32)
    Xt = {k: _z(m) for k, m in {"tr": tr, "va": va, "te": te}.items()}
    yt = {k: torch.as_tensor(y[m], dtype=torch.float32) for k, m in
          {"tr": tr, "va": va, "te": te}.items()}
    d = X.shape[1]
    net = (
        torch.nn.Linear(d, 1)
        if hidden is None
        else torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, 1),
        )
    ).to(device)
    pos_w = torch.tensor([(len(yt["tr"]) - yt["tr"].sum()) / yt["tr"].sum()], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    torch.manual_seed(seed)
    n = len(yt["tr"])
    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 8192):
            idx = perm[i : i + 8192]
            xb, yb = Xt["tr"][idx].to(device), yt["tr"][idx].to(device)
            opt.zero_grad()
            loss = lossf(net(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        with torch.no_grad():
            va_p = net(Xt["va"].to(device)).squeeze(-1).cpu().numpy()
        auc = roc_auc_score(y[va], va_p)
        if auc > best_auc:  # val-AUC early selection, mirroring XGB early stop
            best_auc = auc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        te_p = net(Xt["te"].to(device)).squeeze(-1).cpu().numpy()
    return {
        "auc_roc": float(roc_auc_score(y[te], te_p)),
        "ap": float(average_precision_score(y[te], te_p)),
        "val_auc_roc": float(best_auc),
    }


def _xgb_probe(X, y, masks, params, device="cpu"):
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    tr, va, te = masks
    model = xgb.XGBClassifier(**params, device=device)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    p = model.predict_proba(X[te])[:, 1]
    return {
        "auc_roc": float(roc_auc_score(y[te], p)),
        "ap": float(average_precision_score(y[te], p)),
        "best_iteration": int(model.best_iteration),
    }


def main():
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--scale", default="full")
    p.add_argument("--set", action="append", required=True,
                   help="name=<embeddings_path>:<embedding_column>")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="torch-probe init/shuffle seed — replication runs")
    p.add_argument("--min-match", type=float, default=0.999,
                   help="required benchmark-row join fraction; lower it for "
                        "subset e2e proofs on partial embeddings")
    args = p.parse_args()

    bench_path = f"{args.base_dir}/raw/{args.scale}/benchmark.parquet"
    bench = pd.read_parquet(bench_path)
    results = {}
    for spec in args.set:
        name, rest = spec.split("=", 1)
        path, col = rest.rsplit(":", 1)
        print(f"\n=== {name}: {path} [{col}] ===")
        b, E = _join(bench, path, col, min_match=args.min_match)
        y = b["_target"].to_numpy().astype(np.int64)
        masks = tuple((b["split"] == s).to_numpy() for s in ("train", "val", "test"))
        pre = make_encoder()
        F = pre.fit_transform(b.loc[masks[0], FEATURE_COLS])
        Fva, Fte = pre.transform(b.loc[masks[1], FEATURE_COLS]), pre.transform(b.loc[masks[2], FEATURE_COLS])
        # re-assemble full-order feature matrix for fusion
        Xf = np.zeros((len(b), F.shape[1]), np.float32)
        Xf[masks[0]], Xf[masks[1]], Xf[masks[2]] = F, Fva, Fte
        fused = np.hstack([Xf, E]).astype(np.float32)

        r = {}
        fits = [
            ("logistic", lambda: _torch_probe(E, y, masks, hidden=None, device=args.device, seed=args.seed)),
            ("mlp", lambda: _torch_probe(E, y, masks, hidden=256, device=args.device, seed=args.seed)),
            ("xgb", lambda: _xgb_probe(E, y, masks, XGB_PARAMS_EMBED, device=args.device)),
            ("logistic_fusion", lambda: _torch_probe(fused, y, masks, hidden=None, device=args.device, seed=args.seed)),
            ("xgb_fusion", lambda: _xgb_probe(fused, y, masks, XGB_PARAMS_COMBINED, device=args.device)),
            # COMBINED's depth-12/lr-0.003 was Optuna-tuned for 77 dims; for
            # low-dim features (e.g. the 7-dim surprise vector) the baseline
            # recipe is the fair fusion harness.
            ("xgb_fusion_rawparams", lambda: _xgb_probe(fused, y, masks, XGB_PARAMS_RAW, device=args.device)),
        ]
        for i, (mname, fit) in enumerate(fits, 1):
            print(f"[probe] {i}/{len(fits)} {mname}")
            r[mname] = fit()
            print(f"[probe]   -> {mname}: auc={r[mname]['auc_roc']:.4f} ap={r[mname]['ap']:.4f}")
        results[name] = r

    print(f"\n{'set':<10} {'model':<16} {'ROC-AUC':>9} {'AP':>9}   (baseline 0.9875 / 0.1421)")
    print("-" * 60)
    for name, r in results.items():
        for m, v in r.items():
            print(f"{name:<10} {m:<16} {v['auc_roc']:>9.4f} {v['ap']:>9.4f}")
    out = f"{args.base_dir}/downstream/{args.scale}_probe/probe_metrics_seed{args.seed}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[probe] -> {out}")


if __name__ == "__main__":
    main()
