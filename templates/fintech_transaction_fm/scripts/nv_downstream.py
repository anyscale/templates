"""NVIDIA-recipe downstream on OUR Ray pipeline — raw / fm / fusion.

Ports NVIDIA's NB05 recipe onto our embeddings so the numbers are comparable:
  * balanced train sample (all fraud + sampled normal), scale_pos_weight=1.0
  * stratified val/test eval (preserves fraud rate)
  * OrdinalEncoder for raw categoricals
  * per-feature-set HPO params (RAW / EMBED / COMBINED)

Efficiency: because training is a balanced SAMPLE and eval is a subsample, we embed
only the sampled windows (not all 24.4M). The embed (seq_len-4096 Llama forward) is
the bottleneck, so for iterating on the FM use the fast profile:

    # final, NVIDIA-faithful number (~45 min embed):
    python scripts/nv_downstream.py --tag full  --train-total 1000000 --eval-n 100000
    # fast iteration while tuning the FM (~few min embed):
    python scripts/nv_downstream.py --tag fast  --train-total 200000  --eval-n 30000
    # re-run only the XGBoost fits on an already-embedded sample (seconds):
    python scripts/nv_downstream.py --tag fast --skip-embed

The headline is **fm-only** vs NVIDIA's 0.0123 — does our FM embedding carry fraud
signal at all. fm-only needs no raw features, so it is exact; raw here is APPROXIMATE
(the tokenizer passthrough still hashes merchant_id/city — a separate fix).
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ray

from src.embed import embedding_health, extract_embeddings
from src.paths import artifact_paths, get_demo_base_dir

BASE = get_demo_base_dir()
P = artifact_paths(BASE, "full")
EVAL_DIR = P["tokenized_eval"]

# NVIDIA NB05 params (verbatim); scale_pos_weight=1.0, early_stopping 20, eval_metric auc
PARAMS = {
    "raw": dict(n_estimators=400, max_depth=8, learning_rate=0.0023, colsample_bytree=0.95,
                min_child_weight=12, subsample=0.673, reg_alpha=0.01, reg_lambda=0.001),
    "fm": dict(n_estimators=435, max_depth=12, learning_rate=0.03774, colsample_bytree=0.587,
               min_child_weight=2.61, subsample=0.569, reg_alpha=0.01364, reg_lambda=9.7e-5, gamma=1.7),
    "fusion": dict(n_estimators=512, max_depth=12, learning_rate=0.00305, colsample_bytree=0.768,
                   min_child_weight=25.85, subsample=0.65, reg_alpha=0.01, reg_lambda=0.0001, gamma=4.8),
}


def sample_windows(train_total, eval_n):
    """Balanced train + stratified val/test, drawn lazily from tokenized eval.

    Lazy filters + random_sample(fraction) — NEVER materialize or shuffle the
    24.4M x seq_len eval set (that would spill hundreds of GB of token arrays).
    random_sample is a cheap per-row Bernoulli, so sizes are approximate but rates
    are preserved.
    """
    slim = ray.data.read_parquet(EVAL_DIR, columns=["label", "split"]).materialize()
    def cnt(s, lab=None):
        d = slim.filter(expr=f"split == '{s}'")
        if lab is not None:
            d = d.filter(expr=f"label == {lab}")
        return d.count()
    tr_fraud, tr_norm = cnt("train", 1), cnt("train", 0)
    val_n, test_n = cnt("val"), cnt("test")
    nf = min(tr_fraud, int(train_total * 0.1))
    frac_tn = min(1.0, (train_total - nf) / max(tr_norm, 1))
    frac_val = min(1.0, eval_n / max(val_n, 1))
    frac_test = min(1.0, eval_n / max(test_n, 1))
    print(f"[nv] pool train(f={tr_fraud:,},n={tr_norm:,}) val={val_n:,} test={test_n:,}", flush=True)
    print(f"[nv] sampled: train frauds={nf:,} + normals~{int(frac_tn*tr_norm):,} | "
          f"val~{eval_n:,} test~{eval_n:,} (rate-preserving)", flush=True)

    full = ray.data.read_parquet(EVAL_DIR)
    tr = full.filter(expr="split == 'train'")
    return [
        tr.filter(expr="label == 1"),
        tr.filter(expr="label == 0").random_sample(frac_tn, seed=1),
        full.filter(expr="split == 'val'").random_sample(frac_val, seed=2),
        full.filter(expr="split == 'test'").random_sample(frac_test, seed=3),
    ]


@ray.remote(num_gpus=1, num_cpus=6, memory=40 * 1024 ** 3)
def fit_and_eval(emb_path):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    import xgboost as xgb

    df = ray.data.read_parquet(emb_path).to_pandas()
    print(f"[nv] embeddings sample: {len(df):,} rows  splits={df['split'].value_counts().to_dict()}", flush=True)
    emb = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    EMB = [f"emb_{i}" for i in range(emb.shape[1])]
    X = pd.DataFrame(emb, columns=EMB, index=df.index)
    cid = df["raw_card_id"].to_numpy(np.int64)
    ts = pd.to_datetime(df["raw_ts"].to_numpy(np.int64), unit="s")
    raw = pd.DataFrame(index=df.index)
    raw["Amount"] = df["raw_amount"].astype(np.float32)
    raw["User"] = cid // 100; raw["Card"] = cid % 100
    raw["Year"] = ts.year; raw["Month"] = ts.month; raw["Day"] = ts.day
    raw["Hour"] = df["raw_hour"].astype(np.int32)
    for c in ["raw_use_chip", "raw_merchant_state", "raw_merchant_city", "raw_zip",
              "raw_merchant_id", "raw_mcc"]:
        raw[c] = df[c].astype(np.float32)
    RAW = list(raw.columns)
    full = pd.concat([X, raw], axis=1)
    full["label"] = df["label"].astype(int).to_numpy()
    full["split"] = df["split"].to_numpy()

    def xy(split, cols):
        m = full["split"] == split
        return full.loc[m, cols], full.loc[m, "label"].to_numpy()

    cols = {"raw": RAW, "fm": EMB, "fusion": EMB + RAW}
    results = {}
    for name in ("raw", "fm", "fusion"):
        Xtr, ytr = xy("train", cols[name]); Xva, yva = xy("val", cols[name]); Xte, yte = xy("test", cols[name])
        clf = xgb.XGBClassifier(**PARAMS[name], scale_pos_weight=1.0, tree_method="hist",
                                device="cuda", early_stopping_rounds=20, eval_metric="auc",
                                random_state=42)
        t0 = time.time()
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p = clf.predict_proba(Xte)[:, 1]
        results[name] = {"auc": float(roc_auc_score(yte, p)), "ap": float(average_precision_score(yte, p)),
                         "best_iter": int(clf.best_iteration), "s": round(time.time() - t0, 1),
                         "n_test": int(len(yte)), "test_frauds": int(yte.sum())}
        print(f"[nv] {name:6}  AUC={results[name]['auc']:.4f}  AP={results[name]['ap']:.4f}  "
              f"(best_iter={results[name]['best_iter']}, test_frauds={results[name]['test_frauds']})", flush=True)
    print("\n[nv] === vs NVIDIA ===  raw 0.9885/0.1238 | fm 0.8775/0.0123 | fusion 0.9925/0.1755", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full", help="output subdir (separates fast/full runs)")
    ap.add_argument("--train-total", type=int, default=1_000_000)
    ap.add_argument("--eval-n", type=int, default=100_000)
    ap.add_argument("--model-dir", default=P["checkpoint"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--skip-embed", action="store_true",
                    help="reuse an existing embedded sample; re-run only the XGBoost fits")
    args = ap.parse_args()
    emb_path = f"{BASE}/nv_downstream/{args.tag}_embeddings"

    ray.init(ignore_reinit_error=True)
    if not args.skip_embed:
        if os.path.exists(emb_path):
            shutil.rmtree(emb_path)
        parts = sample_windows(args.train_total, args.eval_n)
        sample_ds = parts[0]
        for p in parts[1:]:
            sample_ds = sample_ds.union(p)
        print(f"[nv] embedding sampled windows (tag={args.tag}) ...", flush=True)
        extract_embeddings(ds=sample_ds, checkpoint_dir=args.model_dir, output_path=emb_path,
                           num_workers=args.num_workers, use_gpu=True, batch_size=args.embed_batch,
                           pooling="last")
        embedding_health(emb_path)
    print(ray.get(fit_and_eval.remote(emb_path)), flush=True)


if __name__ == "__main__":
    main()
