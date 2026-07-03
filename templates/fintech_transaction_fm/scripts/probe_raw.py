"""Fast raw-only XGBoost probe — validate the train_keep fix without embeddings.

Reads tokenized/<scale>/eval directly (raw_* passthrough is already there), builds
the same 14-column raw feature set as src.downstream.expand_features, and fits the
raw baseline with the distributed XGBoostTrainer on the workers. Skips the ~2h embed
stage entirely, so it isolates the question: does training the raw baseline on the
full training set (train_keep=1.0) move raw PR-AUC toward NVIDIA's 0.124?

Everything runs distributed on the A10G workers — NOTHING heavy touches the head node.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import ray  # noqa: E402
from ray.train import ScalingConfig  # noqa: E402

from src.downstream import RAW_FEATURE_COLS, evaluate, train_feature_set  # noqa: E402
from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402

# Only these columns are read from the eval parquet — never the seq_len token arrays.
_READ_COLS = [
    "label", "weight", "split", "raw_amount", "raw_hour", "raw_dow", "raw_mcc",
    "raw_ts", "raw_use_chip", "raw_merchant_state", "raw_merchant_city", "raw_zip",
    "raw_merchant_id", "raw_card_id",
]


def raw_expand(batch):
    """Build the 14 raw features (mirrors src.downstream.expand_features, no embedding)."""
    import pandas as pd

    out = pd.DataFrame(index=batch.index)
    amt = batch["raw_amount"].to_numpy(np.float64)
    out["f_log_amount"] = (np.sign(amt) * np.log1p(np.abs(amt))).astype(np.float32)
    out["raw_hour"] = batch["raw_hour"].astype(np.float32)
    out["raw_dow"] = batch["raw_dow"].astype(np.float32)
    out["raw_mcc"] = batch["raw_mcc"].astype(np.float32)
    for f in ("raw_use_chip", "raw_merchant_state", "raw_merchant_city", "raw_zip",
              "raw_merchant_id"):
        out[f] = batch[f].astype(np.float32)
    cid = batch["raw_card_id"].to_numpy(np.int64)
    out["raw_user"] = (cid // 100).astype(np.float32)
    out["raw_card"] = (cid % 100).astype(np.float32)
    ts = pd.to_datetime(batch["raw_ts"].to_numpy(np.int64), unit="s")
    out["raw_year"] = np.asarray(ts.year, dtype=np.float32)
    out["raw_month"] = np.asarray(ts.month, dtype=np.float32)
    out["raw_day"] = np.asarray(ts.day, dtype=np.float32)
    out["label"] = batch["label"].astype(np.int64)
    out["weight"] = batch["weight"].astype(np.float64)
    out["split"] = batch["split"].astype(str)
    return out


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    load_scale(args.scale, args.scale_config)  # validate config exists

    ray.init(ignore_reinit_error=True)
    ds = (
        ray.data.read_parquet(paths["tokenized_eval"], columns=_READ_COLS)
        .map_batches(raw_expand, batch_format="pandas")
        .materialize()
    )
    splits = {s: ds.filter(expr=f"split == '{s}'").materialize()
              for s in ("train", "val", "test")}
    counts = {s: d.count() for s, d in splits.items()}
    meta = splits["train"].select_columns(["label"]).to_pandas()
    pos = float(meta["label"].sum())
    neg = float(len(meta) - pos)
    print(f"[probe] train={counts['train']:,} (fraud {pos/max(len(meta),1):.4%})  "
          f"val={counts['val']:,}  test={counts['test']:,}", flush=True)

    scaling = ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu)
    storage = os.path.join(os.path.abspath(os.path.dirname(paths["tokenized_eval"])),
                           "probe_raw_results")
    os.makedirs(storage, exist_ok=True)
    cols = list(RAW_FEATURE_COLS)

    t0 = time.time()
    # sqrt(neg/pos): neg/pos itself is ~700 at natural prevalence and wrecks PR-AUC.
    scale_pos_weight = (neg / max(pos, 1.0)) ** 0.5
    booster = train_feature_set(
        splits["train"], splits["val"], cols, scaling, scale_pos_weight, storage
    )
    metrics, _ = evaluate(splits["test"], cols, booster)
    print(f"[probe] raw done in {time.time()-t0:.0f}s", flush=True)
    print(f"[probe] RAW  AUC-ROC={metrics['auc_roc']:.4f}  "
          f"PR-AUC={metrics['pr_auc']:.4f}   (NVIDIA raw: AUC 0.9885 / AP 0.1238)",
          flush=True)


if __name__ == "__main__":
    main()
