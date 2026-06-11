"""Step 2 — distributed static/dynamic tokenization with Ray Data.

Emits two datasets (see ``src/tokenizer.py`` for the protocol):

* ``tokenized/<scale>/pretrain/`` — train-period windows for MLM pretraining
* ``tokenized/<scale>/eval/``     — per-transaction samples (window ends at the
  target txn, label = its is_fraud, split = temporal train/val/test)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402
from src.tokenizer import SEQ_LEN_BY_SCALE, tokenize_dataset, write_vocab  # noqa: E402

# Per-scale sampling: cap pretrain windows per card and target a manageable
# eval-set size (all frauds + downsampled normals). Real cards have thousands
# of transactions; without caps, smoke would tokenize millions of windows.
TOKENIZE_PRESETS = {
    "smoke": dict(target_eval_samples=30_000, max_pretrain_windows=8),
    "small": dict(target_eval_samples=150_000, max_pretrain_windows=None),
    "full": dict(target_eval_samples=400_000, max_pretrain_windows=None),
}

PRETRAIN_DROP = [
    "kind", "split", "label", "weight",
    "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALE_MAP), default="small")
    p.add_argument("--base-dir", default=None)
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    seq_len = SEQ_LEN_BY_SCALE[args.scale]
    preset = TOKENIZE_PRESETS[args.scale]

    with open(paths["splits"]) as f:
        splits = json.load(f)
    n_txn = splits["n_transactions"]
    n_fraud = splits["fraud_rate"] * n_txn
    # Keep enough normals to hit the eval target, never fewer than 4x frauds.
    normals_target = max(preset["target_eval_samples"] - n_fraud, 4 * n_fraud)
    normal_keep = float(min(1.0, normals_target / max(n_txn - n_fraud, 1.0)))
    print(
        f"[02] temporal split train<{splits['train_end']} val<{splits['val_end']} | "
        f"~{int(n_fraud):,} frauds, normal_keep={normal_keep:.4f}"
    )

    ray.init(ignore_reinit_error=True)
    ds = ray.data.read_parquet(paths["raw"])
    tokenized = tokenize_dataset(
        ds,
        seq_len,
        train_end=splits["train_end"],
        val_end=splits["val_end"],
        normal_keep=normal_keep,
        max_pretrain_windows=preset["max_pretrain_windows"],
    ).materialize()

    # Arrow-level filters (no numpy round trip — handles empty blocks cleanly).
    from ray.data.expressions import col

    pre = tokenized.filter(expr=col("kind") == "pretrain").drop_columns(PRETRAIN_DROP)
    pre.write_parquet(paths["tokenized_pretrain"])
    ev = tokenized.filter(expr=col("kind") == "eval").drop_columns(["kind"])
    ev.write_parquet(paths["tokenized_eval"])
    write_vocab(paths["vocab"])
    print(f"[02] pretrain windows -> {paths['tokenized_pretrain']}")
    print(f"[02] eval samples -> {paths['tokenized_eval']}")
    print(f"[02] vocab -> {paths['vocab']}")


if __name__ == "__main__":
    main()
