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

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402
from src.tokenizer import (  # noqa: E402
    PRETRAIN_DROP,
    eval_normal_keep,
    tokenize_dataset,
    write_vocab,
)


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    # Sampling knobs (see configs/<scale>.yaml for what each one means).
    preset = load_scale(args.scale, args.scale_config)["tokenize"]
    seq_len = preset["seq_len"]

    with open(paths["splits"]) as f:
        splits = json.load(f)

    # NVIDIA-protocol benchmark rows (written by 01 on tabformer data): eval
    # windows are emitted for EXACTLY these transactions. Heuristic
    # downsampling is only the fallback for synthetic data (no benchmark file).
    eval_targets, normal_keep = None, 1.0
    if os.path.exists(paths["benchmark"]):
        import pandas as pd

        bench = pd.read_parquet(paths["benchmark"], columns=["card_id", "_ts"])
        eval_targets = {
            int(c): g.to_numpy() for c, g in bench.groupby("card_id")["_ts"]
        }
        print(
            f"[02] temporal split train<{splits['train_end']} val<{splits['val_end']} | "
            f"benchmark eval targets: {len(bench):,} rows / {len(eval_targets):,} cards"
        )
    else:
        normal_keep = eval_normal_keep(splits, preset["target_eval_samples"])
        print(
            f"[02] temporal split train<{splits['train_end']} val<{splits['val_end']} | "
            f"~{int(splits['fraud_rate'] * splits['n_transactions']):,} frauds, "
            f"normal_keep={normal_keep:.4f}"
        )

    ray.init(ignore_reinit_error=True)
    ds = ray.data.read_parquet(paths["raw"])

    # Learned merchant vocab (InfoNCE path): distributed frequency scan, top-K +
    # aggregate tail. Hashed path (default) skips it.
    merchant_vocab = None
    if preset.get("merchant_vocab", "hashed") == "learned":
        from src.merchant_vocab import build_merchant_vocab
        from src.tokenizer import _RESERVED

        merchant_vocab = build_merchant_vocab(
            ds,
            top_k=preset["merchant_top_k"],
            n_aggregate=preset["merchant_aggregate"],
            base=_RESERVED,
        )
        print(
            f"[02] learned merchant vocab: {merchant_vocab['top_k']:,} top + "
            f"{merchant_vocab['n_aggregate']:,} aggregate "
            f"({merchant_vocab['coverage'] * 100:.1f}% covered)"
        )

    tokenized = tokenize_dataset(
        ds,
        seq_len,
        train_end=splits["train_end"],
        val_end=splits["val_end"],
        normal_keep=normal_keep,
        holdout_keep=preset["holdout_keep"],
        max_pretrain_windows=preset["max_pretrain_windows"],
        num_partitions=preset["shuffle_partitions"],
        merchant_vocab=merchant_vocab,
        eval_targets=eval_targets,
    ).materialize()

    # Arrow-level filters (no numpy round trip — handles empty blocks cleanly).
    from ray.data.expressions import col

    pre = tokenized.filter(expr=col("kind") == "pretrain").drop_columns(PRETRAIN_DROP)
    pre.write_parquet(paths["tokenized_pretrain"])
    ev = tokenized.filter(expr=col("kind") == "eval").drop_columns(["kind"])
    ev.write_parquet(paths["tokenized_eval"])
    write_vocab(paths["vocab"], merchant_vocab=merchant_vocab)
    print(f"[02] pretrain windows -> {paths['tokenized_pretrain']}")
    print(f"[02] eval samples -> {paths['tokenized_eval']}")
    print(f"[02] vocab -> {paths['vocab']}")


if __name__ == "__main__":
    main()
