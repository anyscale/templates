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

from src.paths import resolve_artifact_paths  # noqa: E402
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

    paths = resolve_artifact_paths(args.scale, args.base_dir)
    # Sampling knobs (see configs/<scale>.yaml for what each one means).
    preset = load_scale(args.scale, args.scale_config)["tokenize"]
    seq_len = preset["seq_len"]

    with open(paths["splits"]) as f:
        splits = json.load(f)
    normal_keep = eval_normal_keep(splits, preset["target_eval_samples"])
    print(
        f"[02] temporal split train<{splits['train_end']} val<{splits['val_end']} | "
        f"~{int(splits['fraud_rate'] * splits['n_transactions']):,} frauds, "
        f"normal_keep={normal_keep:.4f}"
    )

    ray.init(ignore_reinit_error=True)
    ds = ray.data.read_parquet(paths["raw"])
    tokenized = tokenize_dataset(
        ds,
        seq_len,
        train_end=splits["train_end"],
        val_end=splits["val_end"],
        normal_keep=normal_keep,
        holdout_keep=preset["holdout_keep"],
        max_pretrain_windows=preset["max_pretrain_windows"],
        num_partitions=preset["shuffle_partitions"],
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
