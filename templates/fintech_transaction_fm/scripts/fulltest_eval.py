"""Full-test-period evaluation — the same protocol with sampling variance removed.

NVIDIA's protocol evaluates on a 100k stratified subsample of the test period
(~112 frauds), which makes single-draw AP noisy. Because that subsample is a
random stratified draw, evaluating on the ENTIRE test period (~2.44M rows,
~2.7k frauds) has the same expected AP with ~5x tighter intervals. This script
builds that eval end-to-end from the EXISTING trained checkpoint:

1. benchmark_fulltest.parquet — identical 1M balanced train + 100k val rows
   (same seeded protocol code), test = every test-period transaction
2. eval windows tokenized for exactly those rows (training's merchant vocab,
   from the checkpoint's vocab.json), streamed straight into
3. GPU embedding extraction -> embeddings/<scale>_fulltest
4. stage-05 headline table + test predictions -> downstream/<scale>_fulltest
   (then run scripts/bootstrap_ci.py --scale <scale>_fulltest for CIs)

    python scripts/fulltest_eval.py --scale full --base-dir $BASE --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nvidia_baseline import (  # noqa: E402
    HOLDOUT_N,
    RANDOM_STATE,
    TRAIN_N,
    balanced_train_sample,
    find_cutoff_date,
    frame_from_normalized,
    stratified,
)
from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def build_benchmark_fulltest(raw_path: str, out_path: str) -> None:
    """The notebook-01 protocol with the FULL test period kept.

    Train (1M balanced) and val (100k stratified) use the exact seeded code
    that built the canonical benchmark, so those rows are identical; only the
    test split changes from a 100k stratified sample to every row.
    """
    import pandas as pd
    import pyarrow.dataset as pads

    cols = ["card_id", "timestamp", "user", "card", "hour", "amount", "use_chip",
            "merchant_id", "merchant_city", "merchant_state_raw", "zip", "mcc", "is_fraud"]
    n = pads.dataset(raw_path, format="parquet").to_table(columns=cols).to_pandas()
    n = n.sort_values(["user", "card", "timestamp", "amount", "merchant_id"],
                      kind="mergesort").reset_index(drop=True)
    df = frame_from_normalized(n)
    tr_cut = find_cutoff_date(df, 0.8)
    te_cut = find_cutoff_date(df, 0.9)
    train = df[df["date"] < tr_cut]
    val = df[(df["date"] >= tr_cut) & (df["date"] < te_cut)]
    test = df[df["date"] >= te_cut]
    train_s = balanced_train_sample(train, n=TRAIN_N, rs=RANDOM_STATE)
    val_s = stratified(val, HOLDOUT_N, RANDOM_STATE)
    print(f"[fulltest] train {len(train_s):,} (fraud {train_s['_target'].mean():.2%})  "
          f"val {len(val_s):,}  test {len(test):,} (fraud {test['_target'].mean():.4%}, "
          f"{int(test['_target'].sum()):,} frauds)")
    bench = pd.concat([train_s.assign(split="train"), val_s.assign(split="val"),
                       test.assign(split="test")], ignore_index=True).drop(columns=["date"])
    os.makedirs(os.path.dirname(out_path.rstrip("/")), exist_ok=True)
    bench.to_parquet(out_path, index=False)
    print(f"[fulltest] {len(bench):,} benchmark rows -> {out_path}")


def main():
    import pandas as pd

    import ray

    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=8, help="GPU embed workers")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    embed_cfg = load_scale(args.scale, args.scale_config)["embed"]
    ckpt = paths["checkpoint"]
    with open(os.path.join(ckpt, "model_config.json")) as f:
        mcfg = json.load(f)
    with open(os.path.join(ckpt, "vocab.json")) as f:
        vocab = json.load(f)
    seq_len = mcfg["max_len"]

    bench_path = f"{base}/raw/{args.scale}/benchmark_fulltest.parquet"
    emb_path = f"{base}/embeddings/{args.scale}_fulltest"
    out_dir = f"{base}/downstream/{args.scale}_fulltest"

    if not os.path.exists(bench_path):
        build_benchmark_fulltest(paths["raw"], bench_path)
    else:
        print(f"[fulltest] benchmark exists -> {bench_path}")

    ray.init(ignore_reinit_error=True)

    # Tokenize eval windows for exactly the fulltest rows, with the SAME
    # merchant vocab the model was trained on (stored in its vocab.json).
    from src.tokenizer import tokenize_dataset

    with open(paths["splits"]) as f:
        splits = json.load(f)
    bench_keys = pd.read_parquet(bench_path, columns=["card_id", "_ts"])
    eval_targets = {int(c): g.to_numpy() for c, g in bench_keys.groupby("card_id")["_ts"]}
    print(f"[fulltest] eval targets: {len(bench_keys):,} rows / {len(eval_targets):,} cards")
    ds = ray.data.read_parquet(paths["raw"])
    tokenized = tokenize_dataset(
        ds, seq_len,
        train_end=splits["train_end"],
        val_end=splits["val_end"],
        num_partitions=128,
        emit="eval",  # skip pretrain-window computation entirely
        merchant_vocab=vocab.get("merchant_vocab"),
        eval_targets=eval_targets,
    )
    from ray.data.expressions import col

    ev = tokenized.filter(expr=col("kind") == "eval").drop_columns(["kind"])

    # Stream straight into GPU extraction against the existing checkpoint.
    from src.embed import extract_embeddings

    extract_embeddings(
        ds=ev,
        checkpoint_dir=ckpt,
        output_path=emb_path,
        num_workers=args.num_workers,
        use_gpu=True,
        batch_size=embed_cfg["batch_size"],
        gpus_per_worker=embed_cfg.get("gpus_per_worker"),
    )

    # The exact stage-05 table on the fulltest rows (writes test_predictions
    # for bootstrap_ci --scale <scale>_fulltest).
    from src.benchmark_downstream import print_benchmark, run_benchmark

    summary = run_benchmark(
        benchmark_path=bench_path,
        output_dir=out_dir,
        embeddings_path=emb_path,
        device=args.device,
    )
    print_benchmark(summary)


if __name__ == "__main__":
    main()
