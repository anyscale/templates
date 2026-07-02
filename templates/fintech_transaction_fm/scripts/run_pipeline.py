"""Run the full transaction-FM pipeline end to end in ONE Ray driver.

Unlike the step-by-step scripts (01-05, which read/write Parquet at every
boundary so each stage is independently runnable), this orchestrator keeps the
intermediates in the cluster:

* tokenized **pretrain windows** are materialized in the object store and
  handed to Ray Train as a live Dataset — no Parquet round-trip, and each
  epoch iterates from memory instead of re-reading shared storage;
* tokenized **eval windows** flow straight from the CPU tokenizer tasks into
  the GPU embedding actors in one Ray Data topology — the group-by-card
  shuffle stages its blocks through the object store, never a Parquet write.

Only the durable artifacts hit storage: raw data (+ splits.json), vocab, the
model checkpoint, the embeddings (the product), and downstream metrics +
per-sample test scores.

Usage:
    python scripts/run_pipeline.py                      # smoke (CPU)
    python scripts/run_pipeline.py --scale small        # full TabFormer, GPU
    python scripts/run_pipeline.py --source synthetic   # offline data source

Also the Anyscale Job entrypoint (see job_config.yaml).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def fresh_artifact_dirs(base: str, scale: str) -> None:
    """Remove every stage output for this scale before running.

    Ray's write_parquet APPENDS into an existing directory, and a job retry
    reuses the same cluster storage — leftovers from a previous attempt
    silently double every downstream dataset. Only the scale-independent
    source/ download cache survives; stage [1] rebuilds the raw data from it.
    """
    import shutil

    for key, path in artifact_paths(base, scale).items():
        if key == "source":
            continue
        path = path.rstrip("/")
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            continue
        print(f"[clean] removed stale {path}", flush=True)


def ensure_raw_data(paths: dict, num_cards: int, source: str, seed: int = 42) -> None:
    """Stage [1]: prepare raw transactions + temporal splits (skip if cached)."""
    if os.path.exists(paths["raw"]) and os.path.exists(paths["splits"]):
        print(f"[1/6] raw data exists -> {paths['raw']} (skipping)", flush=True)
        return
    if source == "tabformer":
        from src.tabformer import prepare_tabformer

        prepare_tabformer(
            paths["raw"],
            paths["splits"],
            num_cards=num_cards,
            seed=seed,
            source_dir=paths["source"],
        )
    else:
        from src.generate_data import save_dataset

        save_dataset(paths["raw"], num_cards=num_cards, seed=seed)


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p, default="smoke")
    p.add_argument("--source", choices=["tabformer", "synthetic"], default="tabformer")
    p.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="reuse existing stage outputs instead of starting fresh — unsafe "
        "for full reruns (parquet stages append), only for debugging",
    )
    args = p.parse_args()

    cfg = load_scale(args.scale, args.scale_config)
    base = get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    if not args.keep_artifacts:
        fresh_artifact_dirs(base, args.scale)

    import ray

    from src.tokenizer import (
        PRETRAIN_DROP,
        eval_normal_keep,
        tokenize_dataset,
        write_vocab,
    )

    ray.init(ignore_reinit_error=True)
    from ray.data.expressions import col

    print("=== [1/6] data ===", flush=True)
    ensure_raw_data(paths, cfg["data"]["num_cards"], args.source)
    with open(paths["splits"]) as f:
        splits = json.load(f)
    tk = cfg["tokenize"]
    normal_keep = eval_normal_keep(splits, tk["target_eval_samples"])

    # Learned merchant vocab (InfoNCE path): one distributed frequency scan over
    # the raw transactions, keeping the top-K merchants + aggregate buckets for
    # the tail. The hashed path (default, smoke/CI) skips this entirely.
    merchant_vocab = None
    if tk.get("merchant_vocab", "hashed") == "learned":
        from src.merchant_vocab import build_merchant_vocab
        from src.tokenizer import _RESERVED

        merchant_vocab = build_merchant_vocab(
            # The frequency scan only needs one column — reader-level pruning
            # keeps the wide rows out of the read entirely.
            ray.data.read_parquet(paths["raw"], columns=["merchant_id"]),
            top_k=tk["merchant_top_k"],
            n_aggregate=tk["merchant_aggregate"],
            base=_RESERVED,
        )
        print(
            f"[1/6] learned merchant vocab: {merchant_vocab['top_k']:,} top + "
            f"{merchant_vocab['n_aggregate']:,} aggregate buckets "
            f"({merchant_vocab['n_unique']:,} unique merchants, "
            f"{merchant_vocab['coverage'] * 100:.1f}% covered by top-K)",
            flush=True,
        )
    write_vocab(paths["vocab"], merchant_vocab=merchant_vocab)

    def tokenized(emit: str):
        return tokenize_dataset(
            ray.data.read_parquet(paths["raw"]),
            tk["seq_len"],
            train_end=splits["train_end"],
            val_end=splits["val_end"],
            normal_keep=normal_keep,
            holdout_keep=tk["holdout_keep"],
            max_pretrain_windows=tk["max_pretrain_windows"],
            num_partitions=tk["shuffle_partitions"],
            emit=emit,
            merchant_vocab=merchant_vocab,
        )

    print("=== [2/6] tokenize pretrain windows -> object store ===", flush=True)
    pre = (
        tokenized("pretrain")
        .filter(expr=col("kind") == "pretrain")
        .drop_columns(PRETRAIN_DROP)
        # One global shuffle before caching: the tokenizer emits windows grouped
        # by card, and a fixed card-correlated order hurts MLM convergence. The
        # trainer adds a local shuffle buffer for per-epoch variation on top.
        .random_shuffle(seed=0)
        .materialize()
    )
    n_pretrain = pre.count()
    print(f"[2/6] {n_pretrain:,} pretrain windows materialized", flush=True)

    print("=== [3/6] pretrain ===", flush=True)
    from src.pretrain import pretrain

    pretrain(
        train_ds=pre,
        vocab_path=paths["vocab"],
        checkpoint_out=paths["checkpoint"],
        size=args.scale,
        max_len=tk["seq_len"],
        arch=cfg["model"],
        storage_base=base,
        **cfg["pretrain"],
    )
    del pre  # release the object-store blocks before the embedding pass

    print("=== [4/6] tokenize eval windows -> embed (streaming CPU->GPU) ===", flush=True)
    from src.embed import extract_embeddings

    ev = tokenized("eval").filter(expr=col("kind") == "eval").drop_columns(["kind"])
    e = cfg["embed"]
    extract_embeddings(
        ds=ev,
        checkpoint_dir=paths["checkpoint"],
        output_path=paths["embeddings"],
        num_workers=e["num_workers"],
        use_gpu=e["use_gpu"],
        batch_size=e["batch_size"],
        gpus_per_worker=e.get("gpus_per_worker"),
    )

    print("=== [5/6] downstream fraud eval ===", flush=True)
    from src.downstream import print_summary, run_downstream

    print_summary(run_downstream(paths["embeddings"], paths["downstream"]))

    # Second consumer of the same backbone: next-merchant recommendation. Needs
    # the learned merchant vocab + InfoNCE head, so it's skipped on the hashed
    # path. Re-tokenizes eval windows and streams them through the scorer.
    if merchant_vocab is not None:
        print("=== [+] next-merchant recommendation eval ===", flush=True)
        from src.recommend import print_summary as rec_print
        from src.recommend import run_recommend

        rec_ev = tokenized("eval").filter(expr=col("kind") == "eval").drop_columns(["kind"])
        rec_print(
            run_recommend(
                checkpoint_dir=paths["checkpoint"],
                output_dir=paths["downstream"],
                ds=rec_ev,
                num_workers=e["num_workers"],
                use_gpu=e["use_gpu"],
                batch_size=e["batch_size"],
                gpus_per_worker=e.get("gpus_per_worker"),
            )
        )

    print("=== [6/6] validate ===", flush=True)
    from scripts.validate_results import print_report, validate_pipeline

    # "fusion ≥ raw" only holds once the FM is really trained (small/full).
    print_report(
        validate_pipeline(
            paths,
            n_pretrain_windows=n_pretrain,
            strict_lift=args.scale in ("small", "full"),
        )
    )


if __name__ == "__main__":
    main()
