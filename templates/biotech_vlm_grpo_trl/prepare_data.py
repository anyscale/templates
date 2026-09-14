"""
Ray Data preprocessing: nct_crc_dataset.py parquet  ->  TRL-ready parquet, once.

    uv run --frozen python prepare_data.py --data_dir /mnt/cluster_storage/data/nct_crc

Reads  <data_dir>/{train,val}.parquet   (image as base64 JPEG inside the prompt)
Writes <data_dir>/trl/{train,val}/*.parquet with columns
       prompt        JSON string: chat messages with an {"type": "image"} slot in the user turn
       image         JPEG bytes
       ground_truth  class name

Why: without this, every Ray Train worker decodes all 1,998 base64 images itself
(~50 s each, 4x duplicated). With it, the decode runs once as a Ray Data pipeline over
the cluster's CPUs and workers load the result in seconds. `train_grpo_trl.build_dataset()`
uses the prepared parquet when it exists and falls back to the raw one otherwise.

Why not stream shards into the trainer: TRL's GRPO needs every DDP rank to hold the
same dataset (each prompt is repeated `num_generations` times and the global batch is
sliced across ranks so a group's completions line up). Per-worker shards would silently
mis-group rewards. So Ray Data is the preprocessing stage here, not the dataloader.
This is also the shape a slide-tile pipeline takes at scale: tile/decode/normalise once
with Ray Data, train from the materialised result.
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import ray


def to_trl_row(row: dict) -> dict:
    """One raw parquet row -> one TRL-ready row. Same logic as train_grpo_trl.to_trl_rows."""
    messages, image = [], None
    for msg in row["prompt"]:
        content = msg["content"]
        if isinstance(content, str):
            content = json.loads(content)
        parts = []
        for part in content:
            if isinstance(part, str):
                part = json.loads(part)
            if part["type"] == "image_url":
                image = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                parts.append({"type": "image", "text": None})
            elif part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
        messages.append({"role": msg["role"], "content": parts})
    return {"prompt": json.dumps(messages), "image": image, "ground_truth": row["reward_spec"]["ground_truth"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/mnt/cluster_storage/data/nct_crc")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    args = p.parse_args()

    ray.init()
    # Keep the original row order and write one file, so a resumed run sees the same
    # dataset indices as the run that wrote the checkpoint.
    ray.data.DataContext.get_current().execution_options.preserve_order = True
    for split in args.splits:
        src = os.path.join(args.data_dir, f"{split}.parquet")
        dst = os.path.join(args.data_dir, "trl", split)
        ds = ray.data.read_parquet(src).map(to_trl_row).repartition(1)
        ds.write_parquet(dst)
        n = ray.data.read_parquet(dst).count()
        print(f"[prepare_data] {split}: {n} rows -> {dst}")


if __name__ == "__main__":
    main()
