"""Step 2 — distributed static/dynamic tokenization with Ray Data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402
from src.tokenizer import SEQ_LEN_BY_SCALE, tokenize_dataset, write_vocab  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALE_MAP), default="small")
    p.add_argument("--base-dir", default=None)
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    seq_len = SEQ_LEN_BY_SCALE[args.scale]

    ray.init(ignore_reinit_error=True)
    ds = ray.data.read_parquet(paths["raw"])
    tokenized = tokenize_dataset(ds, seq_len)
    tokenized.write_parquet(paths["tokenized"])
    write_vocab(paths["vocab"])
    print(f"[02] tokenized sequences -> {paths['tokenized']}")
    print(f"[02] vocab -> {paths['vocab']}")


if __name__ == "__main__":
    main()
