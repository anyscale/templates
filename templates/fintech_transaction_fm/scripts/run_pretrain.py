"""Pretrain-only orchestrator: steps 01 (data) -> 02 (tokenize) -> 03 (pretrain).

Runs the first three decoupled stages and stops — no embed/fraud/reco. Each
stage persists to --base-dir, so downstream eval runs as a SEPARATE job against
the saved artifacts (no re-tokenizing, no re-shuffling):

    raw/<scale>/ + splits.json                     (01)
    tokenized/<scale>/{pretrain,eval}/ + vocab     (02 — eval set saved for later)
    model/<scale>/ + ray_results/ (per-epoch ckpts) (03)

This just shells out to the existing, tested scripts/{01,02,03}_*.py in order
(they hand off through parquet on --base-dir, so separate processes are fine).

Usage (point --base-dir at durable storage so a later eval job can read it):
    python scripts/run_pretrain.py --scale xl \
        --base-dir /mnt/user_storage/transaction-fm --use-gpu
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(step: str, extra: list) -> None:
    cmd = [sys.executable, os.path.join(HERE, step), *extra]
    print(f"\n=== run_pretrain: {step} {' '.join(extra)} ===", flush=True)
    subprocess.run(cmd, check=True)  # non-zero exit aborts the chain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="xl")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--num-workers", type=int, default=None, help="GPU workers for pretrain")
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args()

    common = ["--scale", args.scale]
    if args.base_dir:
        common += ["--base-dir", args.base_dir]

    run("01_generate_data.py", common)
    run("02_tokenize.py", common)

    pre = list(common)
    if args.num_workers is not None:
        pre += ["--num-workers", str(args.num_workers)]
    if args.use_gpu:
        pre.append("--use-gpu")
    run("03_pretrain.py", pre)

    print("\n=== run_pretrain: done (data + tokenize + pretrain) ===", flush=True)


if __name__ == "__main__":
    main()
