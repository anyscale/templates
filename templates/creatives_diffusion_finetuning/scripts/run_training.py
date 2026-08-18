"""
Anyscale Job entrypoint for Stable Diffusion LoRA fine-tuning.

Usage:
    python scripts/run_training.py --num-workers 2 --num-epochs 3
"""
import argparse
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import ray

from src.data_pipeline import load_pokemon_dataset, preprocess_batch
from src.train_lora import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Train workers get the locked deps only from here. The job config cannot use
    # `requirements:` -- that inlines this 175 KB lock into the submitted runtime_env,
    # past its 120 KB cap -- so the entrypoint installs it for the driver and this
    # covers everything Ray schedules.
    ray.init(
        runtime_env={"pip": os.path.join(REPO_ROOT, "python_depset.lock")},
        ignore_reinit_error=True,
    )

    print(f"Loading Pokemon dataset...")
    train_ds = load_pokemon_dataset()
    train_ds = train_ds.map_batches(
        preprocess_batch, batch_size=32, num_cpus=1, batch_format="numpy",
    )
    print(f"  {train_ds.count()} preprocessed samples")

    print(f"\nLaunching training: {args.num_workers} GPUs, {args.num_epochs} epochs")
    result = run_training(
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        train_ds=train_ds,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(f"\nFinal loss: {result.metrics['loss']:.4f}")
    print(f"Checkpoint: {result.checkpoint}")


if __name__ == "__main__":
    main()
