"""
Anyscale Job entrypoint for Stable Diffusion LoRA fine-tuning.

Usage:
    python scripts/run_training.py --num-workers 2 --num-epochs 3
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    ray.init(ignore_reinit_error=True)

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
