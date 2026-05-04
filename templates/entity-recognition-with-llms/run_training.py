"""
Run LLaMA-Factory fine-tuning on a GPU worker node using Ray.

LLaMA-Factory's built-in Ray mode (USE_RAY=1) is incompatible with Ray 2.54.0+
due to API changes in ray.nodes() and environment variable propagation issues.
This script works around those issues by launching llamafactory-cli as a Ray
remote task on a single GPU worker node.

Usage:
    python run_training.py [--config CONFIG_PATH]
"""

import argparse
import ray
import subprocess
import os
import shutil


@ray.remote(num_gpus=1, num_cpus=4, accelerator_type="L4")
def run_training(config_path: str) -> int:
    """Run LLaMA-Factory training on a GPU worker."""
    env = os.environ.copy()
    env["DISABLE_VERSION_CHECK"] = "1"
    result = subprocess.run(
        ["llamafactory-cli", "train", config_path],
        env=env,
        cwd="/home/ray/default",
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run LLaMA-Factory training on a GPU worker")
    parser.add_argument(
        "--config",
        default="/mnt/cluster_storage/viggo/lora_sft_ray.yaml",
        help="Path to training config (must be on shared storage)",
    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    # Copy config to shared storage so the worker can access it
    config_path = args.config
    if not config_path.startswith("/mnt/cluster_storage"):
        shared_path = "/mnt/cluster_storage/viggo/lora_sft_ray.yaml"
        shutil.copy2(config_path, shared_path)
        config_path = shared_path
        print(f"Copied config to shared storage: {config_path}")

    print(f"Starting single-GPU training on a worker node...")
    print(f"Config: {config_path}")

    result = ray.get(run_training.remote(config_path))
    print(f"Training finished with exit code: {result}")

    if result != 0:
        raise SystemExit(f"Training failed with exit code {result}")


if __name__ == "__main__":
    main()
