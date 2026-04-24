#!/usr/bin/env bash
set -euo pipefail

echo "=== SkyRL Template Test ==="

git clone --branch skyrl_train-v0.4.0 https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/skyrl-train/

uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir /mnt/cluster_storage/data/gsm8k

if [[ -f /mnt/cluster_storage/data/gsm8k/train.parquet ]] && [[ -f /mnt/cluster_storage/data/gsm8k/validation.parquet ]]; then
    echo "SUCCESS: Dataset preparation completed"
else
    echo "FAILED: Dataset files not found"
    exit 1
fi

echo "=== SkyRL Template Test PASSED ==="
