#!/usr/bin/env bash
set -euo pipefail

echo "=== SkyRL Template Test ==="

# Clone and setup SkyRL
git clone --branch skyrl_train-v0.2.0 https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/skyrl-train/

# Patch for Ray 2.54.1 compatibility
sed -i 's/"ray==2.48.0"/"ray==2.54.1"/' pyproject.toml
sed -i 's|from ray.experimental.collective.util import get_address_and_port|from ray.util.collective.collective import get_address_and_port|' skyrl_train/inference_engines/utils.py
uv lock

# Test dataset preparation (quick validation)
uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir /mnt/cluster_storage/data/gsm8k

# Verify output files exist
if [[ -f /mnt/cluster_storage/data/gsm8k/train.parquet ]] && [[ -f /mnt/cluster_storage/data/gsm8k/validation.parquet ]]; then
    echo "SUCCESS: Dataset preparation completed"
else
    echo "FAILED: Dataset files not found"
    exit 1
fi

echo "=== SkyRL Template Test PASSED ==="
